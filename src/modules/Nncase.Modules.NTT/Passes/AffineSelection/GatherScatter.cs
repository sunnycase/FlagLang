// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.TIR;

namespace Nncase.Passes;

public partial class NTTAffineSelectionPass
{
    public Expr SelectAffineGather(IR.Affine.Gather gather, Call call, Expr output)
    {
        var source = (Expr)call[IR.Affine.Gather.Source];
        var defaultValue = (Expr)call[IR.Affine.Gather.DefaultValue];
        var domainRank = gather.Relation.Domains.Length;
        ValidateSingleAddressRelation(gather.Relation, "Affine.Gather selection");
        ValidateRank(output, domainRank, "Affine.Gather output");

        var builder = IR.F.Affine.Grid()
            .Domain(domainRank, out var _)
            .Read(source, CreateScalarPointerMap(domainRank), out var sourcePtr);

        var defaultArg = defaultValue;
        if (defaultValue is not None && GetCheckedShape(defaultValue) is { Rank: > 0 })
        {
            ValidateRank(defaultValue, domainRank, "Affine.Gather default value");
            builder.Read(defaultValue, AffineMap.Identity(domainRank), out var defaultTile);
            defaultArg = defaultTile;
        }

        return builder
            .Write(output, AffineMap.Identity(domainRank), out var outTile)
            .Body(TIR.F.NTT.AffineGather(sourcePtr, defaultArg, outTile, gather.Relation, gather.Symbols, gather.Shape))
            .Build();
    }

    public Expr SelectAffineScatter(IR.Affine.Scatter scatter, Call call)
    {
        var source = (Expr)call[IR.Affine.Scatter.Source];
        var dest = (Expr)call[IR.Affine.Scatter.Dest];
        var domainRank = scatter.Relation.Domains.Length;
        ValidateSingleAddressRelation(scatter.Relation, "Affine.Scatter selection");
        ValidateRank(source, domainRank, "Affine.Scatter source");

        return IR.F.Affine.Grid()
            .Domain(domainRank, out var _)
            .Read(source, AffineMap.Identity(domainRank), out var sourceTile)
            .Read(dest, CreateScalarPointerMap(domainRank), out var destPtr)
            .Body(TIR.F.NTT.AffineScatter(sourceTile, destPtr, scatter.Relation, scatter.Symbols, source.CheckedShape))
            .BuildEffect();
    }

    public Expr SelectGather(IR.Tensors.Gather gather, Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.Gather.Input];
        var index = (Expr)call[IR.Tensors.Gather.Index];
        if (!TryGetRankedTensorShapes("Gather affine selection", [input, index, output], out var shapes))
        {
            return call;
        }

        var inputShape = shapes[0];
        var indexShape = shapes[1];
        var outputShape = shapes[2];
        var axis = (int)Util.PositiveIndex(gather.Axis, inputShape.Rank);
        if (axis < 0 || axis >= inputShape.Rank)
        {
            throw new NotSupportedException($"Gather affine selection requires axis in [0,{inputShape.Rank}), got {gather.Axis}.");
        }

        var expectedOutputRank = checked(inputShape.Rank - 1 + indexShape.Rank);
        if (outputShape.Rank != expectedOutputRank)
        {
            throw new InvalidOperationException($"Gather affine selection expected output rank {expectedOutputRank}, got {outputShape.Rank}.");
        }

        var domainRank = outputShape.Rank;
        var domains = IR.F.Affine.Domains(domainRank);
        var inputMap = CreateGatherInputMap(inputShape, indexShape, axis, domains);
        var indexMap = CreateGatherIndexMap(indexShape, axis, domains);

        return IR.F.Affine.Grid()
            .Domain(domainRank, out var _)
            .Read(input, inputMap, out var inputTile)
            .Read(index, indexMap, out var indexTile)
            .Write(output, AffineMap.Identity(domainRank), out var outTile)
            .Body(TIR.F.NTT.Gather(inputTile, indexTile, outTile, axis))
            .Build();
    }

    public Expr SelectScatterND(IR.Tensors.ScatterND scatterND, Call call, Expr output)
    {
        var input = (Expr)call[IR.Tensors.ScatterND.Input];
        var indices = (Expr)call[IR.Tensors.ScatterND.Indices];
        var updates = (Expr)call[IR.Tensors.ScatterND.Updates];
        if (!TryGetRankedTensorShapes("ScatterND affine selection", [input, indices, updates, output], out var shapes))
        {
            return call;
        }

        var inputShape = shapes[0];
        var indicesShape = shapes[1];
        var updatesShape = shapes[2];
        var outputShape = shapes[3];
        if (inputShape != outputShape)
        {
            throw new InvalidOperationException($"ScatterND affine selection expects output shape {inputShape}, got {outputShape}.");
        }

        var domainRank = outputShape.Rank;
        var domains = IR.F.Affine.Domains(domainRank);
        return IR.F.Affine.Grid()
            .Domain(domainRank, out var _)
            .Read(input, AffineMap.Identity(domainRank), out var inputTile)
            .Read(indices, CreateFullRegionMap(indicesShape, domains, "ScatterND indices affine selection"), out var indicesTile)
            .Read(updates, CreateFullRegionMap(updatesShape, domains, "ScatterND updates affine selection"), out var updatesTile)
            .Write(output, AffineMap.Identity(domainRank), out var outTile)
            .Body(TIR.F.NTT.ScatterND(inputTile, indicesTile, updatesTile, outTile))
            .Build();
    }

    private static AffineMap CreateGatherInputMap(Shape inputShape, Shape indexShape, int axis, AffineDomain[] domains)
    {
        var results = new AffineRange[inputShape.Rank];
        for (int i = 0; i < inputShape.Rank; i++)
        {
            if (i < axis)
            {
                results[i] = DomainRange(domains[i]);
            }
            else if (i == axis)
            {
                results[i] = FixedFullRange(inputShape[i], "Gather input axis affine selection");
            }
            else
            {
                results[i] = DomainRange(domains[i + indexShape.Rank - 1]);
            }
        }

        return new AffineMap(domains, default, results);
    }

    private static AffineMap CreateGatherIndexMap(Shape indexShape, int axis, AffineDomain[] domains)
    {
        var results = new AffineRange[indexShape.Rank];
        for (int i = 0; i < indexShape.Rank; i++)
        {
            results[i] = DomainRange(domains[axis + i]);
        }

        return new AffineMap(domains, default, results);
    }

    private static AffineMap CreateFullRegionMap(Shape shape, AffineDomain[] domains, string context)
    {
        var results = new AffineRange[shape.Rank];
        for (int i = 0; i < shape.Rank; i++)
        {
            results[i] = FixedFullRange(shape[i], context);
        }

        return new AffineMap(domains, default, results);
    }

    private static AffineMap CreateScalarPointerMap(int domainRank)
    {
        var domains = IR.F.Affine.Domains(domainRank);
        return new AffineMap(domains, default, Array.Empty<AffineRange>());
    }

    private static void ValidateSingleAddressRelation(AffineRelation relation, string context)
    {
        if (relation.Results.Length != 1)
        {
            throw new NotSupportedException($"{context} requires one address result, got {relation.Results.Length}.");
        }
    }

    private static void ValidateRank(Expr expr, int expectedRank, string context)
    {
        if (GetCheckedShape(expr) is not { IsUnranked: false } shape)
        {
            throw new NotSupportedException($"{context} requires a ranked tensor, got {expr.CheckedType}.");
        }

        if (shape.Rank != expectedRank)
        {
            throw new InvalidOperationException($"{context} rank {shape.Rank} does not match affine domain rank {expectedRank}.");
        }
    }

    private static Shape GetCheckedShape(Expr expr) => expr.CheckedType switch
    {
        TensorType tensorType => tensorType.Shape,
        DistributedType distributedType => distributedType.TensorType.Shape,
        _ => Shape.Scalar,
    };

    private static AffineRange DomainRange(AffineDomain domain) => new(domain.Offset, domain.Extent);

    private static AffineRange FixedFullRange(Dimension extent, string context)
    {
        if (!extent.IsFixed)
        {
            throw new NotSupportedException($"{context} requires fixed full-region extent, got {extent}.");
        }

        return new AffineRange(new AffineConstant(0), new AffineConstant(extent.FixedValue));
    }

    private static bool TryGetRankedTensorShapes(string context, Expr[] exprs, out Shape[] shapes)
    {
        shapes = new Shape[exprs.Length];
        for (int i = 0; i < exprs.Length; i++)
        {
            if (exprs[i].CheckedShape is not { IsUnranked: false } shape)
            {
                return false;
            }

            shapes[i] = shape;
        }

        return true;
    }
}
