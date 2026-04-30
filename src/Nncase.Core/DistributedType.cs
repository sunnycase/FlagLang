// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.Utilities;

namespace Nncase.IR;

public enum HierarchyKind : byte
{
    Parallel = 0,
    SMT = 1,
}

/// <summary>
/// Per-thread element ownership order inside a Triton blocked tile.
/// </summary>
public enum TritonThreadElementOrder : byte
{
    Contiguous,
    Strided,
}

[JsonConverter(typeof(SBPConverter))]
public abstract record SBP
{
    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPPartial P(ReduceOp op = ReduceOp.Sum) => new SBPPartial(op);

    public static SBPSplit S(IRArray<int> axes) => new SBPSplit(axes);

    public static SBPSplit S(params int[] axes) => new SBPSplit(axes);
}

public sealed record SBPSplit(IRArray<int> Axes) : SBP
{
    public override string ToString() => $"S({string.Join(",", Axes)})";
}

public sealed record SBPPartial(ReduceOp Op) : SBP
{
    public override string ToString() => $"P({Op})";
}

public sealed record SBPBroadCast : SBP
{
    public static readonly SBPBroadCast Instance = new SBPBroadCast();

    public override string ToString() => "B";
}

public class SBPConverter : JsonConverter<SBP>
{
    public override SBP Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        string? typeDiscriminator = null;
        SBPSplit? sbpSplit = null;
        SBPPartial? sbpPartial = null;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string? propertyName = reader.GetString();
                reader.Read(); // Move to property value

                switch (propertyName)
                {
                    case "$type":
                        typeDiscriminator = reader.GetString();
                        break;
                    case "Axes":
                        int[] axes = JsonSerializer.Deserialize<int[]>(ref reader, options)!;
                        var irAxes = new IRArray<int>(axes);
                        if (typeDiscriminator == "S")
                        {
                            sbpSplit = new SBPSplit(irAxes);
                        }
                        else
                        {
                            throw new InvalidDataException("Axes must be used in SBP split");
                        }

                        break;
                    case "Op":
                        ReduceOp partialOp = JsonSerializer.Deserialize<ReduceOp>(ref reader, options);
                        sbpPartial = new SBPPartial(partialOp);
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }
        }

        switch (typeDiscriminator)
        {
            case "B":
                return SBP.B;
            case "P":
                return sbpPartial!;
            case "S":
                return sbpSplit!;
            default:
                throw new JsonException($"Unknown '$type' discriminator: {typeDiscriminator}");
        }
    }

    public override void Write(Utf8JsonWriter writer, SBP value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        if (value is SBPBroadCast)
        {
            writer.WriteString("$type", "B");
        }
        else if (value is SBPPartial partialValue)
        {
            writer.WriteString("$type", "P");
            writer.WriteString("Op", partialValue.Op.ToString());
        }
        else if (value is SBPSplit splitValue)
        {
            writer.WriteString("$type", "S");
            writer.WritePropertyName("Axes");
            JsonSerializer.Serialize(writer, splitValue.Axes.ToArray(), options);
        }
        else
        {
            throw new JsonException($"Unknown SBP type: {value.GetType()}");
        }

        writer.WriteEndObject();
    }
}

// public sealed record Placement(Placement.DeviceKind Kind, IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind)
public sealed record Placement(IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind = HierarchyKind.Parallel)
{
    // public enum DeviceKind : uint
    // {
    //     CPU = 0,
    // }
    public int Rank => Hierarchy.Count;

    public bool HasWarp => Name.Contains('w', StringComparison.Ordinal);

    public override string ToString() => $"[{string.Join(',', Hierarchy.Zip(Name).Select(t => t.Second.ToString() + ':' + t.First.ToString()))}]";
}

public abstract record IndexExpr
{
    public static IndexExpr Var(string name) => new IndexVar(name);

    public static IndexExpr Const(long value) => new IndexConst(value);

    public static IndexExpr Add(params IndexExpr[] terms) => new IndexAdd(terms);

    public static IndexExpr Mul(params IndexExpr[] factors) => new IndexMul(factors);

    public static IndexExpr FloorDiv(IndexExpr value, IndexExpr divisor) => new IndexFloorDiv(value, divisor);

    public static IndexExpr Mod(IndexExpr value, IndexExpr divisor) => new IndexMod(value, divisor);
}

public sealed record IndexVar(string Name) : IndexExpr
{
    public override string ToString() => Name;
}

public sealed record IndexConst(long Value) : IndexExpr
{
    public override string ToString() => Value.ToString(CultureInfo.InvariantCulture);
}

public sealed record IndexAny : IndexExpr
{
    public static readonly IndexAny Instance = new();

    public override string ToString() => "any";
}

public sealed record IndexNamedPrimitive(string Name, IRArray<IndexExpr> Arguments) : IndexExpr
{
    public override string ToString() => $"{Name}({string.Join(",", Arguments)})";
}

public sealed record IndexAdd(IRArray<IndexExpr> Terms) : IndexExpr
{
    public override string ToString() => string.Join("+", Terms.Select(FormatTerm));

    private static string FormatTerm(IndexExpr expr) => expr is IndexAdd ? $"({expr})" : expr.ToString();
}

public sealed record IndexMul(IRArray<IndexExpr> Factors) : IndexExpr
{
    public override string ToString() => string.Join("*", Factors.Select(FormatFactor));

    private static string FormatFactor(IndexExpr expr) => expr is IndexAdd ? $"({expr})" : expr.ToString();
}

public sealed record IndexFloorDiv(IndexExpr Value, IndexExpr Divisor) : IndexExpr
{
    public override string ToString() => $"floor({Value}/{Divisor})";
}

public sealed record IndexMod(IndexExpr Value, IndexExpr Divisor) : IndexExpr
{
    public override string ToString() => $"{Format(Value)}%{Divisor}";

    private static string Format(IndexExpr expr) => expr is IndexVar or IndexConst or IndexFloorDiv ? expr.ToString() : $"({expr})";
}

public sealed record IndexMapBinding(string Name, IndexExpr Expr)
{
    public static IndexMapBinding Any(string name) => new(name, IndexAny.Instance);

    public override string ToString() => $"{Name}={Expr}";
}

public sealed record IndexMapDescriptor(
    string Name,
    IRArray<string> Inputs,
    IRArray<IndexMapBinding> Outputs,
    IRArray<string> InputDomain,
    IRArray<string> OutputDomain,
    string Predicate = "true",
    string? Inverse = null)
{
    public override string ToString()
    {
        var input = string.Join(", ", Inputs);
        var output = string.Join(", ", Outputs);
        return $"{Name}: ({input}) -> ({output}) where {Predicate}";
    }
}

public static class LayoutVerifier
{
    public static void Verify(DistributionLayout layout)
    {
        VerifyMap(layout.GlobalToOwnerLocal, layout.OwnerLocalToGlobal.Name, layout.Kind);
        VerifyMap(layout.OwnerLocalToGlobal, layout.GlobalToOwnerLocal.Name, layout.Kind);
        if (layout.LocalShape.IsUnranked)
        {
            throw new InvalidOperationException($"DistributionLayout {layout.Kind} has unranked local shape.");
        }
    }

    public static void Verify(DistributionLayout distributionLayout, StorageLayout storageLayout)
    {
        Verify(distributionLayout);
        VerifyMap(storageLayout.LogicalToPhysical, storageLayout.LogicalToPhysical.Name, storageLayout.Kind);
        if (storageLayout.LogicalShape != distributionLayout.LocalShape && storageLayout.ViewMap is null)
        {
            throw new InvalidOperationException($"StorageLayout {storageLayout.Kind} logical shape {storageLayout.LogicalShape} does not match DistributionLayout {distributionLayout.Kind} local shape {distributionLayout.LocalShape} and has no view map.");
        }

        if (storageLayout.ViewMap is { } viewMap)
        {
            VerifyMap(viewMap, distributionLayout.GlobalToOwnerLocal.Name, storageLayout.Kind);
        }
    }

    private static void VerifyMap(IndexMapDescriptor map, string expectedInverse, string layoutKind)
    {
        if (map.Outputs.Count == 0)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} has no outputs.");
        }

        if (map.Inverse != expectedInverse)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} must declare inverse {expectedInverse}, but got {map.Inverse ?? "<none>"}.");
        }

        if (map.InputDomain.Count == 0 || map.OutputDomain.Count == 0)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} must declare input and output domains.");
        }

        foreach (var output in map.Outputs)
        {
            VerifyExpr(output.Expr, layoutKind, map.Name);
        }
    }

    private static void VerifyExpr(IndexExpr expr, string layoutKind, string mapName)
    {
        switch (expr)
        {
            case IndexVar:
            case IndexConst:
            case IndexAny:
                return;
            case IndexAdd add:
                foreach (var term in add.Terms)
                {
                    VerifyExpr(term, layoutKind, mapName);
                }

                return;
            case IndexMul mul:
                foreach (var factor in mul.Factors)
                {
                    VerifyExpr(factor, layoutKind, mapName);
                }

                return;
            case IndexFloorDiv floorDiv:
                VerifyExpr(floorDiv.Value, layoutKind, mapName);
                VerifyExpr(floorDiv.Divisor, layoutKind, mapName);
                return;
            case IndexMod mod:
                VerifyExpr(mod.Value, layoutKind, mapName);
                VerifyExpr(mod.Divisor, layoutKind, mapName);
                return;
            case IndexNamedPrimitive named:
                throw new NotSupportedException($"Layout {layoutKind} map {mapName} uses unsupported named primitive {named.Name}; add forward, inverse, domain, cost, and codegen lowering before using it.");
            default:
                throw new NotSupportedException($"Layout {layoutKind} map {mapName} uses unsupported index expression {expr.GetType().Name}.");
        }
    }
}

public sealed record DistributionLayout(
    string Kind,
    IndexMapDescriptor GlobalToOwnerLocal,
    IndexMapDescriptor OwnerLocalToGlobal,
    Shape LocalShape,
    string ValidPredicate = "true",
    IRArray<string>? Attributes = null)
{
    public static DistributionLayout FromAxisPolicies(TensorType tensorType, IRArray<SBP> axisPolicies, Placement placement)
    {
        if (axisPolicies.Count != tensorType.Shape.Rank)
        {
            throw new InvalidOperationException($"Axis policy rank {axisPolicies.Count} does not match tensor rank {tensorType.Shape.Rank} for shape {tensorType.Shape}.");
        }

        if (!DistributedUtility.IsDistributable(tensorType, axisPolicies.ToArray(), placement))
        {
            throw new InvalidOperationException($"Axis policies ({string.Join(',', axisPolicies)}) are not distributable for shape {tensorType.Shape} and placement {placement}.");
        }

        if (!DistributedUtility.TryGetDividedTensorType(new DistributedType(tensorType, axisPolicies, placement), out var localTensorType))
        {
            throw new InvalidOperationException($"Failed to derive local tensor type for shape {tensorType.Shape}, policies ({string.Join(',', axisPolicies)}), placement {placement}.");
        }

        var rank = tensorType.Shape.Rank;
        var globalInputs = Enumerable.Range(0, rank).Select(i => $"g{i}").ToArray();
        var ownerLocalOutputs = BuildSbpOwnerLocalOutputs(axisPolicies, placement, localTensorType.Shape, rank);
        var ownerInputs = Enumerable.Range(0, placement.Rank).Select(i => $"owner{i}").ToArray();
        var localInputs = Enumerable.Range(0, rank).Select(i => $"l{i}").ToArray();
        var globalOutputs = BuildSbpGlobalOutputs(axisPolicies, placement, localTensorType.Shape, rank);

        return new DistributionLayout(
            "SBP",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                globalInputs,
                ownerLocalOutputs,
                BuildShapeDomain("g", tensorType.Shape, rank),
                BuildOwnerLocalDomain(placement, localTensorType.Shape, rank),
                BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
                "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ownerInputs.Concat(localInputs).ToArray(),
                globalOutputs,
                BuildOwnerLocalDomain(placement, localTensorType.Shape, rank),
                BuildShapeDomain("g", tensorType.Shape, rank),
                BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
                "GlobalToOwnerLocal"),
            localTensorType.Shape,
            BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
            axisPolicies.Select((sbp, i) => $"axis{i}:{sbp}").ToArray());
    }

    public static DistributionLayout TritonBlocked(Shape tensorShape, TritonBlockedLayout layout)
    {
        if (tensorShape.Rank != 1)
        {
            throw new NotSupportedException($"TritonBlocked DistributionLayout currently supports rank-1 tensors only, but got shape {tensorShape}.");
        }

        layout.Validate();
        var threadsPerCTA = layout.ThreadsPerWarp * layout.WarpsPerCTA;
        var elementsPerCTA = layout.SizePerThread * threadsPerCTA;
        var localShape = new RankedShape(layout.SizePerThread);
        return new DistributionLayout(
            "TritonBlocked",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"g{i}").ToArray(),
                BuildTritonBlockedOwnerLocalOutputs(layout, threadsPerCTA, elementsPerCTA),
                BuildShapeDomain("g", tensorShape, tensorShape.Rank),
                [$"0<=cta && 0<=warp<{layout.WarpsPerCTA} && 0<=lane<{layout.ThreadsPerWarp} && 0<=elem<{layout.SizePerThread}"],
                $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
                "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["cta", "warp", "lane", "elem"],
                BuildTritonBlockedGlobalOutputs(layout, threadsPerCTA, elementsPerCTA),
                [$"0<=cta && 0<=warp<{layout.WarpsPerCTA} && 0<=lane<{layout.ThreadsPerWarp} && 0<=elem<{layout.SizePerThread}"],
                BuildShapeDomain("g", tensorShape, tensorShape.Rank),
                $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
                "GlobalToOwnerLocal"),
            localShape,
            $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
            layout.ToAttributes());
    }

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LocalShape={LocalShape}, Valid={ValidPredicate}{attrs}";
    }

    private static IndexMapBinding[] BuildSbpOwnerLocalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var ownerOutputs = new IndexMapBinding[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            ownerOutputs[i] = IndexMapBinding.Any($"owner{i}");
        }

        var localOutputs = new List<IndexMapBinding>();
        for (int dim = 0; dim < rank; dim++)
        {
            var globalIndex = IndexExpr.Var($"g{dim}");
            if (axisPolicies[dim] is not SBPSplit split)
            {
                localOutputs.Add(new IndexMapBinding($"l{dim}", globalIndex));
            }
            else
            {
                var localExtent = DimensionExpr(localShape[dim]);
                localOutputs.Add(new IndexMapBinding($"l{dim}", IndexExpr.Mod(globalIndex, localExtent)));

                var stride = 1;
                foreach (var axis in split.Axes)
                {
                    ValidatePlacementAxis(axis, placement);
                    ownerOutputs[axis] = new IndexMapBinding(
                        $"owner{axis}",
                        IndexExpr.Mod(
                            IndexExpr.FloorDiv(globalIndex, Product(localExtent, stride)),
                            IndexExpr.Const(placement.Hierarchy[axis])));
                    stride *= placement.Hierarchy[axis];
                }
            }
        }

        return ownerOutputs.Concat(localOutputs).ToArray();
    }

    private static IndexMapBinding[] BuildSbpGlobalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var outputs = new List<IndexMapBinding>();
        for (int dim = 0; dim < rank; dim++)
        {
            var localIndex = IndexExpr.Var($"l{dim}");
            if (axisPolicies[dim] is not SBPSplit split)
            {
                outputs.Add(new IndexMapBinding($"g{dim}", localIndex));
            }
            else
            {
                var localExtent = DimensionExpr(localShape[dim]);
                var ownerLinear = BuildOwnerLinearExpression(split, placement);
                outputs.Add(new IndexMapBinding($"g{dim}", IndexExpr.Add(IndexExpr.Mul(ownerLinear, localExtent), localIndex)));
            }
        }

        return outputs.ToArray();
    }

    private static string BuildSbpValidPredicate(Placement placement, Shape localShape, int rank)
    {
        var predicates = new List<string>();
        for (int axis = 0; axis < placement.Rank; axis++)
        {
            predicates.Add($"0<=owner{axis}<{placement.Hierarchy[axis]}");
        }

        for (int dim = 0; dim < rank; dim++)
        {
            predicates.Add($"0<=l{dim}<{FormatDimension(localShape[dim])}");
        }

        return string.Join(" && ", predicates);
    }

    private static IndexExpr BuildOwnerLinearExpression(SBPSplit split, Placement placement)
    {
        var stride = 1;
        var terms = new List<IndexExpr>();
        foreach (var axis in split.Axes)
        {
            ValidatePlacementAxis(axis, placement);
            var owner = IndexExpr.Var($"owner{axis}");
            terms.Add(stride == 1 ? owner : IndexExpr.Mul(owner, IndexExpr.Const(stride)));
            stride *= placement.Hierarchy[axis];
        }

        return terms.Count == 1 ? terms[0] : IndexExpr.Add(terms.ToArray());
    }

    private static IndexExpr Product(IndexExpr extent, int multiplier) => multiplier == 1 ? extent : IndexExpr.Mul(extent, IndexExpr.Const(multiplier));

    private static IndexExpr DimensionExpr(Dimension dimension) => dimension.IsFixed ? IndexExpr.Const(dimension.FixedValue) : IndexExpr.Var(FormatDimension(dimension));

    private static string[] BuildShapeDomain(string prefix, Shape shape, int rank) =>
        Enumerable.Range(0, rank).Select(i => $"0<={prefix}{i}<{FormatDimension(shape[i])}").ToArray();

    private static string[] BuildOwnerLocalDomain(Placement placement, Shape localShape, int rank) =>
        Enumerable.Range(0, placement.Rank).Select(axis => $"0<=owner{axis}<{placement.Hierarchy[axis]}")
            .Concat(Enumerable.Range(0, rank).Select(dim => $"0<=l{dim}<{FormatDimension(localShape[dim])}"))
            .ToArray();

    private static string FormatDimension(Dimension dimension) => dimension.IsFixed ? dimension.FixedValue.ToString(CultureInfo.InvariantCulture) : dimension.ToString();

    private static void ValidatePlacementAxis(int axis, Placement placement)
    {
        if (axis < 0 || axis >= placement.Rank)
        {
            throw new InvalidOperationException($"Split axis {axis} is outside placement rank {placement.Rank} for placement {placement}.");
        }

        if (placement.Hierarchy[axis] <= 0)
        {
            throw new InvalidOperationException($"Placement axis {axis} has invalid hierarchy {placement.Hierarchy[axis]} for placement {placement}.");
        }
    }

    private static IndexMapBinding[] BuildTritonBlockedOwnerLocalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA)
    {
        var g0 = IndexExpr.Var("g0");
        var elementInCta = IndexExpr.Mod(g0, IndexExpr.Const(elementsPerCTA));
        return layout.ThreadElementOrder switch
        {
            TritonThreadElementOrder.Contiguous =>
            [
                new("cta", IndexExpr.FloorDiv(g0, IndexExpr.Const(elementsPerCTA))),
                new("warp", IndexExpr.FloorDiv(elementInCta, IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp))),
                new("lane", IndexExpr.FloorDiv(IndexExpr.Mod(g0, IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp)), IndexExpr.Const(layout.SizePerThread))),
                new("elem", IndexExpr.Mod(g0, IndexExpr.Const(layout.SizePerThread))),
            ],
            TritonThreadElementOrder.Strided =>
            [
                new("cta", IndexExpr.FloorDiv(g0, IndexExpr.Const(elementsPerCTA))),
                new("warp", IndexExpr.FloorDiv(IndexExpr.Mod(g0, IndexExpr.Const(threadsPerCTA)), IndexExpr.Const(layout.ThreadsPerWarp))),
                new("lane", IndexExpr.Mod(IndexExpr.Mod(g0, IndexExpr.Const(threadsPerCTA)), IndexExpr.Const(layout.ThreadsPerWarp))),
                new("elem", IndexExpr.FloorDiv(elementInCta, IndexExpr.Const(threadsPerCTA))),
            ],
            _ => throw new NotSupportedException($"Unsupported Triton thread element order {layout.ThreadElementOrder}."),
        };
    }

    private static IndexMapBinding[] BuildTritonBlockedGlobalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA) => layout.ThreadElementOrder switch
    {
        TritonThreadElementOrder.Contiguous =>
        [
            new(
                "g0",
                IndexExpr.Add(
                    IndexExpr.Mul(IndexExpr.Var("cta"), IndexExpr.Const(elementsPerCTA)),
                    IndexExpr.Mul(IndexExpr.Var("warp"), IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp)),
                    IndexExpr.Mul(IndexExpr.Var("lane"), IndexExpr.Const(layout.SizePerThread)),
                    IndexExpr.Var("elem"))),
        ],
        TritonThreadElementOrder.Strided =>
        [
            new(
                "g0",
                IndexExpr.Add(
                    IndexExpr.Mul(IndexExpr.Var("cta"), IndexExpr.Const(elementsPerCTA)),
                    IndexExpr.Mul(IndexExpr.Var("warp"), IndexExpr.Const(layout.ThreadsPerWarp)),
                    IndexExpr.Var("lane"),
                    IndexExpr.Mul(IndexExpr.Var("elem"), IndexExpr.Const(threadsPerCTA)))),
        ],
        _ => throw new NotSupportedException($"Unsupported Triton thread element order {layout.ThreadElementOrder}."),
    };
}

public sealed record TritonBlockedLayout(
    int SizePerThread,
    int ThreadsPerWarp,
    int WarpsPerCTA,
    IRArray<int> Order,
    IRArray<int> CTAsPerCGA,
    IRArray<int> CTASplitNum,
    IRArray<int> CTAOrder,
    TritonThreadElementOrder ThreadElementOrder = TritonThreadElementOrder.Contiguous)
{
    public IRArray<string> ToAttributes() =>
    [
        $"sizePerThread={SizePerThread}",
        $"threadsPerWarp={ThreadsPerWarp}",
        $"warpsPerCTA={WarpsPerCTA}",
        $"threadElementOrder={ThreadElementOrder}",
        $"order=[{string.Join(",", Order)}]",
        $"ctasPerCGA=[{string.Join(",", CTAsPerCGA)}]",
        $"ctaSplitNum=[{string.Join(",", CTASplitNum)}]",
        $"ctaOrder=[{string.Join(",", CTAOrder)}]",
    ];

    public void Validate()
    {
        if (SizePerThread <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout SizePerThread must be positive, got {SizePerThread}.");
        }

        if (ThreadsPerWarp <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout ThreadsPerWarp must be positive, got {ThreadsPerWarp}.");
        }

        if (WarpsPerCTA <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout WarpsPerCTA must be positive, got {WarpsPerCTA}.");
        }
    }
}

public sealed record StorageLayout(
    string Kind,
    Shape LogicalShape,
    IndexMapDescriptor LogicalToPhysical,
    string ValidPredicate = "true",
    IRArray<string>? Attributes = null,
    IndexMapDescriptor? ViewMap = null)
{
    public static StorageLayout Identity(Shape localShape) =>
        new(
            "Identity",
            localShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                Enumerable.Range(0, localShape.Rank).Select(i => $"l{i}").ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => new IndexMapBinding($"p{i}", IndexExpr.Var($"l{i}"))).ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => $"0<=l{i}<{FormatDimension(localShape[i])}").ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => $"0<=p{i}<{FormatDimension(localShape[i])}").ToArray(),
                "true",
                "LogicalToPhysical"));

    public static StorageLayout SharedBlock(Shape tensorShape, DistributionLayout distributionLayout)
    {
        if (tensorShape.IsUnranked)
        {
            throw new InvalidOperationException("SharedBlock storage layout requires a ranked tensor shape.");
        }

        return new(
            "SharedBlock",
            tensorShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"g{i}").ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => new IndexMapBinding($"p{i}", IndexExpr.Var($"g{i}"))).ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"0<=g{i}<{FormatDimension(tensorShape[i])}").ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"0<=p{i}<{FormatDimension(tensorShape[i])}").ToArray(),
                "true",
                "LogicalToPhysical"),
            distributionLayout.ValidPredicate,
            ["scope:block_local"],
            distributionLayout.OwnerLocalToGlobal);
    }

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LogicalShape={LogicalShape}, Valid={ValidPredicate}{attrs}";
    }

    private static string FormatDimension(Dimension dimension) => dimension.IsFixed ? dimension.FixedValue.ToString(CultureInfo.InvariantCulture) : dimension.ToString();
}

public sealed record DistributedType(
    TensorType TensorType,
    IRArray<SBP> AxisPolicies,
    Placement Placement,
    bool Partial = false,
    DistributionLayout? ExplicitDistributionLayout = null,
    StorageLayout? ExplicitStorageLayout = null) : IRType
{
    public DistributionLayout DistributionLayout => ExplicitDistributionLayout ?? DistributionLayout.FromAxisPolicies(TensorType, AxisPolicies, Placement);

    public StorageLayout StorageLayout => ExplicitStorageLayout ?? StorageLayout.Identity(DistributionLayout.LocalShape);

    public static DistributedType FromLayouts(
        TensorType tensorType,
        Placement placement,
        DistributionLayout distributionLayout,
        StorageLayout? storageLayout = null,
        IRArray<SBP>? axisPolicies = null,
        bool partial = false)
    {
        storageLayout ??= StorageLayout.Identity(distributionLayout.LocalShape);
        LayoutVerifier.Verify(distributionLayout, storageLayout);
        return new DistributedType(
            tensorType,
            axisPolicies ?? Enumerable.Range(0, tensorType.Shape.Rank).Select(_ => SBP.B).ToArray(),
            placement,
            partial,
            distributionLayout,
            storageLayout);
    }

    public override string ToString() => $"{TensorType}, ({string.Join(',', AxisPolicies)}), {Placement}, Layout: {DistributionLayout.Kind}, Storage: {StorageLayout.Kind}, Partial: {Partial}";
}
