// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Tiling;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

public sealed class DirectAffineTilingPass : FunctionPass
{
    private static readonly BufferStorage ThreadTileStorage = new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable);

    public DirectAffineTilingPass(string moduleKind)
    {
        ModuleKind = moduleKind;
    }

    public string ModuleKind { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function func)
        {
            return Task.FromResult(input);
        }

        if (input.ModuleKind != ModuleKind && !new DirectAffineTileAnalyzer().ContainsDirectAffine(func.Body.Body))
        {
            return Task.FromResult(input);
        }

        var analyzer = new DirectAffineTileAnalyzer();
        var decisions = analyzer.Analyze(func.Body.Body);
        if (DumpScope.Current.IsEnabled(DumpFlags.Tiling) || DumpScope.Current.IsEnabled(DumpFlags.PassIR))
        {
            DumpDecisions(func, decisions);
        }

        return Task.FromResult(input);
    }

    private static void DumpDecisions(Function func, IReadOnlyList<TileDecision> decisions)
    {
        using var writer = new StreamWriter(DumpScope.Current.OpenFile("direct-affine-decisions.md"), Encoding.UTF8);
        writer.WriteLine($"# Direct Affine Tile Decisions: {func.Name}");
        writer.WriteLine();
        if (decisions.Count == 0)
        {
            writer.WriteLine("No direct affine tiling-eligible expressions were found.");
            return;
        }

        foreach (var decision in decisions)
        {
            writer.WriteLine(decision.ToDumpString());
        }
    }

    private sealed class DirectAffineTileAnalyzer
    {
        private readonly HashSet<BaseExpr> _visited = new(ReferenceEqualityComparer.Instance);
        private readonly List<TileDecision> _decisions = new();

        public IReadOnlyList<TileDecision> Analyze(BaseExpr root)
        {
            Visit(root);
            return _decisions;
        }

        public bool ContainsDirectAffine(BaseExpr root)
        {
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            return ContainsDirectAffine(root, visited);
        }

        private static DistributionLayout? GetDistributionLayout(IRType type, IRArray<SBP> ndsbp, Placement placement)
        {
            if (type is DistributedType distributedType)
            {
                LayoutVerifier.Verify(distributedType.DistributionLayout, distributedType.StorageLayout);
                return distributedType.DistributionLayout;
            }

            if (placement.Rank == 0)
            {
                return null;
            }

            var tensorType = GetTensorType(type, "distributed direct affine tile");
            var distributed = new DistributedType(tensorType, ndsbp, placement);
            LayoutVerifier.Verify(distributed.DistributionLayout, distributed.StorageLayout);
            return distributed.DistributionLayout;
        }

        private static TensorType GetTensorType(IRType type, string opKind) => type switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => throw new NotSupportedException($"{opKind} tiling requires tensor output, got {type}."),
        };

        private static void ValidateOneDimensionalDirectAffine(AffineRelation relation, Shape shape, RankedShape symbols, string opKind)
        {
            ValidateOneDimensionalShape(shape, opKind);
            if (relation.Domains.Length != shape.Rank)
            {
                throw new NotSupportedException($"{opKind} tiling expects relation domain rank {shape.Rank}, got {relation.Domains.Length} for shape {shape}.");
            }

            if (relation.Results.Length != 1)
            {
                throw new NotSupportedException($"{opKind} tiling expects exactly one address result, got {relation.Results.Length} for relation {relation}.");
            }

            if (symbols.Rank != relation.Symbols.Length)
            {
                throw new NotSupportedException($"{opKind} tiling symbol payload rank {symbols.Rank} does not match relation symbol count {relation.Symbols.Length}.");
            }
        }

        private static void ValidateOneDimensionalShape(Shape shape, string opKind)
        {
            if (shape.IsUnranked)
            {
                throw new NotSupportedException($"{opKind} tiling requires ranked shape.");
            }

            if (shape.Rank != 1)
            {
                throw new NotSupportedException($"{opKind} tiling currently supports rank-1 direct affine tiles only, got shape {shape}.");
            }

            if (!shape[0].IsFixed)
            {
                throw new NotSupportedException($"{opKind} tiling requires a fixed tile extent for capacity accounting, got {shape}.");
            }
        }

        private static long GetFixedByteSize(TensorType tensorType, string opKind)
        {
            ValidateOneDimensionalShape(tensorType.Shape, opKind);
            checked
            {
                return tensorType.Shape[0].FixedValue * tensorType.DType.SizeInBytes;
            }
        }

        private void Visit(BaseExpr expr)
        {
            if (!_visited.Add(expr))
            {
                return;
            }

            foreach (var operand in expr.Operands)
            {
                Visit(operand);
            }

            if (expr is not Call call)
            {
                return;
            }

            switch (call.Target)
            {
                case Gather gather:
                    AttachGatherDecision(call, gather);
                    break;
                case Scatter scatter:
                    AttachScatterDecision(call, scatter);
                    break;
                case IR.Math.Binary:
                    AttachElementwiseDecisionIfTilingInput(call, "Binary");
                    break;
                case IR.Math.Unary:
                    AttachElementwiseDecisionIfTilingInput(call, "Unary");
                    break;
                case IR.Tensors.Cast:
                    AttachElementwiseDecisionIfTilingInput(call, "Cast");
                    break;
                case IR.Tensors.Where:
                    AttachElementwiseDecisionIfTilingInput(call, "Where");
                    break;
            }
        }

        private void AttachGatherDecision(Call call, Gather gather)
        {
            var tensorType = GetTensorType(call.CheckedType, "Affine.Gather");
            ValidateOneDimensionalDirectAffine(gather.Relation, tensorType.Shape, gather.Symbols, "Affine.Gather");
            var distributionLayout = GetDistributionLayout(call.CheckedType, gather.NdSBP, gather.Placement);
            AttachDecision(call, "Affine.Gather", tensorType, distributionLayout, ThreadTileStorage, "direct affine gather tile");
        }

        private void AttachScatterDecision(Call call, Scatter scatter)
        {
            var source = (Expr)call[Scatter.Source];
            var tensorType = GetTensorType(source.CheckedType, "Affine.Scatter source");
            ValidateOneDimensionalDirectAffine(scatter.Relation, tensorType.Shape, scatter.Symbols, "Affine.Scatter");
            var distributionLayout = GetDistributionLayout(source.CheckedType, new IRArray<SBP>(), new Placement([], string.Empty));
            AttachDecision(call, "Affine.Scatter", tensorType, distributionLayout, ThreadTileStorage, "direct affine scatter tile terminator");
        }

        private void AttachElementwiseDecisionIfTilingInput(Call call, string opKind)
        {
            if (!call.Arguments.ToArray().OfType<Expr>().Any(arg => TileDecisionMetadata.TryGet(arg, out _)))
            {
                return;
            }

            var tensorType = GetTensorType(call.CheckedType, opKind);
            ValidateOneDimensionalShape(tensorType.Shape, opKind);
            var distributionLayout = GetDistributionLayout(call.CheckedType, new IRArray<SBP>(), new Placement([], string.Empty));
            AttachDecision(call, opKind, tensorType, distributionLayout, ThreadTileStorage, "producer consumes direct affine tile");
        }

        private void AttachDecision(Call call, string opKind, TensorType tensorType, DistributionLayout? distributionLayout, BufferStorage storage, string reason)
        {
            var byteSize = GetFixedByteSize(tensorType, opKind);
            var storageLayout = StorageLayout.Identity(distributionLayout?.LocalShape ?? tensorType.Shape);
            if (distributionLayout is not null)
            {
                LayoutVerifier.Verify(distributionLayout, storageLayout);
            }

            var decision = new TileDecision(
                $"tile_{_decisions.Count}",
                opKind,
                tensorType.Shape,
                distributionLayout,
                storageLayout,
                storage,
                new TileLifetime(_decisions.Count, _decisions.Count),
                byteSize,
                new TileCapacity(byteSize, null, "direct-affine-conservative"),
                storage.Scope is BufferScope.BlockLocal,
                reason);
            TileDecisionMetadata.Set(call, decision);
            _decisions.Add(decision);
        }

        private bool ContainsDirectAffine(BaseExpr expr, ISet<BaseExpr> visited)
        {
            if (!visited.Add(expr))
            {
                return false;
            }

            if (expr is Call { Target: Gather or Scatter })
            {
                return true;
            }

            foreach (var operand in expr.Operands)
            {
                if (ContainsDirectAffine(operand, visited))
                {
                    return true;
                }
            }

            return false;
        }
    }
}
