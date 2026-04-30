// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.IR.Logics;
using Nncase.IR.Shapes;
using Nncase.Tiling;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes
{
    /// <summary>
    /// Lowers <see cref="TIR.NTT.AffineGather"/> and <see cref="TIR.NTT.AffineScatter"/> into canonical buffer loops
    /// so downstream codegen paths can be reused.
    /// </summary>
    public sealed class NTTAffineIOLoweringPass : FunctionPass
    {
        private readonly AffineIOFusionRewriter _fusionRewriter = new();
        private readonly AffineIOLoweringRewriter _rewriter = new();

        /// <inheritdoc/>
        protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context) => Task.FromResult(input switch
        {
            PrimFunction primFunc => LowerPrimFunction(primFunc),
            PrimFunctionWrapper { Target: PrimFunction target } wrapper => LowerWrapper(wrapper, target),
            _ => input,
        });

        private PrimFunction LowerPrimFunction(PrimFunction func)
        {
            var fusedBody = (Sequential)_fusionRewriter.Visit(func.Body, default);
            var newBody = (Sequential)_rewriter.Visit(fusedBody, default);
            return ReferenceEquals(newBody, func.Body) ? func : func.With(body: newBody);
        }

        private BaseFunction LowerWrapper(PrimFunctionWrapper wrapper, PrimFunction target)
        {
            var loweredTarget = LowerPrimFunction(target);
            return ReferenceEquals(loweredTarget, target) ? wrapper : wrapper.With(target: loweredTarget);
        }

        private sealed class AffineIOFusionRewriter : ExprRewriter<Unit>
        {
            protected override BaseExpr RewriteLeafSequential(Sequential expr, Unit context)
            {
                var fields = expr.Fields.ToArray();
                var rewrittenFields = new List<Expr>(fields.Length);
                var mutated = false;
                for (int i = 0; i < fields.Length; i++)
                {
                    if (i + 1 < fields.Length && TryFuseAffineGatherTensorLoad(fields[i], fields[i + 1], out var fused))
                    {
                        rewrittenFields.Add(fused);
                        i++;
                        mutated = true;
                    }
                    else
                    {
                        rewrittenFields.Add(fields[i]);
                    }
                }

                return mutated ? expr.With(fields: rewrittenFields.ToArray()) : expr;
            }

            private static bool TryFuseAffineGatherTensorLoad(Expr first, Expr second, [MaybeNullWhen(false)] out Expr fused)
            {
                fused = null;
                if (first is not Call { Target: TIR.NTT.AffineGather gather } gatherCall ||
                    second is not Call { Target: TIR.NTT.TensorLoad tensorLoad } tensorLoadCall ||
                    tensorLoadCall[TIR.NTT.TensorLoad.Dest] is not TIR.Buffer { DistributedType: not null } distributedOutput ||
                    !Equals(gatherCall[TIR.NTT.AffineGather.Output], tensorLoadCall[TIR.NTT.TensorLoad.Src]))
                {
                    return false;
                }

                fused = TIR.F.NTT.AffineGather(
                    (Expr)gatherCall[TIR.NTT.AffineGather.Source],
                    (Expr)gatherCall[TIR.NTT.AffineGather.DefaultValue],
                    distributedOutput,
                    gather.Relation,
                    gather.Symbols,
                    gather.Shape).InheritMetaData(gatherCall);
                return true;
            }
        }

        private sealed class AffineIOLoweringRewriter : ExprRewriter<Unit>
        {
            private static readonly BufferStorage DefaultTileSourceStorage = new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable);

            private int _bufferIndex;

            protected override BaseExpr RewriteLeafCall(Call expr, Unit context)
            {
                return expr.Target switch
                {
                    TIR.NTT.AffineGather gather => LowerGather(expr, gather, context),
                    TIR.NTT.AffineScatter scatter => LowerScatter(expr, scatter, context),
                    _ => base.RewriteLeafCall(expr, context),
                };
            }

            private Expr LowerGather(Call call, TIR.NTT.AffineGather gather, Unit context)
            {
                var source = (Expr)Visit(call[TIR.NTT.AffineGather.Source], context);
                var defaultValue = (Expr)Visit(call[TIR.NTT.AffineGather.DefaultValue], context);
                var output = RequireBuffer(Visit(call[TIR.NTT.AffineGather.Output], context));

                ValidateRelation(gather.Relation, output.Dimensions.Length);

                var globalExtents = output.Dimensions.ToArray();
                var iterationExtents = AffineIOLayoutEvaluator.GetIterationExtents(output);
                var symbolMap = BuildSymbolMap(gather.Relation, gather.Symbols);
                Expr? defaultSetup = null;
                if (gather.Relation.Constraint != LogicalExpr.True)
                {
                    (defaultValue, defaultSetup) = PrepareGatherDefault(defaultValue);
                }

                var loopNest = BuildLoopNest(iterationExtents, loopVars =>
                {
                    var domainValues = AffineIOLayoutEvaluator.GetDomainValues(output, loopVars, globalExtents);
                    var address = EvaluateAddress(gather.Relation, domainValues, globalExtents, symbolMap);
                    var loaded = T.Load(source, address);
                    var loopIndices = loopVars.AsExprs();
                    var storageIndices = AffineIOLayoutEvaluator.GetStorageIndices(output, loopVars, domainValues);
                    var storeLoaded = T.BufferStore(output, storageIndices, loaded);
                    if (gather.Relation.Constraint == LogicalExpr.True)
                    {
                        return storeLoaded;
                    }

                    var fallback = ReadDefaultValue(defaultValue, loopIndices, output.ElemType);
                    var storeFallback = T.BufferStore(output, storageIndices, fallback);
                    return T.If(EvaluateConstraint(gather.Relation.Constraint, domainValues)).Then(storeLoaded).Else(storeFallback).Build();
                });
                return defaultSetup is null ? loopNest : T.Sequential(defaultSetup, loopNest);
            }

            private Expr LowerScatter(Call call, TIR.NTT.AffineScatter scatter, Unit context)
            {
                var sourceExpr = Visit(call[TIR.NTT.AffineScatter.Source], context);
                var (source, sourceSetup) = RequireReadableBuffer(sourceExpr);
                var dest = (Expr)Visit(call[TIR.NTT.AffineScatter.Dest], context);

                ValidateRelation(scatter.Relation, source.Dimensions.Length);

                var globalExtents = source.Dimensions.ToArray();
                var iterationExtents = AffineIOLayoutEvaluator.GetIterationExtents(source);
                var symbolMap = BuildSymbolMap(scatter.Relation, scatter.Symbols);
                var loopNest = BuildLoopNest(iterationExtents, loopVars =>
                {
                    var domainValues = AffineIOLayoutEvaluator.GetDomainValues(source, loopVars, globalExtents);
                    var value = T.BufferLoad(source, AffineIOLayoutEvaluator.GetStorageIndices(source, loopVars, domainValues));
                    var address = EvaluateAddress(scatter.Relation, domainValues, globalExtents, symbolMap);
                    var store = T.Store(dest, address, value);
                    return scatter.Relation.Constraint == LogicalExpr.True
                        ? store
                        : T.If(EvaluateConstraint(scatter.Relation.Constraint, domainValues)).Then(store).Build();
                });
                return sourceSetup is null ? loopNest : T.Sequential(sourceSetup, loopNest);
            }

            private Expr BuildLoopNest(Dimension[] extents, Func<DimVar[], Expr> bodyFactory)
            {
                var loopVars = new DimVar[extents.Length];
                return BuildLoopNestRecursive(extents, loopVars, 0, bodyFactory);
            }

            private Expr BuildLoopNestRecursive(IReadOnlyList<Dimension> extents, DimVar[] loopVars, int axis, Func<DimVar[], Expr> bodyFactory)
            {
                if (axis == loopVars.Length)
                {
                    return bodyFactory(loopVars);
                }

                var range = new TIR.Range(Dimension.Zero, extents[axis], Dimension.One);
                var loopBuilder = T.Serial(out loopVars[axis], range, $"d{axis}");
                var inner = BuildLoopNestRecursive(extents, loopVars, axis + 1, bodyFactory);
                return loopBuilder.Body(inner).Build();
            }

            private void ValidateRelation(AffineRelation relation, int expectedRank)
            {
                if (relation.Results.Length != 1)
                {
                    throw new NotSupportedException($"Affine IO lowering expects one address result, got {relation.Results.Length}.");
                }

                if (relation.Domains.Length != expectedRank)
                {
                    throw new NotSupportedException($"Domain rank {relation.Domains.Length} does not match buffer rank {expectedRank}.");
                }
            }

            private Dimension EvaluateAddress(AffineRelation relation, IReadOnlyList<Dimension> domainValues, IReadOnlyList<Dimension> extents, IReadOnlyDictionary<int, Dimension>? symbolMap)
            {
                return EvaluateAffineExpr(relation.Results[0], domainValues, extents, symbolMap);
            }

            private Expr ReadDefaultValue(Expr defaultValue, Expr[] indices, DataType elemType)
            {
                return defaultValue switch
                {
                    TIR.Buffer buffer => T.BufferLoad(buffer, indices),
                    None => Const.FromTensor(Tensor.Zero(elemType)),
                    Expr expr when expr.CheckedType is TensorType { Shape.IsScalar: true } => expr,
                    _ => throw new NotSupportedException($"Unsupported affine gather default value {defaultValue.GetType().Name}."),
                };
            }

            private (Expr DefaultValue, Expr? Setup) PrepareGatherDefault(Expr defaultValue)
            {
                if (defaultValue is None or TIR.Buffer || defaultValue.CheckedType is TensorType { Shape.IsScalar: true })
                {
                    return (defaultValue, null);
                }

                var (buffer, setup) = RequireReadableBuffer(defaultValue, "affine gather default value", "affine_gather_default");
                return (buffer, setup);
            }

            private LogicalExpr EvaluateConstraint(LogicalExpr constraint, IReadOnlyList<Dimension> domainValues)
            {
                return constraint switch
                {
                    LogicalConst logicalConst => logicalConst,
                    DimCompare compare => new DimCompare(compare.Op, EvaluateDimension(compare.Lhs, domainValues), EvaluateDimension(compare.Rhs, domainValues)),
                    LogicalAnd logicalAnd => new LogicalAnd(logicalAnd.Operands.ToArray().Select(x => EvaluateConstraint(x, domainValues)).ToArray()),
                    LogicalOr logicalOr => new LogicalOr(logicalOr.Operands.ToArray().Select(x => EvaluateConstraint(x, domainValues)).ToArray()),
                    _ => throw new NotSupportedException($"Unsupported affine IO constraint node {constraint.GetType().Name}."),
                };
            }

            private Dimension EvaluateDimension(Dimension dim, IReadOnlyList<Dimension> domainValues)
            {
                return dim switch
                {
                    DimConst constant => constant,
                    DimVar dimVar when TryGetDomainIndex(dimVar, domainValues.Count, out var index) => domainValues[index],
                    DimVar dimVar => dimVar,
                    ThreadIdDim threadId => threadId,
                    ProgramIdDim programId => programId,
                    DimSum sum => EvaluateDimSum(sum, domainValues),
                    DimProduct product => EvaluateDimProduct(product, domainValues),
                    DimFraction fraction => new DimFraction(fraction.DivMode, EvaluateDimension(fraction.Numerator, domainValues), EvaluateDimension(fraction.Denominator, domainValues)),
                    DimRemainder remainder => new DimRemainder(EvaluateDimension(remainder.Numerator, domainValues), EvaluateDimension(remainder.Denominator, domainValues)),
                    _ => throw new NotSupportedException($"Unsupported affine IO constraint dimension {dim.GetType().Name}."),
                };
            }

            private Dimension EvaluateDimSum(DimSum sum, IReadOnlyList<Dimension> domainValues)
            {
                Dimension result = sum.Bias;
                foreach (var operand in sum.Operands)
                {
                    result += EvaluateDimension(operand, domainValues);
                }

                return result;
            }

            private Dimension EvaluateDimProduct(DimProduct product, IReadOnlyList<Dimension> domainValues)
            {
                Dimension result = product.Scale;
                foreach (var operand in product.Operands)
                {
                    result *= EvaluateDimension(operand, domainValues);
                }

                return result;
            }

            private bool TryGetDomainIndex(DimVar dimVar, int rank, out int index)
            {
                if (dimVar.Name.Length > 1 && dimVar.Name[0] == 'd' && int.TryParse(dimVar.Name[1..], out index) && index >= 0 && index < rank)
                {
                    return true;
                }

                index = -1;
                return false;
            }

            private Dimension EvaluateAffineExpr(AffineExpr expr, IReadOnlyList<Dimension> dims, IReadOnlyList<Dimension> extents, IReadOnlyDictionary<int, Dimension>? symbols)
            {
                return expr switch
                {
                    AffineConstant constant => constant.Value,
                    AffineDim dim => dims[dim.Position],
                    AffineExtent extent => extents[extent.Position],
                    AffineSymbol symbol => symbols is null ? throw new NotSupportedException("Symbolic relations require bound symbol map.") : symbols[symbol.Position],
                    AffineAddBinary add => EvaluateAffineExpr(add.Lhs, dims, extents, symbols) + EvaluateAffineExpr(add.Rhs, dims, extents, symbols),
                    AffineMulBinary mul => EvaluateAffineExpr(mul.Lhs, dims, extents, symbols) * EvaluateAffineExpr(mul.Rhs, dims, extents, symbols),
                    AffineDivBinary div => ApplyDivBinary(div.BinaryOp, EvaluateAffineExpr(div.Lhs, dims, extents, symbols), EvaluateAffineExpr(div.Rhs, dims, extents, symbols)),
                    _ => throw new NotSupportedException($"Unsupported affine expression node {expr.GetType().Name}"),
                };
            }

            private Dimension ApplyDivBinary(AffineDivBinaryOp op, Dimension lhs, Dimension rhs) => op switch
            {
                AffineDivBinaryOp.FloorDiv => lhs / rhs,
                AffineDivBinaryOp.CeilDiv => Dimension.CeilDiv(lhs, rhs),
                AffineDivBinaryOp.Mod => lhs % rhs,
                _ => throw new ArgumentOutOfRangeException(nameof(op), $"Unsupported affine division operator {op}"),
            };

            private IReadOnlyDictionary<int, Dimension>? BuildSymbolMap(AffineRelation relation, RankedShape symbols)
            {
                if (relation.Symbols.Length == 0)
                {
                    return null;
                }

                var dims = symbols.Dimensions.ToArray();
                if (dims.Length != relation.Symbols.Length)
                {
                    throw new InvalidOperationException("Symbol payload does not match relation requirement.");
                }

                var map = new Dictionary<int, Dimension>(dims.Length);
                for (int i = 0; i < dims.Length; i++)
                {
                    map.Add(relation.Symbols[i].Position, dims[i]);
                }

                return map;
            }

            private TIR.Buffer RequireBuffer(BaseExpr expr)
            {
                if (expr is not TIR.Buffer buffer)
                {
                    throw new NotSupportedException("Affine IO lowering expects buffer arguments.");
                }

                return buffer;
            }

            private (TIR.Buffer Buffer, Expr? Setup) RequireReadableBuffer(BaseExpr expr)
            {
                return RequireReadableBuffer(expr, "affine scatter source", "affine_scatter_source");
            }

            private (TIR.Buffer Buffer, Expr? Setup) RequireReadableBuffer(BaseExpr expr, string role, string bufferNamePrefix)
            {
                if (expr is TIR.Buffer buffer)
                {
                    return (buffer, null);
                }

                if (expr is TensorConst tensorConst)
                {
                    return (T.AttachBuffer(tensorConst, out _, $"{bufferNamePrefix}_{_bufferIndex++}"), null);
                }

                if (expr is Expr sourceExpr)
                {
                    var (tensorType, distributedType) = sourceExpr.CheckedType switch
                    {
                        TensorType { Shape: RankedShape } tt => (tt, null),
                        DistributedType { TensorType: TensorType { Shape: RankedShape } tt } dt => (tt, dt),
                        _ => throw new NotSupportedException($"{role} must be a ranked tensor or buffer."),
                    };
                    if (TileDecisionMetadata.TryGet(sourceExpr, out var tileDecision))
                    {
                        if (tileDecision.Storage.PhysicalLocation is PhysicalMemorySpace.Register)
                        {
                            throw new InvalidOperationException($"{role} has register tile decision {tileDecision.Storage}; affine IO lowering must consume it as SSA/register values instead of materializing an addressable buffer.");
                        }

                        throw new NotSupportedException($"{role} has tile decision {tileDecision.Storage}; affine IO lowering cannot materialize tiled direct-affine values through an addressable temporary. Add a storage-specific lowering for this tile.");
                    }

                    var storage = DefaultTileSourceStorage;
                    var sourceBuffer = T.CreateBuffer(tensorType, storage, out _, $"{bufferNamePrefix}_{_bufferIndex++}", distributedType);
                    return (sourceBuffer, T.Memcopy(sourceBuffer, sourceExpr));
                }

                throw new NotSupportedException($"{role} must be an expression.");
            }
        }
    }
}

internal static class DimVarExtensions
{
    public static Expr[] AsExprs(this IReadOnlyList<DimVar> loopVars)
    {
        var result = new Expr[loopVars.Count];
        for (int i = 0; i < loopVars.Count; i++)
        {
            result[i] = global::Nncase.IR.F.Tensors.Cast(global::Nncase.IR.F.Shapes.AsTensor(loopVars[i]), global::Nncase.DataTypes.Int32);
        }

        return result;
    }

    public static Expr[] AsExprs(this IReadOnlyList<Dimension> dimensions)
    {
        var result = new Expr[dimensions.Count];
        for (int i = 0; i < dimensions.Count; i++)
        {
            result[i] = global::Nncase.IR.F.Tensors.Cast(global::Nncase.IR.F.Shapes.AsTensor(dimensions[i]), global::Nncase.DataTypes.Int32);
        }

        return result;
    }
}
