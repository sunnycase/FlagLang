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
using Nncase.Targets;
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
        private readonly AffineIOLoweringRewriter _rewriter;

        public NTTAffineIOLoweringPass()
        {
            var targetOptions = CompileSession.CompileOptions.TargetOptions as INTTTargetOptions;
            if (targetOptions is null && CompileSession.Target is NTTTarget)
            {
                targetOptions = new NTTTargetOptions();
            }

            _rewriter = new(targetOptions);
        }

        public NTTAffineIOLoweringPass(INTTTargetOptions targetOptions)
        {
            _rewriter = new(targetOptions);
        }

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
                    if (i + 1 < fields.Length && TryFuseAffineGatherTensorLoad(fields, i, out var fused))
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

            private static bool TryFuseAffineGatherTensorLoad(IReadOnlyList<Expr> fields, int index, [MaybeNullWhen(false)] out Expr fused)
            {
                fused = null;
                var first = fields[index];
                var second = fields[index + 1];
                if (first is not Call { Target: TIR.NTT.AffineGather gather } gatherCall ||
                    second is not Call { Target: TIR.NTT.TensorLoad tensorLoad } tensorLoadCall ||
                    tensorLoadCall[TIR.NTT.TensorLoad.Dest] is not TIR.Buffer { Type: DistributedType } distributedOutput ||
                    !Equals(gatherCall[TIR.NTT.AffineGather.Output], tensorLoadCall[TIR.NTT.TensorLoad.Src]))
                {
                    return false;
                }

                if (HasOtherUses(fields, (Expr)gatherCall[TIR.NTT.AffineGather.Output], index, index + 1))
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

            private static bool HasOtherUses(IReadOnlyList<Expr> fields, Expr value, int gatherIndex, int tensorLoadIndex)
            {
                for (int i = 0; i < fields.Count; i++)
                {
                    if (i == gatherIndex || i == tensorLoadIndex)
                    {
                        continue;
                    }

                    if (ExprCollector.Collect(fields[i]).Any(expr => ReferenceEquals(expr, value)))
                    {
                        return true;
                    }
                }

                return false;
            }
        }

        private sealed class AffineIOLoweringRewriter : ExprRewriter<Unit>
        {
            private static readonly BufferStorage DefaultTileSourceStorage = new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable);

            private readonly INTTTargetOptions? _targetOptions;
            private readonly Dictionary<Expr, TensorView> _tensorViewBindings = new(ReferenceEqualityComparer.Instance);
            private int _bufferIndex;

            public AffineIOLoweringRewriter(INTTTargetOptions? targetOptions)
            {
                _targetOptions = targetOptions;
            }

            protected override BaseExpr RewriteLeafCall(Call expr, Unit context)
            {
                return expr.Target switch
                {
                    TIR.NTT.AffineGather gather => LowerGather(expr, gather, context),
                    TIR.NTT.AffineScatter scatter => LowerScatter(expr, scatter, context),
                    _ => base.RewriteLeafCall(expr, context),
                };
            }

            protected override BaseExpr VisitLet(Let expr, Unit context)
            {
                var expression = Visit(expr.Expression, context);
                TensorView? oldBinding = null;
                var hadOldBinding = false;
                var pushedBinding = false;
                var bindingVar = expr.Var as Expr;
                if (bindingVar is not null && TryCreateTensorViewBinding(bindingVar, expression, out var tensorView))
                {
                    hadOldBinding = _tensorViewBindings.TryGetValue(bindingVar, out oldBinding);
                    _tensorViewBindings[bindingVar] = tensorView;
                    pushedBinding = true;
                }

                Sequential body;
                try
                {
                    body = (Sequential)Visit(expr.Body, context);
                }
                finally
                {
                    if (pushedBinding && bindingVar is not null)
                    {
                        if (hadOldBinding)
                        {
                            _tensorViewBindings[bindingVar] = oldBinding!;
                        }
                        else
                        {
                            _tensorViewBindings.Remove(bindingVar);
                        }
                    }
                }

                return expr.With(expression: expression, body: body);
            }

            private Expr LowerGather(Call call, TIR.NTT.AffineGather gather, Unit context)
            {
                var source = (Expr)Visit(call[TIR.NTT.AffineGather.Source], context);
                var defaultValue = (Expr)Visit(call[TIR.NTT.AffineGather.DefaultValue], context);
                var output = AnalyzeTensorView(Visit(call[TIR.NTT.AffineGather.Output], context), "affine gather output");

                ValidateRelation(gather.Relation, output.Rank);

                var globalExtents = GetGlobalExtents(gather.Shape, output);
                var useUnrolledLoop = RequiresUnrolledIteration(output, "affine gather output");
                var iterationExtents = useUnrolledLoop
                    ? GetUnrolledIterationExtents(output, "affine gather output")
                    : GetIterationExtents(output);
                var symbolMap = BuildSymbolMap(gather.Relation, gather.Symbols);
                Expr? defaultSetup = null;
                if (gather.Relation.Constraint != LogicalExpr.True)
                {
                    (defaultValue, defaultSetup) = PrepareGatherDefault(defaultValue);
                }

                var loopNest = BuildLoopNest(
                    iterationExtents,
                    loopVars =>
                    {
                        var domainValues = GetDomainValues(output, loopVars, globalExtents);
                        var address = EvaluateAddress(gather.Relation, domainValues, globalExtents, symbolMap);
                        var loaded = T.Load(source, address);
                        var storageIndices = GetStorageIndices(output, loopVars, domainValues);
                        var storeLoaded = BufferStore(output.Expr, storageIndices, loaded);
                        if (gather.Relation.Constraint == LogicalExpr.True)
                        {
                            return storeLoaded;
                        }

                        var fallback = ReadDefaultValue(defaultValue, loopVars, domainValues, output.ElemType);
                        var storeFallback = BufferStore(output.Expr, storageIndices, fallback);
                        return T.If(EvaluateConstraint(gather.Relation.Constraint, domainValues)).Then(storeLoaded).Else(storeFallback).Build();
                    },
                    useUnrolledLoop ? LoopMode.Unrolled : LoopMode.Serial);
                return defaultSetup is null ? loopNest : T.Sequential(defaultSetup, loopNest);
            }

            private Expr LowerScatter(Call call, TIR.NTT.AffineScatter scatter, Unit context)
            {
                var sourceExpr = Visit(call[TIR.NTT.AffineScatter.Source], context);
                var (source, sourceSetup) = RequireReadableView(sourceExpr);
                var dest = (Expr)Visit(call[TIR.NTT.AffineScatter.Dest], context);

                ValidateRelation(scatter.Relation, source.Rank);

                var globalExtents = GetGlobalExtents(scatter.Shape, source);
                var useUnrolledLoop = RequiresUnrolledIteration(source, "affine scatter source");
                var iterationExtents = useUnrolledLoop
                    ? GetUnrolledIterationExtents(source, "affine scatter source")
                    : GetIterationExtents(source);
                var symbolMap = BuildSymbolMap(scatter.Relation, scatter.Symbols);
                var loopNest = BuildLoopNest(
                    iterationExtents,
                    loopVars =>
                    {
                        var domainValues = GetDomainValues(source, loopVars, globalExtents);
                        var value = BufferLoad(source.Expr, GetStorageIndices(source, loopVars, domainValues));
                        var address = EvaluateAddress(scatter.Relation, domainValues, globalExtents, symbolMap);
                        var store = T.Store(dest, address, value);
                        return scatter.Relation.Constraint == LogicalExpr.True
                            ? store
                            : T.If(EvaluateConstraint(scatter.Relation.Constraint, domainValues)).Then(store).Build();
                    },
                    useUnrolledLoop ? LoopMode.Unrolled : LoopMode.Serial);
                return sourceSetup is null ? loopNest : T.Sequential(sourceSetup, loopNest);
            }

            private Expr BuildLoopNest(Dimension[] extents, Func<DimVar[], Expr> bodyFactory, LoopMode loopMode)
            {
                var loopVars = new DimVar[extents.Length];
                return BuildLoopNestRecursive(extents, loopVars, 0, bodyFactory, loopMode);
            }

            private Expr BuildLoopNestRecursive(IReadOnlyList<Dimension> extents, DimVar[] loopVars, int axis, Func<DimVar[], Expr> bodyFactory, LoopMode loopMode)
            {
                if (axis == loopVars.Length)
                {
                    return bodyFactory(loopVars);
                }

                var range = new TIR.Range(Dimension.Zero, extents[axis], Dimension.One);
                var loopBuilder = loopMode switch
                {
                    LoopMode.Serial => T.Serial(out loopVars[axis], range, $"d{axis}"),
                    LoopMode.Unrolled => T.Unrolled(out loopVars[axis], range, $"d{axis}"),
                    _ => throw new NotSupportedException($"Affine IO lowering does not support loop mode {loopMode}."),
                };
                var inner = BuildLoopNestRecursive(extents, loopVars, axis + 1, bodyFactory, loopMode);
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

            private Expr ReadDefaultValue(Expr defaultValue, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> domainValues, DataType elemType)
            {
                return defaultValue switch
                {
                    TIR.Buffer buffer => T.BufferLoad(buffer, GetDefaultValueIndices(buffer, loopVars, domainValues)),
                    None => Const.FromTensor(Tensor.Zero(elemType)),
                    Expr expr when expr.CheckedType is TensorType { Shape.IsScalar: true } => expr,
                    _ => throw new NotSupportedException($"Unsupported affine gather default value {defaultValue.GetType().Name}."),
                };
            }

            private Expr[] GetDefaultValueIndices(TIR.Buffer buffer, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> domainValues)
            {
                if (buffer.Type is DistributedType)
                {
                    return AffineIOLayoutEvaluator.GetStorageIndices(buffer, loopVars, domainValues);
                }

                return domainValues.ToArray().AsExprs();
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

            private Expr BufferLoad(Expr tensor, Expr[] indices) =>
                new Call(new IR.Buffers.BufferLoad(), tensor, new IR.Tuple(indices));

            private Expr BufferStore(Expr tensor, Expr[] indices, Expr value) =>
                new Call(new IR.Buffers.BufferStore(), tensor, new IR.Tuple(indices), value);

            private Dimension[] GetGlobalExtents(Shape shape, TensorView view)
            {
                if (shape is not { IsUnranked: false } rankedShape)
                {
                    return view.Shape;
                }

                var extents = rankedShape.ToArray();
                if (extents.Length != view.Rank)
                {
                    throw new InvalidOperationException($"Affine IO global shape rank {extents.Length} does not match view rank {view.Rank}.");
                }

                return extents;
            }

            private Dimension[] GetIterationExtents(TensorView view)
            {
                if (view.DistributedTensorType is DistributedType distributedType &&
                    view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetIterationExtents(distributedType, $"affine IO view {view.Expr}");
                }

                if (view.Buffer is TIR.Buffer buffer && view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetIterationExtents(buffer);
                }

                return view.Shape;
            }

            private bool RequiresUnrolledIteration(TensorView view, string context)
            {
                if (view.Buffer is not TIR.Buffer buffer)
                {
                    return false;
                }

                var storage = buffer.Storage.WithoutAlignment();
                var memoryLevel = TryResolveMemoryLevel(storage, context);
                return memoryLevel is { IsAddressable: false };
            }

            private Dimension[] GetUnrolledIterationExtents(TensorView view, string context)
            {
                foreach (var extent in view.Shape)
                {
                    if (!extent.IsFixed)
                    {
                        throw new InvalidOperationException($"{context} uses non-addressable storage and requires fixed local tile extents for T.Unrolled, got {extent} in shape [{string.Join(", ", view.Shape.Select(x => x.ToString()))}].");
                    }
                }

                return view.Shape;
            }

            private MemoryHierarchyLevel? TryResolveMemoryLevel(BufferStorage storage, string context)
            {
                if (_targetOptions is null)
                {
                    throw new InvalidOperationException($"{context} uses buffer storage {storage}, but NTTAffineIOLoweringPass has no INTTTargetOptions. Pass target memory hierarchy attributes into affine IO lowering.");
                }

                var levels = _targetOptions.MemoryHierarchyLevels;
                if ((uint)storage.Hierarchy < (uint)levels.Length)
                {
                    var indexedLevel = levels[storage.Hierarchy];
                    if (indexedLevel.Scope == storage.Scope && indexedLevel.PhysicalLocation == storage.PhysicalLocation)
                    {
                        return indexedLevel;
                    }
                }

                var matchingLevelIndex = Array.FindIndex(
                    levels,
                    level => level.Scope == storage.Scope && level.PhysicalLocation == storage.PhysicalLocation);
                if (matchingLevelIndex >= 0)
                {
                    throw new InvalidOperationException(
                        $"{context} uses storage {storage}, but matching target hierarchy level is {matchingLevelIndex}: {levels[matchingLevelIndex]}. " +
                        "Buffer storage hierarchy index and target memory hierarchy attributes must agree.");
                }

                return null;
            }

            private Dimension[] GetDomainValues(TensorView view, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> globalExtents)
            {
                if (view.DistributedTensorType is DistributedType distributedType &&
                    view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetDomainValues(distributedType, loopVars, globalExtents, $"affine IO view {view.Expr}");
                }

                if (view.Buffer is TIR.Buffer buffer && view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetDomainValues(buffer, loopVars, globalExtents);
                }

                var domainValues = new Dimension[loopVars.Count];
                for (int i = 0; i < loopVars.Count; i++)
                {
                    domainValues[i] = loopVars[i] + view.GlobalOffsets[i];
                }

                return domainValues;
            }

            private Expr[] GetStorageIndices(TensorView view, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> domainValues)
            {
                if (view.DistributedTensorType is DistributedType distributedType &&
                    view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetStorageIndices(distributedType, loopVars, domainValues, $"affine IO view {view.Expr}");
                }

                if (view.Buffer is TIR.Buffer buffer && view.GlobalOffsets.All(offset => offset == Dimension.Zero))
                {
                    return AffineIOLayoutEvaluator.GetStorageIndices(buffer, loopVars, domainValues);
                }

                return loopVars.AsExprs();
            }

            private TensorView AnalyzeTensorView(BaseExpr expr, string role)
            {
                if (expr is Expr exprNode && _tensorViewBindings.TryGetValue(exprNode, out var boundView))
                {
                    return boundView with { Expr = exprNode };
                }

                if (expr is TIR.Buffer buffer)
                {
                    return new TensorView(buffer, buffer, buffer.Type as DistributedType, buffer.Dimensions.ToArray(), Enumerable.Repeat<Dimension>(Dimension.Zero, buffer.Rank).ToArray());
                }

                if (expr is Call { Target: IR.Buffers.AllocateBufferView } allocateBufferView &&
                    allocateBufferView[IR.Buffers.AllocateBufferView.Buffer] is TIR.Buffer allocatedBuffer)
                {
                    return CreateTensorViewBinding((Expr)expr, allocatedBuffer);
                }

                if (expr is Call { Target: IR.Buffers.BufferSubview } subview)
                {
                    var parent = AnalyzeTensorView(subview[IR.Buffers.BufferSubview.Buffer], role);
                    var offsets = ((RankedShape)subview[IR.Buffers.BufferSubview.Offset]).ToArray();
                    var shape = ((RankedShape)subview[IR.Buffers.BufferSubview.Shape]).ToArray();
                    if (offsets.Length != parent.Rank || shape.Length != parent.Rank)
                    {
                        throw new InvalidOperationException($"{role} subview rank does not match parent rank.");
                    }

                    var globalOffsets = new Dimension[offsets.Length];
                    for (int i = 0; i < offsets.Length; i++)
                    {
                        globalOffsets[i] = parent.GlobalOffsets[i] + offsets[i];
                    }

                    return new TensorView((Expr)expr, parent.Buffer, parent.DistributedTensorType, shape, globalOffsets);
                }

                if (expr is Expr tensorExpr && tensorExpr.CheckedShape is { IsUnranked: false } shapeExpr)
                {
                    var distributedType = tensorExpr.CheckedType as DistributedType;
                    return new TensorView(tensorExpr, null, distributedType, shapeExpr.ToArray(), Enumerable.Repeat<Dimension>(Dimension.Zero, shapeExpr.Rank).ToArray());
                }

                throw new NotSupportedException($"{role} must be a ranked tensor view, got {expr.GetType().Name}.");
            }

            private bool TryCreateTensorViewBinding(Expr varExpr, BaseExpr expression, [MaybeNullWhen(false)] out TensorView tensorView)
            {
                if (expression is Call { Target: IR.Buffers.AllocateBufferView } allocateBufferView &&
                    allocateBufferView[IR.Buffers.AllocateBufferView.Buffer] is TIR.Buffer buffer)
                {
                    tensorView = CreateTensorViewBinding(varExpr, buffer);
                    return true;
                }

                tensorView = null;
                return false;
            }

            private TensorView CreateTensorViewBinding(Expr expr, TIR.Buffer buffer) =>
                new(expr, buffer, buffer.Type as DistributedType, buffer.Dimensions.ToArray(), Enumerable.Repeat<Dimension>(Dimension.Zero, buffer.Rank).ToArray());

            private TIR.Buffer RequireBuffer(BaseExpr expr)
            {
                if (expr is not TIR.Buffer buffer)
                {
                    throw new NotSupportedException("Affine IO lowering expects buffer arguments.");
                }

                return buffer;
            }

            private (TensorView View, Expr? Setup) RequireReadableView(BaseExpr expr)
            {
                if (expr is TensorConst tensorConst)
                {
                    var buffer = T.AttachBuffer(tensorConst, out _, $"affine_scatter_source_{_bufferIndex++}");
                    return (AnalyzeTensorView(buffer, "affine scatter source"), null);
                }

                if (expr is TIR.Buffer ||
                    expr is Call { Target: IR.Buffers.AllocateBufferView or IR.Buffers.BufferSubview } ||
                    (expr is Expr exprNode && _tensorViewBindings.ContainsKey(exprNode)))
                {
                    return (AnalyzeTensorView(expr, "affine scatter source"), null);
                }

                if (expr is Expr sourceExpr &&
                    sourceExpr.CheckedType is not TensorType { Shape: RankedShape } &&
                    sourceExpr.CheckedType is not DistributedType { TensorType: TensorType { Shape: RankedShape } })
                {
                    throw new NotSupportedException($"affine scatter source must be a ranked tensor or buffer, got {sourceExpr.CheckedType}.");
                }

                var (sourceBuffer, setup) = RequireReadableBuffer(expr);
                return (AnalyzeTensorView(sourceBuffer, "affine scatter source"), setup);
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
                    IRType sourceBufferType = distributedType is null ? tensorType : distributedType;
                    var sourceBuffer = T.CreateBuffer(sourceBufferType, storage, out _, $"{bufferNamePrefix}_{_bufferIndex++}");
                    return (sourceBuffer, T.Memcopy(sourceBuffer, sourceExpr));
                }

                throw new NotSupportedException($"{role} must be an expression.");
            }

            private sealed record TensorView(Expr Expr, TIR.Buffer? Buffer, DistributedType? DistributedTensorType, Dimension[] Shape, Dimension[] GlobalOffsets)
            {
                public int Rank => Shape.Length;

                public DataType ElemType => Expr.CheckedDataType;
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
