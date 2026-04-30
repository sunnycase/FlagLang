// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
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
                var iterationExtents = GetIterationExtents(output);
                var symbolMap = BuildSymbolMap(gather.Relation, gather.Symbols);
                Expr? defaultSetup = null;
                if (gather.Relation.Constraint != LogicalExpr.True)
                {
                    (defaultValue, defaultSetup) = PrepareGatherDefault(defaultValue);
                }

                var loopNest = BuildLoopNest(iterationExtents, loopVars =>
                {
                    var domainValues = GetDomainValues(output, loopVars, globalExtents);
                    var address = EvaluateAddress(gather.Relation, domainValues, globalExtents, symbolMap);
                    var loaded = T.Load(source, address);
                    var loopIndices = loopVars.AsExprs();
                    var storageIndices = GetStorageIndices(output, loopVars, domainValues);
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
                var iterationExtents = GetIterationExtents(source);
                var symbolMap = BuildSymbolMap(scatter.Relation, scatter.Symbols);
                var loopNest = BuildLoopNest(iterationExtents, loopVars =>
                {
                    var domainValues = GetDomainValues(source, loopVars, globalExtents);
                    var value = T.BufferLoad(source, GetStorageIndices(source, loopVars, domainValues));
                    var address = EvaluateAddress(scatter.Relation, domainValues, globalExtents, symbolMap);
                    var store = T.Store(dest, address, value);
                    return scatter.Relation.Constraint == LogicalExpr.True
                        ? store
                        : T.If(EvaluateConstraint(scatter.Relation.Constraint, domainValues)).Then(store).Build();
                });
                return sourceSetup is null ? loopNest : T.Sequential(sourceSetup, loopNest);
            }

            private Expr[] GetStorageIndices(TIR.Buffer buffer, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> domainValues)
            {
                if (buffer.DistributedType is not DistributedType distributedType)
                {
                    return loopVars.AsExprs();
                }

                ValidateDistributedBuffer(buffer, distributedType);
                var storageLayout = distributedType.StorageLayout;
                if (IsIdentityStorageMap(storageLayout.LogicalToPhysical, loopVars.Count))
                {
                    return loopVars.AsExprs();
                }

                var bindings = BuildOwnerLocalBindings(buffer, distributedType, loopVars);
                if (storageLayout.ViewMap is not null)
                {
                    var viewValues = EvaluateMap(storageLayout.ViewMap, bindings, $"storage view map {storageLayout.ViewMap.Name} for buffer {buffer.Name}");
                    foreach (var value in viewValues)
                    {
                        bindings[value.Name] = value.Value;
                    }
                }
                else
                {
                    for (int i = 0; i < domainValues.Count; i++)
                    {
                        bindings[$"g{i}"] = domainValues[i];
                    }
                }

                return EvaluateMap(storageLayout.LogicalToPhysical, bindings, $"storage map {storageLayout.LogicalToPhysical.Name} for buffer {buffer.Name}")
                    .Select(value => value.Value)
                    .ToArray()
                    .AsExprs();
            }

            private bool IsIdentityStorageMap(IndexMapDescriptor map, int rank)
            {
                if (map.Inputs.Count != rank || map.Outputs.Count != rank)
                {
                    return false;
                }

                for (int i = 0; i < rank; i++)
                {
                    if (map.Inputs[i] != $"l{i}" ||
                        map.Outputs[i].Name != $"p{i}" ||
                        map.Outputs[i].Expr is not IndexVar { Name: var name } ||
                        name != $"l{i}")
                    {
                        return false;
                    }
                }

                return true;
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

            private Dimension[] GetIterationExtents(TIR.Buffer buffer)
            {
                var globalExtents = buffer.Dimensions.ToArray();
                if (buffer.DistributedType is not DistributedType distributedType)
                {
                    return globalExtents;
                }

                ValidateDistributedBuffer(buffer, distributedType);
                if (distributedType.ExplicitDistributionLayout is not null)
                {
                    return GetExplicitIterationExtents(buffer, distributedType);
                }

                return globalExtents.Select((extent, axis) => GetLocalShardExtent(distributedType, axis, extent)).ToArray();
            }

            private Dimension[] GetDomainValues(TIR.Buffer buffer, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> globalExtents)
            {
                if (buffer.DistributedType is not DistributedType distributedType)
                {
                    return loopVars.Select<DimVar, Dimension>(loopVar => loopVar).ToArray();
                }

                ValidateDistributedBuffer(buffer, distributedType);
                if (distributedType.ExplicitDistributionLayout is not null)
                {
                    return GetExplicitDomainValues(buffer, distributedType, loopVars);
                }

                var domainValues = new Dimension[loopVars.Count];
                for (int axis = 0; axis < loopVars.Count; axis++)
                {
                    domainValues[axis] = loopVars[axis] + GetShardOffset(distributedType, axis, globalExtents[axis]);
                }

                return domainValues;
            }

            private void ValidateDistributedBuffer(TIR.Buffer buffer, DistributedType distributedType)
            {
                if (distributedType.Partial)
                {
                    throw new NotSupportedException("Affine IO lowering cannot directly lower partial distributed buffers. Resolve Partial with a distributed reduction before affine IO lowering.");
                }

                if (distributedType.TensorType.Shape.Rank != buffer.Rank || distributedType.AxisPolicies.Count != buffer.Rank)
                {
                    throw new NotSupportedException($"Distributed buffer {buffer.Name} has rank {buffer.Rank}, but its distributed type has tensor rank {distributedType.TensorType.Shape.Rank} and {distributedType.AxisPolicies.Count} axis policies.");
                }

                if (distributedType.ExplicitDistributionLayout is not null)
                {
                    LayoutVerifier.Verify(distributedType, $"Affine IO lowering buffer {buffer.Name}");
                    if (distributedType.DistributionLayout.LocalShape.Rank != buffer.Rank)
                    {
                        throw new NotSupportedException($"Distributed buffer {buffer.Name} has rank {buffer.Rank}, but explicit layout {distributedType.DistributionLayout.Kind} has local shape {distributedType.DistributionLayout.LocalShape}.");
                    }
                }
            }

            private Dimension[] GetExplicitIterationExtents(TIR.Buffer buffer, DistributedType distributedType)
            {
                var layout = distributedType.DistributionLayout;
                ValidateSupportedExplicitMap(buffer, layout);
                return layout.LocalShape.ToArray();
            }

            private Dimension[] GetExplicitDomainValues(TIR.Buffer buffer, DistributedType distributedType, IReadOnlyList<DimVar> loopVars)
            {
                var layout = distributedType.DistributionLayout;
                ValidateSupportedExplicitMap(buffer, layout);
                return EvaluateMap(layout.OwnerLocalToGlobal, BuildOwnerLocalBindings(buffer, distributedType, loopVars), $"distribution map {layout.OwnerLocalToGlobal.Name} for buffer {buffer.Name}")
                    .Select(value => value.Value)
                    .ToArray();
            }

            private void ValidateSupportedExplicitMap(TIR.Buffer buffer, DistributionLayout layout)
            {
                if (layout.Kind is not ("SBP" or "TritonBlocked"))
                {
                    throw new NotSupportedException($"Affine IO lowering cannot evaluate explicit distribution layout {layout.Kind} for buffer {buffer.Name}. Add a DistributionLayout evaluator for this map instead of falling back to AxisPolicies.");
                }

                if (layout.Kind == "TritonBlocked" && layout.LocalShape is not { IsUnranked: false, Rank: 1 })
                {
                    throw new NotSupportedException($"Affine IO lowering supports explicit {layout.Kind} layout only for rank-1 buffers, got {layout.LocalShape} on {buffer.Name}.");
                }
            }

            private Dimension GetLocalShardExtent(DistributedType distributedType, int tensorAxis, Dimension globalExtent)
            {
                return distributedType.AxisPolicies[tensorAxis] switch
                {
                    SBPBroadCast => globalExtent,
                    SBPSplit split => GetSplitLocalShardExtent(distributedType, split, globalExtent),
                    SBPPartial partial => throw new NotSupportedException($"Affine IO lowering cannot directly lower partial shard policy {partial}. Resolve Partial before affine IO lowering."),
                    SBP policy => throw new NotSupportedException($"Unsupported affine IO shard policy {policy.GetType().Name}."),
                };
            }

            private Dimension GetSplitLocalShardExtent(DistributedType distributedType, SBPSplit split, Dimension globalExtent)
            {
                var maxLocalExtent = Dimension.CeilDiv(globalExtent, GetSplitDivisor(distributedType, split));
                var offset = GetSplitShardOffset(distributedType, split, globalExtent);
                return Dimension.Min(maxLocalExtent, globalExtent - offset);
            }

            private Dimension GetShardOffset(DistributedType distributedType, int tensorAxis, Dimension globalExtent)
            {
                return distributedType.AxisPolicies[tensorAxis] switch
                {
                    SBPBroadCast => Dimension.Zero,
                    SBPSplit split => GetSplitShardOffset(distributedType, split, globalExtent),
                    SBPPartial partial => throw new NotSupportedException($"Affine IO lowering cannot directly lower partial shard policy {partial}. Resolve Partial before affine IO lowering."),
                    SBP policy => throw new NotSupportedException($"Unsupported affine IO shard policy {policy.GetType().Name}."),
                };
            }

            private Dimension GetSplitShardOffset(DistributedType distributedType, SBPSplit split, Dimension globalExtent)
            {
                var maxLocalExtent = Dimension.CeilDiv(globalExtent, GetSplitDivisor(distributedType, split));
                return Dimension.Min(maxLocalExtent * GetSplitLinearShardIndex(distributedType, split), globalExtent);
            }

            private Dimension GetSplitDivisor(DistributedType distributedType, SBPSplit split)
            {
                Dimension divisor = Dimension.One;
                foreach (var meshAxis in split.Axes)
                {
                    ValidateMeshAxis(distributedType.Placement, meshAxis);
                    divisor *= distributedType.Placement.Hierarchy[meshAxis];
                }

                return divisor;
            }

            private Dimension GetSplitLinearShardIndex(DistributedType distributedType, SBPSplit split)
            {
                Dimension linearIndex = Dimension.Zero;
                foreach (var meshAxis in split.Axes)
                {
                    ValidateMeshAxis(distributedType.Placement, meshAxis);
                    linearIndex = (linearIndex * distributedType.Placement.Hierarchy[meshAxis]) + GetMeshAxisIndex(distributedType.Placement, meshAxis);
                }

                return linearIndex;
            }

            private void ValidateMeshAxis(Placement placement, int meshAxis)
            {
                if (meshAxis < 0 || meshAxis >= placement.Rank || meshAxis >= placement.Name.Length)
                {
                    throw new NotSupportedException($"Invalid distributed mesh axis {meshAxis} for placement {placement}.");
                }
            }

            private Dimension GetMeshAxisIndex(Placement placement, int meshAxis)
            {
                return placement.Name[meshAxis] switch
                {
                    't' => IR.F.Distributed.ThreadId(),
                    'b' => IR.F.Distributed.ProgramId(0),
                    var name => throw new NotSupportedException($"Affine IO lowering only supports thread ('t') and block ('b') mesh axes, got '{name}' in placement {placement}."),
                };
            }

            private Dimension GetTritonBlockedDomainValue(DistributionLayout distributionLayout, DimVar elem)
            {
                var layout = ParseTritonBlockedLayout(distributionLayout);
                if (distributionLayout.LocalShape[0] != layout.SizePerThread)
                {
                    throw new NotSupportedException($"TritonBlocked affine IO lowering requires LocalShape=[sizePerThread], got LocalShape={distributionLayout.LocalShape}, sizePerThread={layout.SizePerThread}.");
                }

                var threadId = IR.F.Distributed.ThreadId();
                var threadsPerCTA = (Dimension)(layout.ThreadsPerWarp * layout.WarpsPerCTA);
                var ctaElements = (Dimension)(layout.SizePerThread * layout.ThreadsPerWarp * layout.WarpsPerCTA);
                var ctaBase = IR.F.Distributed.ProgramId(0) * ctaElements;
                var local = layout.ThreadElementOrder switch
                {
                    TritonThreadElementOrder.Contiguous => (threadId * layout.SizePerThread) + elem,
                    TritonThreadElementOrder.Strided => threadId + (elem * threadsPerCTA),
                    _ => throw new NotSupportedException($"Unsupported TritonBlocked thread element order {layout.ThreadElementOrder}."),
                };
                return ctaBase + local;
            }

            private TritonBlockedLayout ParseTritonBlockedLayout(DistributionLayout distributionLayout)
            {
                if (distributionLayout.Attributes is null)
                {
                    throw new NotSupportedException("TritonBlocked affine IO lowering requires layout attributes.");
                }

                return new TritonBlockedLayout(
                    SizePerThread: RequireTritonBlockedIntAttribute(distributionLayout, "sizePerThread"),
                    ThreadsPerWarp: RequireTritonBlockedIntAttribute(distributionLayout, "threadsPerWarp"),
                    WarpsPerCTA: RequireTritonBlockedIntAttribute(distributionLayout, "warpsPerCTA"),
                    Order: RequireTritonBlockedIntListAttribute(distributionLayout, "order"),
                    CTAsPerCGA: RequireTritonBlockedIntListAttribute(distributionLayout, "ctasPerCGA"),
                    CTASplitNum: RequireTritonBlockedIntListAttribute(distributionLayout, "ctaSplitNum"),
                    CTAOrder: RequireTritonBlockedIntListAttribute(distributionLayout, "ctaOrder"),
                    ThreadElementOrder: RequireTritonBlockedThreadOrderAttribute(distributionLayout));
            }

            private int RequireTritonBlockedIntAttribute(DistributionLayout distributionLayout, string name)
            {
                var value = RequireTritonBlockedAttribute(distributionLayout, name);
                return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                    ? parsed
                    : throw new NotSupportedException($"TritonBlocked layout attribute {name} must be an integer, got '{value}'.");
            }

            private IRArray<int> RequireTritonBlockedIntListAttribute(DistributionLayout distributionLayout, string name)
            {
                var value = RequireTritonBlockedAttribute(distributionLayout, name);
                if (!value.StartsWith("[", StringComparison.Ordinal) || !value.EndsWith("]", StringComparison.Ordinal))
                {
                    throw new NotSupportedException($"TritonBlocked layout attribute {name} must be an integer list, got '{value}'.");
                }

                var body = value[1..^1];
                if (string.IsNullOrWhiteSpace(body))
                {
                    return Array.Empty<int>();
                }

                return body.Split(',', StringSplitOptions.TrimEntries)
                    .Select(item => int.TryParse(item, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                        ? parsed
                        : throw new NotSupportedException($"TritonBlocked layout attribute {name} contains non-integer item '{item}'."))
                    .ToArray();
            }

            private TritonThreadElementOrder RequireTritonBlockedThreadOrderAttribute(DistributionLayout distributionLayout)
            {
                var value = RequireTritonBlockedAttribute(distributionLayout, "threadElementOrder");
                return Enum.TryParse<TritonThreadElementOrder>(value, ignoreCase: false, out var parsed)
                    ? parsed
                    : throw new NotSupportedException($"TritonBlocked layout has unsupported threadElementOrder '{value}'.");
            }

            private string RequireTritonBlockedAttribute(DistributionLayout distributionLayout, string name)
            {
                var prefix = name + "=";
                var matches = distributionLayout.Attributes!.Value.Where(attr => attr.StartsWith(prefix, StringComparison.Ordinal)).ToArray();
                return matches.Length == 1
                    ? matches[0][prefix.Length..]
                    : throw new NotSupportedException($"TritonBlocked layout must provide exactly one {name} attribute, got {matches.Length}.");
            }

            private Dictionary<string, Dimension> BuildOwnerLocalBindings(TIR.Buffer buffer, DistributedType distributedType, IReadOnlyList<DimVar> loopVars)
            {
                var bindings = new Dictionary<string, Dimension>(StringComparer.Ordinal);
                for (int axis = 0; axis < distributedType.Placement.Rank; axis++)
                {
                    bindings[$"owner{axis}"] = GetMeshAxisIndex(distributedType.Placement, axis);
                }

                for (int i = 0; i < loopVars.Count; i++)
                {
                    bindings[$"l{i}"] = loopVars[i];
                    bindings[$"d{i}"] = loopVars[i];
                }

                if (distributedType.DistributionLayout.Kind == "TritonBlocked")
                {
                    BindTritonBlockedOwnerCoordinates(distributedType.DistributionLayout, bindings);
                    if (loopVars.Count != 1)
                    {
                        throw new NotSupportedException($"TritonBlocked affine IO lowering requires one loop coordinate for buffer {buffer.Name}, got {loopVars.Count}.");
                    }

                    bindings["elem"] = loopVars[0];
                }

                return bindings;
            }

            private void BindTritonBlockedOwnerCoordinates(DistributionLayout layout, IDictionary<string, Dimension> bindings)
            {
                var tritonLayout = ParseTritonBlockedLayout(layout);
                var threadId = IR.F.Distributed.ThreadId();
                bindings["cta"] = IR.F.Distributed.ProgramId(0);
                bindings["warp"] = threadId / (Dimension)tritonLayout.ThreadsPerWarp;
                bindings["lane"] = threadId % (Dimension)tritonLayout.ThreadsPerWarp;
            }

            private IReadOnlyList<(string Name, Dimension Value)> EvaluateMap(IndexMapDescriptor map, IReadOnlyDictionary<string, Dimension> bindings, string context)
            {
                foreach (var input in map.Inputs)
                {
                    if (!bindings.ContainsKey(input))
                    {
                        throw new NotSupportedException($"Cannot evaluate {context}: input {input} is not bound.");
                    }
                }

                var outputs = new List<(string Name, Dimension Value)>(map.Outputs.Count);
                foreach (var output in map.Outputs)
                {
                    outputs.Add((output.Name, EvaluateIndexExpr(output.Expr, bindings, context)));
                }

                return outputs;
            }

            private Dimension EvaluateIndexExpr(IndexExpr expr, IReadOnlyDictionary<string, Dimension> bindings, string context)
            {
                return expr switch
                {
                    IndexVar variable => bindings.TryGetValue(variable.Name, out var value)
                        ? value
                        : throw new NotSupportedException($"Cannot evaluate {context}: variable {variable.Name} is not bound."),
                    IndexConst constant => constant.Value,
                    IndexAdd add => EvaluateIndexAdd(add, bindings, context),
                    IndexMul mul => EvaluateIndexMul(mul, bindings, context),
                    IndexFloorDiv floorDiv => EvaluateIndexExpr(floorDiv.Value, bindings, context) / EvaluateIndexExpr(floorDiv.Divisor, bindings, context),
                    IndexMod mod => EvaluateIndexExpr(mod.Value, bindings, context) % EvaluateIndexExpr(mod.Divisor, bindings, context),
                    IndexAny => throw new NotSupportedException($"Cannot evaluate {context}: wildcard index output requires an explicit binding."),
                    IndexNamedPrimitive named => throw new NotSupportedException($"Cannot evaluate {context}: named primitive {named.Name} has no affine IO evaluator."),
                    _ => throw new NotSupportedException($"Cannot evaluate {context}: unsupported index expression {expr.GetType().Name}."),
                };
            }

            private Dimension EvaluateIndexAdd(IndexAdd add, IReadOnlyDictionary<string, Dimension> bindings, string context)
            {
                Dimension result = Dimension.Zero;
                foreach (var term in add.Terms)
                {
                    result += EvaluateIndexExpr(term, bindings, context);
                }

                return result;
            }

            private Dimension EvaluateIndexMul(IndexMul mul, IReadOnlyDictionary<string, Dimension> bindings, string context)
            {
                Dimension result = Dimension.One;
                foreach (var factor in mul.Factors)
                {
                    result *= EvaluateIndexExpr(factor, bindings, context);
                }

                return result;
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
