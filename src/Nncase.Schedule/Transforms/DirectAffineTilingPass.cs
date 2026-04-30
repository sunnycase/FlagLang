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
    private const int CudaWarpLanes = 32;
    private static readonly BufferStorage BlockLocalSmemTileStorage = new(BufferUsage.Temp, BufferScope.BlockLocal, PhysicalMemorySpace.SMem);
    private static readonly BufferStorage DeviceOutputAbiStorage = new(BufferUsage.Output, BufferScope.Device, PhysicalMemorySpace.GMem);
    private static readonly BufferStorage ThreadAddressableTileStorage = new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable);
    private static readonly BufferStorage ThreadRegisterTileStorage = new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.Register);

    private readonly CompileOptions _compileOptions;

    public DirectAffineTilingPass(string moduleKind, CompileOptions compileOptions)
    {
        ModuleKind = moduleKind;
        _compileOptions = compileOptions;
    }

    public string ModuleKind { get; }

    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is not Function func)
        {
            return Task.FromResult(input);
        }

        var analyzer = new DirectAffineTileAnalyzer(ModuleKind, _compileOptions);
        if (input.ModuleKind != ModuleKind && !analyzer.ContainsDirectAffine(func.Body.Body))
        {
            return Task.FromResult(input);
        }

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
        private readonly string _moduleKind;
        private readonly HashSet<BaseExpr> _visited = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<BaseExpr> _blockLocalEligible = new(ReferenceEqualityComparer.Instance);
        private readonly HashSet<BaseExpr> _registerEligible = new(ReferenceEqualityComparer.Instance);
        private readonly List<TileDecision> _decisions = new();
        private readonly List<Call> _decisionExprs = new();
        private readonly Dictionary<BaseExpr, int> _decisionIndexes = new(ReferenceEqualityComparer.Instance);
        private readonly CompileOptions _compileOptions;
        private Dictionary<BaseExpr, HashSet<Call>> _callUsers = new(ReferenceEqualityComparer.Instance);

        public DirectAffineTileAnalyzer(string moduleKind, CompileOptions compileOptions)
        {
            _moduleKind = moduleKind;
            _compileOptions = compileOptions;
        }

        public IReadOnlyList<TileDecision> Analyze(BaseExpr root)
        {
            _callUsers = BuildCallUsers(root);
            MarkBlockLocalEligible(root);
            MarkRegisterEligible(root, new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance));
            Visit(root);
            FinalizeDecisionSchedule(root);
            return _decisions;
        }

        public bool ContainsDirectAffine(BaseExpr root)
        {
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            return ContainsDirectAffine(root, visited);
        }

        private DistributionLayout? GetDistributionLayout(IRType type, IRArray<SBP> ndsbp, Placement placement, BufferStorage storage, TensorType tensorType, string opKind)
        {
            DistributionLayout? layout;
            IRArray<SBP> axisPolicies;
            Placement resolvedPlacement;
            if (type is DistributedType distributedType)
            {
                LayoutVerifier.Verify(distributedType.DistributionLayout, distributedType.StorageLayout);
                layout = distributedType.DistributionLayout;
                axisPolicies = distributedType.AxisPolicies;
                resolvedPlacement = distributedType.Placement;
            }
            else if (placement.Rank == 0)
            {
                return null;
            }
            else
            {
                var distributed = new DistributedType(tensorType, ndsbp, placement);
                LayoutVerifier.Verify(distributed.DistributionLayout, distributed.StorageLayout);
                layout = distributed.DistributionLayout;
                axisPolicies = distributed.AxisPolicies;
                resolvedPlacement = distributed.Placement;
            }

            return storage.PhysicalLocation is PhysicalMemorySpace.Register && string.Equals(_moduleKind, "cuda", StringComparison.Ordinal)
                ? GetCudaRegisterDistributionLayout(tensorType, layout, axisPolicies, resolvedPlacement, opKind)
                : layout;
        }

        private DistributionLayout GetCudaRegisterDistributionLayout(
            TensorType tensorType,
            DistributionLayout layout,
            IRArray<SBP> axisPolicies,
            Placement placement,
            string opKind)
        {
            if (layout.Kind != "SBP")
            {
                return layout;
            }

            if (tensorType.Shape is not { IsUnranked: false, Rank: 1 } || !tensorType.Shape[0].IsFixed)
            {
                throw new NotSupportedException($"{opKind} CUDA register tiling requires a fixed rank-1 distributed tile shape, got {tensorType.Shape}.");
            }

            if (axisPolicies.Count != 1 || axisPolicies[0] is not SBPSplit { Axes: var splitAxes } || splitAxes.Count != 1 || splitAxes[0] != 0)
            {
                throw new NotSupportedException($"{opKind} CUDA register tiling requires a rank-1 thread split policy S(0), got ({string.Join(',', axisPolicies)}).");
            }

            if (placement.Rank != 1 || placement.Name != "t")
            {
                throw new NotSupportedException($"{opKind} CUDA register tiling requires a single thread placement [t:N], got {placement}.");
            }

            var threadsPerCTA = placement.Hierarchy[0];
            if (threadsPerCTA <= 0 || threadsPerCTA % CudaWarpLanes != 0)
            {
                throw new NotSupportedException($"{opKind} CUDA register tiling requires thread count to be a positive multiple of {CudaWarpLanes}, got {threadsPerCTA} in placement {placement}.");
            }

            var blockExtent = tensorType.Shape[0].FixedValue;
            if (blockExtent % threadsPerCTA != 0)
            {
                throw new NotSupportedException($"{opKind} CUDA register tiling requires block extent {blockExtent} to be divisible by thread count {threadsPerCTA}; add a masked uneven TritonBlocked owner map before enabling this shape.");
            }

            var blocked = new TritonBlockedLayout(
                SizePerThread: checked((int)(blockExtent / threadsPerCTA)),
                ThreadsPerWarp: CudaWarpLanes,
                WarpsPerCTA: threadsPerCTA / CudaWarpLanes,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0],
                ThreadElementOrder: TritonThreadElementOrder.Strided);
            return DistributionLayout.TritonBlocked(tensorType.Shape, blocked);
        }

        private TensorType GetTensorType(IRType type, string opKind) => type switch
        {
            TensorType tensorType => tensorType,
            DistributedType distributedType => distributedType.TensorType,
            _ => throw new NotSupportedException($"{opKind} tiling requires tensor output, got {type}."),
        };

        private void ValidateOneDimensionalDirectAffine(AffineRelation relation, Shape shape, RankedShape symbols, string opKind)
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

        private void ValidateOneDimensionalShape(Shape shape, string opKind)
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
                    AttachGatherDecision(call, gather, GetStorage(call, "Affine.Gather"));
                    break;
                case Scatter scatter:
                    AttachScatterDecision(call, scatter, GetStorage(call, "Affine.Scatter"));
                    break;
                case IR.Math.Binary:
                    AttachElementwiseDecisionIfTilingInput(call, "Binary", GetStorage(call, "Binary"));
                    break;
                case IR.Math.Unary:
                    AttachElementwiseDecisionIfTilingInput(call, "Unary", GetStorage(call, "Unary"));
                    break;
                case IR.Tensors.Cast:
                    AttachElementwiseDecisionIfTilingInput(call, "Cast", GetStorage(call, "Cast"));
                    break;
                case IR.Tensors.Where:
                    AttachElementwiseDecisionIfTilingInput(call, "Where", GetStorage(call, "Where"));
                    break;
                case PrimFunctionWrapper:
                    AttachElementwiseDecisionIfTilingInput(call, "PrimFunctionWrapper", GetStorage(call, "PrimFunctionWrapper"));
                    break;
            }
        }

        private void AttachGatherDecision(Call call, Gather gather, BufferStorage storage)
        {
            var tensorType = GetTensorType(call.CheckedType, "Affine.Gather");
            ValidateOneDimensionalDirectAffine(gather.Relation, tensorType.Shape, gather.Symbols, "Affine.Gather");
            var distributionLayout = GetDistributionLayout(call.CheckedType, gather.NdSBP, gather.Placement, storage, tensorType, "Affine.Gather");
            AttachDecision(call, "Affine.Gather", tensorType, distributionLayout, storage, "direct affine gather tile");
        }

        private void AttachScatterDecision(Call call, Scatter scatter, BufferStorage storage)
        {
            var source = (Expr)call[Scatter.Source];
            var tensorType = GetTensorType(source.CheckedType, "Affine.Scatter source");
            ValidateOneDimensionalDirectAffine(scatter.Relation, tensorType.Shape, scatter.Symbols, "Affine.Scatter");
            var distributionLayout = GetDistributionLayout(source.CheckedType, new IRArray<SBP>(), new Placement([], string.Empty), storage, tensorType, "Affine.Scatter");
            AttachDecision(call, "Affine.Scatter", tensorType, distributionLayout, storage, "direct affine scatter tile terminator");
        }

        private void AttachElementwiseDecisionIfTilingInput(Call call, string opKind, BufferStorage storage)
        {
            if (!call.Arguments.ToArray().OfType<Expr>().Any(arg => TileDecisionMetadata.TryGet(arg, out _)))
            {
                return;
            }

            var tensorType = GetTensorType(call.CheckedType, opKind);
            ValidateOneDimensionalShape(tensorType.Shape, opKind);
            var distributionLayout = GetDistributionLayout(call.CheckedType, new IRArray<SBP>(), new Placement([], string.Empty), storage, tensorType, opKind);
            AttachDecision(call, opKind, tensorType, distributionLayout, storage, "producer consumes direct affine tile");
        }

        private void AttachDecision(Call call, string opKind, TensorType tensorType, DistributionLayout? distributionLayout, BufferStorage storage, string reason)
        {
            var ownerLocalShape = distributionLayout?.LocalShape ?? tensorType.Shape;
            var storageLayout = GetStorageLayout(tensorType.Shape, ownerLocalShape, distributionLayout, storage);
            var tileShape = storageLayout.LogicalShape;
            var byteSize = GetFixedByteSize(tensorType.DType, tileShape, opKind);
            if (distributionLayout is not null)
            {
                LayoutVerifier.Verify(distributionLayout, storageLayout);
            }

            var budgetBytes = GetBudgetBytes(storage);
            if (budgetBytes.HasValue && byteSize > budgetBytes.Value)
            {
                throw new InvalidOperationException($"{opKind} tile requires {byteSize} bytes for shape {tileShape}, exceeding {storage} budget {budgetBytes.Value} bytes.");
            }

            var decisionIndex = _decisions.Count;
            var reuseCount = _callUsers.TryGetValue(call, out var users) ? users.Count : 0;
            var lifetime = new TileLifetime(decisionIndex, decisionIndex);
            var telemetry = CreateTelemetry(storage, byteSize, reuseCount, lifetime, GetUnscheduledSlot(storage));
            var decision = new TileDecision(
                $"tile_{decisionIndex}",
                opKind,
                tileShape,
                distributionLayout,
                storageLayout,
                storage,
                lifetime,
                byteSize,
                new TileCapacity(byteSize, budgetBytes, GetBudgetSource(storage)),
                telemetry,
                storage.Scope is BufferScope.BlockLocal,
                reason);
            TileDecisionMetadata.Set(call, decision);
            _decisions.Add(decision);
            _decisionExprs.Add(call);
            _decisionIndexes.Add(call, decisionIndex);
            ExtendInputLifetimes(call, decisionIndex);
        }

        private StorageLayout GetStorageLayout(Shape tensorShape, Shape ownerLocalShape, DistributionLayout? distributionLayout, BufferStorage storage)
        {
            if (distributionLayout is not null &&
                storage is { Scope: BufferScope.BlockLocal, PhysicalLocation: PhysicalMemorySpace.SMem })
            {
                return StorageLayout.SharedBlock(tensorShape, distributionLayout);
            }

            return StorageLayout.Identity(ownerLocalShape);
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

        private BufferStorage GetStorage(Call call, string opKind)
        {
            if (_blockLocalEligible.Contains(call))
            {
                return BlockLocalSmemTileStorage;
            }

            if (_registerEligible.Contains(call))
            {
                return ThreadRegisterTileStorage;
            }

            if (IsCudaModule() && call.Target is Scatter)
            {
                return DeviceOutputAbiStorage;
            }

            if (IsCudaModule())
            {
                throw new NotSupportedException($"{opKind} direct affine CUDA tiling could not prove register or block-local SMem storage for {call.CheckedType}. Unsupported direct-affine DAGs must fail fast instead of falling back to thread-local addressable memory.");
            }

            return ThreadAddressableTileStorage;
        }

        private bool IsCudaModule() => string.Equals(_moduleKind, "cuda", StringComparison.Ordinal);

        private void MarkBlockLocalEligible(BaseExpr root)
        {
            if (!IsCudaModule())
            {
                return;
            }

            foreach ((var expr, var users) in _callUsers)
            {
                if (expr is Call { Target: Gather } && users.Count >= 2)
                {
                    _blockLocalEligible.Add(expr);
                }
            }
        }

        private Dictionary<BaseExpr, HashSet<Call>> BuildCallUsers(BaseExpr root)
        {
            var users = new Dictionary<BaseExpr, HashSet<Call>>(ReferenceEqualityComparer.Instance);
            CollectCallUsers(root, null, users, new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance));
            return users;
        }

        private void CollectCallUsers(BaseExpr expr, Call? parent, IDictionary<BaseExpr, HashSet<Call>> users, ISet<BaseExpr> path)
        {
            if (parent is not null)
            {
                if (!users.TryGetValue(expr, out var exprUsers))
                {
                    exprUsers = new HashSet<Call>(ReferenceEqualityComparer.Instance);
                    users.Add(expr, exprUsers);
                }

                exprUsers.Add(parent);
            }

            if (!path.Add(expr))
            {
                return;
            }

            var callParent = expr as Call;
            foreach (var operand in expr.Operands)
            {
                CollectCallUsers(operand, callParent, users, path);
            }

            path.Remove(expr);
        }

        private void MarkRegisterEligible(BaseExpr expr, ISet<BaseExpr> visited)
        {
            if (!visited.Add(expr))
            {
                return;
            }

            foreach (var operand in expr.Operands)
            {
                MarkRegisterEligible(operand, visited);
            }

            if (expr is not Call { Target: Scatter } scatterCall)
            {
                return;
            }

            if (TryGetRegisterProducer((Expr)scatterCall[Scatter.Source], out var producer, out var registerCalls))
            {
                if (registerCalls.Any(_blockLocalEligible.Contains))
                {
                    return;
                }

                _registerEligible.Add(scatterCall);
                _registerEligible.Add(producer);
                foreach (var registerCall in registerCalls)
                {
                    _registerEligible.Add(registerCall);
                }
            }
        }

        private bool TryGetRegisterProducer(Expr source, out Call producer, out IReadOnlyList<Call> registerCalls)
        {
            producer = null!;
            registerCalls = Array.Empty<Call>();
            if (source is not Call sourceCall)
            {
                return false;
            }

            var calls = new List<Call>();
            if (!TryCollectRegisterExpression(sourceCall, calls) ||
                !calls.Any(call => call.Target is Gather))
            {
                return false;
            }

            producer = sourceCall;
            registerCalls = calls;
            return true;
        }

        private bool TryCollectRegisterExpression(Expr expr, List<Call> registerCalls)
        {
            if (expr is TensorConst { CheckedType: TensorType { IsScalar: true } })
            {
                return true;
            }

            if (expr is not Call call)
            {
                return false;
            }

            var arguments = call.Arguments.ToArray().OfType<Expr>().ToArray();
            var supported = call.Target switch
            {
                Gather => true,
                IR.Math.Unary => arguments.Length == 1 && TryCollectAllRegisterExpressions(arguments, registerCalls),
                IR.Math.Binary => arguments.Length == 2 && TryCollectAllRegisterExpressions(arguments, registerCalls),
                IR.Tensors.Cast => arguments.Length == 1 && TryCollectAllRegisterExpressions(arguments, registerCalls),
                IR.Tensors.Where where => !where.IsTfWhere && arguments.Length == 3 && TryCollectAllRegisterExpressions(arguments, registerCalls),
                PrimFunctionWrapper => arguments.Length == 2 &&
                    arguments.All(argument => argument is Call { Target: Gather }) &&
                    TryCollectAllRegisterExpressions(arguments, registerCalls),
                _ => false,
            };

            if (supported)
            {
                registerCalls.Add(call);
            }

            return supported;
        }

        private bool TryCollectAllRegisterExpressions(IReadOnlyList<Expr> arguments, List<Call> registerCalls)
        {
            foreach (var argument in arguments)
            {
                if (!TryCollectRegisterExpression(argument, registerCalls))
                {
                    return false;
                }
            }

            return true;
        }

        private TileTelemetry CreateTelemetry(BufferStorage storage, long byteSize, int reuseCount, TileLifetime lifetime, string allocationSlot)
        {
            var registerBytes = storage.PhysicalLocation is PhysicalMemorySpace.Register ? byteSize : 0;
            var smemBytes = storage.PhysicalLocation is PhysicalMemorySpace.SMem ? byteSize : 0;
            var estimatedTraffic = storage.PhysicalLocation switch
            {
                PhysicalMemorySpace.Register => byteSize * Math.Max(1, reuseCount),
                PhysicalMemorySpace.SMem => byteSize + (byteSize * Math.Max(0, reuseCount - 1)),
                _ => byteSize,
            };
            return new TileTelemetry(reuseCount, estimatedTraffic, registerBytes, smemBytes, allocationSlot);
        }

        private string GetUnscheduledSlot(BufferStorage storage) => storage.PhysicalLocation switch
        {
            PhysicalMemorySpace.Register => "register-live[unscheduled]",
            PhysicalMemorySpace.SMem => "smem-slot[unscheduled]",
            PhysicalMemorySpace.GMem when storage.Usage is BufferUsage.Output => "output-abi",
            _ => "addressable",
        };

        private long? GetBudgetBytes(BufferStorage storage) => storage.PhysicalLocation switch
        {
            PhysicalMemorySpace.Register => RequireNttTargetOptions("register tile capacity").RegisterTileBudgetBytes,
            PhysicalMemorySpace.SMem => RequireNttTargetOptions("shared-memory tile capacity").SharedMemoryTileBudgetBytes,
            _ => null,
        };

        private string GetBudgetSource(BufferStorage storage) => storage.PhysicalLocation switch
        {
            PhysicalMemorySpace.Register => "target-options:RegisterTileBudgetBytes",
            PhysicalMemorySpace.SMem => "target-options:SharedMemoryTileBudgetBytes",
            _ => "not-capacity-limited",
        };

        private INTTTargetOptions RequireNttTargetOptions(string context)
        {
            if (_compileOptions.TargetOptions is INTTTargetOptions targetOptions)
            {
                return targetOptions;
            }

            throw new InvalidOperationException($"Direct affine {context} requires CompileOptions.TargetOptions to implement INTTTargetOptions; missing target options would hide capacity failures.");
        }

        private long GetFixedByteSize(DataType dtype, Shape tileShape, string opKind)
        {
            ValidateOneDimensionalShape(tileShape, opKind);
            checked
            {
                return tileShape[0].FixedValue * dtype.SizeInBytes;
            }
        }

        private void ExtendInputLifetimes(Call consumer, int consumerIndex)
        {
            foreach (var argument in consumer.Arguments.ToArray().OfType<Expr>())
            {
                if (!_decisionIndexes.TryGetValue(argument, out var producerIndex))
                {
                    continue;
                }

                var producer = _decisions[producerIndex];
                var lifetime = producer.Lifetime with { End = Math.Max(producer.Lifetime.End, consumerIndex) };
                var extended = producer with
                {
                    Lifetime = lifetime,
                    Telemetry = CreateTelemetry(producer.Storage, producer.ByteSize, producer.Telemetry.ReuseCount, lifetime, GetUnscheduledSlot(producer.Storage)),
                };
                ReplaceDecision(producerIndex, extended);
            }
        }

        private void FinalizeDecisionSchedule(BaseExpr root)
        {
            if (_decisions.Count == 0)
            {
                return;
            }

            var executionOrder = BuildDecisionExecutionOrder(root);
            for (int index = 0; index < _decisions.Count; index++)
            {
                var expr = _decisionExprs[index];
                var start = executionOrder.TryGetValue(expr, out var orderedStart) ? orderedStart : index;
                var end = start;
                if (_callUsers.TryGetValue(expr, out var users))
                {
                    foreach (var user in users)
                    {
                        if (executionOrder.TryGetValue(user, out var userOrder))
                        {
                            end = Math.Max(end, userOrder);
                        }
                        else if (_decisionIndexes.TryGetValue(user, out var userIndex))
                        {
                            end = Math.Max(end, userIndex);
                        }
                    }
                }

                var decision = _decisions[index];
                ReplaceDecision(index, decision with
                {
                    Lifetime = new TileLifetime(start, end),
                    Telemetry = CreateTelemetry(decision.Storage, decision.ByteSize, decision.Telemetry.ReuseCount, new TileLifetime(start, end), GetUnscheduledSlot(decision.Storage)),
                });
            }

            CheckAggregateCapacity();
            ScheduleSharedMemorySlots();
            FinalizeUnslottedTelemetry();
        }

        private Dictionary<BaseExpr, int> BuildDecisionExecutionOrder(BaseExpr root)
        {
            var order = new Dictionary<BaseExpr, int>(ReferenceEqualityComparer.Instance);
            var visited = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
            var next = 0;
            VisitForOrder(root, order, visited, ref next);
            return order;
        }

        private void VisitForOrder(BaseExpr expr, IDictionary<BaseExpr, int> order, ISet<BaseExpr> visited, ref int next)
        {
            if (!visited.Add(expr))
            {
                return;
            }

            switch (expr)
            {
                case IRBlock block:
                    VisitForOrder(block.Body, order, visited, ref next);
                    return;
                case Sequential sequential:
                    foreach (var field in sequential.Fields.ToArray())
                    {
                        VisitForOrder(field, order, visited, ref next);
                    }

                    return;
                case IR.Tuple tuple:
                    foreach (var field in tuple.Fields.ToArray().OfType<Expr>())
                    {
                        VisitForOrder(field, order, visited, ref next);
                    }

                    return;
            }

            foreach (var operand in expr.Operands)
            {
                VisitForOrder(operand, order, visited, ref next);
            }

            if (expr is Call call && _decisionIndexes.ContainsKey(call))
            {
                order[call] = next++;
            }
        }

        private void CheckAggregateCapacity()
        {
            var capacityGroups = _decisions
                .Where(decision => decision.Capacity.BudgetBytes.HasValue)
                .GroupBy(decision => new StoragePressureKey(decision.Storage.Scope, decision.Storage.PhysicalLocation));
            foreach (var group in capacityGroups)
            {
                var decisions = group.ToArray();
                var budget = decisions[0].Capacity.BudgetBytes!.Value;
                var minPoint = decisions.Min(decision => decision.Lifetime.Start);
                var maxPoint = decisions.Max(decision => decision.Lifetime.End);
                for (int point = minPoint; point <= maxPoint; point++)
                {
                    var liveBytes = decisions
                        .Where(decision => decision.Lifetime.Start <= point && point <= decision.Lifetime.End)
                        .Sum(decision => decision.ByteSize);
                    if (liveBytes > budget)
                    {
                        throw new InvalidOperationException($"Direct affine aggregate live {group.Key.Scope}/{group.Key.Location} pressure requires {liveBytes} bytes at schedule point {point}, exceeding budget {budget} bytes. Live tiles: {string.Join(", ", decisions.Where(decision => decision.Lifetime.Start <= point && point <= decision.Lifetime.End).Select(decision => $"{decision.Id}:{decision.OpKind}:{decision.TileShape}:{decision.ByteSize}B:{decision.Lifetime}"))}.");
                    }
                }
            }
        }

        private void ScheduleSharedMemorySlots()
        {
            var smemIndexes = _decisions
                .Select((decision, index) => (Decision: decision, Index: index))
                .Where(item => item.Decision.Storage is { Scope: BufferScope.BlockLocal, PhysicalLocation: PhysicalMemorySpace.SMem })
                .OrderBy(item => item.Decision.Lifetime.Start)
                .ThenBy(item => item.Decision.Id, StringComparer.Ordinal)
                .ToArray();
            if (smemIndexes.Length == 0)
            {
                return;
            }

            var budget = smemIndexes[0].Decision.Capacity.BudgetBytes
                ?? throw new InvalidOperationException("Block-local SMem tile scheduling requires a shared-memory budget.");
            var slots = new List<ScheduledTileSlot>();
            var placements = new Dictionary<int, ScheduledTileSlot>();
            foreach (var item in smemIndexes)
            {
                var decision = item.Decision;
                var slot = slots.FirstOrDefault(candidate => candidate.LastEnd < decision.Lifetime.Start);
                if (slot is null)
                {
                    var offset = slots.Count == 0 ? 0 : slots.Max(candidate => candidate.Offset + candidate.Size);
                    slot = new ScheduledTileSlot(slots.Count, offset, decision.ByteSize, decision.Lifetime.End);
                    slots.Add(slot);
                }
                else
                {
                    slot.Size = Math.Max(slot.Size, decision.ByteSize);
                    slot.LastEnd = decision.Lifetime.End;
                }

                var poolBytes = slots.Max(candidate => candidate.Offset + candidate.Size);
                if (poolBytes > budget)
                {
                    throw new InvalidOperationException($"Direct affine SMem slot schedule requires {poolBytes} bytes after placing {decision.Id}, exceeding shared-memory budget {budget} bytes.");
                }

                placements[item.Index] = slot;
            }

            foreach (var placement in placements)
            {
                var decision = _decisions[placement.Key];
                var slot = placement.Value;
                var slotText = $"smem-slot{slot.Id}@{slot.Offset}+{slot.Size}[{decision.Lifetime.Start},{decision.Lifetime.End}]";
                ReplaceDecision(placement.Key, decision with
                {
                    Telemetry = CreateTelemetry(decision.Storage, decision.ByteSize, decision.Telemetry.ReuseCount, decision.Lifetime, slotText),
                });
            }
        }

        private void FinalizeUnslottedTelemetry()
        {
            for (int index = 0; index < _decisions.Count; index++)
            {
                var decision = _decisions[index];
                if (decision.Storage.PhysicalLocation is PhysicalMemorySpace.SMem)
                {
                    continue;
                }

                var slot = decision.Storage.PhysicalLocation switch
                {
                    PhysicalMemorySpace.Register => $"register-live[{decision.Lifetime.Start},{decision.Lifetime.End}]",
                    PhysicalMemorySpace.GMem when decision.Storage.Usage is BufferUsage.Output => "output-abi",
                    _ => "addressable",
                };
                ReplaceDecision(index, decision with
                {
                    Telemetry = CreateTelemetry(decision.Storage, decision.ByteSize, decision.Telemetry.ReuseCount, decision.Lifetime, slot),
                });
            }
        }

        private void ReplaceDecision(int index, TileDecision decision)
        {
            _decisions[index] = decision;
            TileDecisionMetadata.Set(_decisionExprs[index], decision);
        }

        private sealed record StoragePressureKey(BufferScope Scope, PhysicalMemorySpace Location);

        private sealed class ScheduledTileSlot
        {
            public ScheduledTileSlot(int id, long offset, long size, int lastEnd)
            {
                Id = id;
                Offset = offset;
                Size = size;
                LastEnd = lastEnd;
            }

            public int Id { get; }

            public long Offset { get; }

            public long Size { get; set; }

            public int LastEnd { get; set; }
        }
    }
}
