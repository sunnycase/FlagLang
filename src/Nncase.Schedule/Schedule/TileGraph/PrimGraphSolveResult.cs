// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Reactive;
using System.Reflection;
using Google.OrTools.Sat;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Logics;
using Nncase.TIR;
using Nncase.TIR.Builders;
using Nncase.Utilities;
using static Nncase.TIR.TIRExtensions;
using Isl = IntegerSetLibrary;

namespace Nncase.Schedule.TileGraph;

/// <summary>
/// because a tile node may have multiple buffer, and each buffer may store at different level.
/// so we need a class to represent the buffer identity.
/// </summary>
public record class NodeWithBuffer(TileNode Node, BufferIdentity Id)
{
    public long MaxSize => TensorUtilities.GetProduct(Id.Node.BufferShapes[Id.Index].ToArray()) * Id.Node.GetBufferElemSize(Id.Index);
}

public record NodeWithBufferInfo(long Size, Tuple<int, int> Liveness, long[] Shape, long[] Strides)
{
    public ulong Offset { get; set; } = ulong.MaxValue;
}

/// <summary>
/// Represents the view information of a buffer.
/// </summary>
/// <param name="Parent">The parent view information.</param>
/// <param name="View">The view expression.</param>
/// <param name="ViewVar">The variable for the view expression.</param>
/// <param name="Buffer">The buffer expression.</param>
/// <param name="GlobalOffsets">The global offsets for the buffer.</param>
/// <param name="LocalOffsets">The local offsets of parent for the buffer.</param>
/// <param name="Shape">The shape of the view.</param>
internal sealed record ViewInfo(ViewInfo? Parent, Expr View, Var ViewVar, Expr Buffer, RankedShape GlobalOffsets, RankedShape LocalOffsets, RankedShape Shape)
{
}

public sealed class TreeSolveResult : TreeSolverBase<long>, ITreeNodeVisitor<TreeSolveResult.Context, Unit>
{
    private readonly Dictionary<ITileable, Dictionary<BufferIdentity, ViewInfo>> _viewInfoMemo;

    public TreeSolveResult(BufferGraph primBufferGraph, long objectiveValue, Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> levelNodeBufferInfos, Dictionary<OpNode, OpNodeInfo<long>> primitiveBufferInfo, Dictionary<TileNode, TileNodeInfo<long>> levelBufferInfos, Dictionary<ITileable, DomainInfo<long>> domainInfos, INTTTargetOptions targetOptions, string moduleKind)
        : base(null!, primitiveBufferInfo, levelBufferInfos, domainInfos, targetOptions)
    {
        PrimBufferGraph = primBufferGraph;
        (Inputs, Outputs) = primBufferGraph.GetInputsOutputs(primBufferGraph.Parent as BufferGraph);
        InputOutputVars = new();
        foreach (var bid in Inputs.Concat(Outputs))
        {
            var expr = bid.Node.Grid.Buffers[bid.Index];
            var tensorType = GetBufferTensorType(expr);
            if (!InputOutputVars.TryGetValue(bid, out _))
            {
                var ivar = new Var($"{bid}", tensorType);
                InputOutputVars.Add(bid, ivar);
            }
        }

        ObjectiveValue = objectiveValue;
        LevelNodeBufferInfos = levelNodeBufferInfos;
        ModuleKind = moduleKind;
        _viewInfoMemo = new();
    }

    public BufferGraph PrimBufferGraph { get; }

    public HashSet<BufferIdentity> Inputs { get; }

    public HashSet<BufferIdentity> Outputs { get; }

    public Dictionary<BufferIdentity, Var> InputOutputVars { get; }

    public long ObjectiveValue { get; }

    public Dictionary<int, Dictionary<NodeWithBuffer, NodeWithBufferInfo>> LevelNodeBufferInfos { get; }

    public string ModuleKind { get; }

    public RankedShape PartialShapeFromDomain(Isl.set parentDomain, DomainRelation domainRel, Isl.set tiledDomain, AffineMap access, uint dim, Dictionary<string, Dimension> paramDimMap)
    {
        var domainRank = tiledDomain.dim(Isl.dim_type.set);
        var shapeRank = access.Results.Length;

        var (domainRelMinMpa, domainRelMaxMpa) = TilingUtilities.ToMinMaxMpa(domainRel.Map);
        var parentMaxMpa = parentDomain.max_multi_pw_aff();
        var parentMinMpa = parentDomain.min_multi_pw_aff();
        var currentMaxMpa = domainRelMaxMpa.pullback(parentMaxMpa);
        var currentMinMpa = domainRelMinMpa.pullback(parentMinMpa);
        var tiledMaxMpa = tiledDomain.max_multi_pw_aff();
        var tiledMinMpa = tiledDomain.min_multi_pw_aff();
        var (accessMinMpa, accessMaxMpa) = TilingUtilities.ToMinMaxMpa(access);

        for (int i = (int)dim; i < domainRank; i++)
        {
            tiledMaxMpa = tiledMaxMpa.set_at(i, currentMaxMpa.at(i));
            tiledMinMpa = tiledMinMpa.set_at(i, currentMinMpa.at(i));
        }

        var bufferMaxMpa = accessMaxMpa.pullback(tiledMaxMpa.add_constant(1));
        var bufferMinMpa = accessMinMpa.pullback(tiledMinMpa);
        var bufferShapeMpa = bufferMaxMpa.sub(bufferMinMpa);
        var dimensions = new Dimension[shapeRank];
        for (int i = 0; i < bufferShapeMpa.size(); i++)
        {
            var accessMax = accessMaxMpa.at(i);
            if (accessMax.is_cst() && accessMax.max_val().num_si() == 0)
            {
                dimensions[i] = 1;
            }
            else
            {
                var pa = bufferShapeMpa.at(i);
                dimensions[i] = ISLUtility.ToDimension(pa, paramDimMap);
            }
        }

        return new RankedShape(dimensions);
    }

    public Unit Visit(TileNode value, Context context)
    {
        var (parentbuilder, parentOffsets, parentExtents) = context;
        {
            var newParentExtents = new Dimension[parentExtents.Length];
            for (int i = 0; i < parentExtents.Length; i++)
            {
                if (parentExtents[i] is AsDim { Dim: Call { Target: IR.Tensors.LocalShardDim } } localShardDim)
                {
                    var letDim = T.LetDim(out var dimVar, localShardDim, $"L{value.Level}_d{i}");
                    parentbuilder.Body(letDim);
                    parentbuilder = letDim;
                    dimVar.Metadata = new() { Range = localShardDim.Metadata.Range };
                    newParentExtents[i] = dimVar;
                }
                else
                {
                    newParentExtents[i] = parentExtents[i];
                }
            }

            parentExtents = newParentExtents;
        }

        // get current tile node's domain.
        // todo use domain map to introduce the new dimensions.
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var loopBuilders = new ISequentialBuilder<TIR.For>[value.DomainRelation.Map.Results.Length];
        var loopVars = new DimVar[value.DomainRelation.Map.Results.Length];

        var nodeMemo = TileNodeMemo[value];
        var domainRank = value.DomainRelation.Map.Results.Length;

        // create tile map from tile vars
        Isl.map tilemap;
        {
            var dims = new List<string>();
            var outerDims = new List<string>();
            var innerDims = new List<string>();
            var constraints = new List<string>();
            for (int i = 0; i < value.DomainRelation.Map.Results.Length; i++)
            {
                var tilesize = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
                dims.Add($"d{i}");
                outerDims.Add($"d{i}_out");
                innerDims.Add($"d{i}_in");
                constraints.Add($"d{i}_out = {tilesize} * (d{i} // {tilesize}) and d{i}_in = d{i} - d{i}_out");
            }

            tilemap = new Isl.map(Isl.ctx.Current, $"{{ [{string.Join(',', dims)}] -> [{string.Join(',', outerDims)},{string.Join(',', innerDims)}] : {string.Join(" and ", constraints)} }}");
        }

        var currentDomain = parentDomain.apply(value.DomainRelation.ToMap());
        var currentRanges = value.DomainRelation.Map.Apply(parentOffsets, parentExtents);
        var currentOffsets = currentRanges.Select(r => r.Start).ToArray();
        var currentExtents = currentRanges.Select(r => r.Stop).ToArray();
        var tiledParentDomain = tilemap.intersect_domain(currentDomain).range();
        var tiledChildDomain = tiledParentDomain.move_dims(Isl.dim_type.param, 0, Isl.dim_type.set, 0, (uint)domainRank);
        var childBoundsMpa = tiledChildDomain.max_multi_pw_aff().add_constant(1).sub(tiledChildDomain.min_multi_pw_aff());

        // from inner to outer
        var forwardExtents = new Dimension[domainRank];
        for (int i = value.DomainRelation.Map.Results.Length - 1; i >= 0; i--)
        {
            Dimension start = 0L;
            Dimension stop = currentExtents[i];
            Dimension stride = nodeMemo.BackWardExtents[0][i] / TileableNodeMemo[value].TileVars[i];
            loopBuilders[i] = T.Serial(out var loopVar, (0L, stop, stride), $"d{i}_Op{value.OpId}_L{value.Level}");
            loopVar.Metadata.Range = new(0, nodeMemo.BackWardExtents[0][i]);
            loopVars[i] = loopVar;
            paramDimMap.Add($"d{i}_out", loopVar);
            {
                Dimension forwardExtent;
                var boundPa = childBoundsMpa.at(i);
                if (boundPa.is_cst())
                {
                    forwardExtent = boundPa.max_val().num_si();
                }
                else
                {
                    var build = new Isl.ast_build(Isl.ctx.Current);
                    var astExpr = build.expr_from(boundPa);
                    forwardExtent = ISLUtility.ToDimension(astExpr, paramDimMap);
                    forwardExtent.Metadata = new()
                    {
                        Range = new(boundPa.min_val().num_si(), boundPa.max_val().num_si()),
                    };
                }

                forwardExtents[i] = forwardExtent;
            }
        }

        // forwardOffsets[0] means partentOffsets, forwardOffsets[i] means partentOffsets[0:i] + loop vars[0:i]
        var forwardOffsets = new Dimension[loopVars.Length + 1][];
        for (int i = 0; i < loopVars.Length + 1; i++)
        {
            var offsets = forwardOffsets[i] = currentOffsets.ToArray();

            for (int j = 0; j < i; j++)
            {
                offsets[j] += loopVars[j];
            }
        }

        // var domainLetBuilders = Enumerable.Range(0, value.DimNames.Length).Select(i => new List<ISequentialBuilder<Expr>>()).ToArray();
        var cntBuilder = parentbuilder;
        var childBuilders = new List<ISequentialBuilder<Expr>>();
        for (int i = 0; i < value.Children.Length; i++)
        {
            var childBuilder = T.Sequential();
            childBuilders.Add(childBuilder);
        }

        for (int ci = 0; ci < loopVars.Length + 1; ci++)
        {
            foreach (var (bid, bufferInfo) in nodeMemo.BufferInfoMap)
            {
                var place = bufferInfo.Places[ci];
                var expr = bid.Node.Grid.Buffers[bid.Index];
                var distributedType = GetBufferDistributedType(expr);
                for (int sl = 0; sl < place.Length; sl++)
                {
                    if (place[sl] != 1)
                    {
                        continue;
                    }

                    var kernelInfo = bid.Node.GetKernelInfo(TargetOptions);
                    var partialShape = PartialShapeFromDomain(parentDomain, value.DomainRelation, tiledChildDomain, bufferInfo.Map, (uint)ci, paramDimMap);

                    var viewInfo = GetViewInfo(sl, value, bid, bufferInfo.Map, forwardOffsets[ci], partialShape);
                    var letBuilder = T.Let(viewInfo.ViewVar, viewInfo.View);
                    cntBuilder.Body(letBuilder);
                    cntBuilder = letBuilder;

                    // note when create loop is inner loop, the buffer load store should be instert by children's order.
                    {
                        var localBuilder = ci < loopVars.Length ? cntBuilder : childBuilders[FetchBidOwnerIndex(value, bid)];
                        if (!TargetOptions.UnifiedMemoryArch && viewInfo.Parent is ViewInfo parentViewInfo)
                        {
                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Read))
                            {
                                localBuilder.Body(T.Memcopy(viewInfo.ViewVar, IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar ?? parentViewInfo.Buffer, viewInfo.LocalOffsets, viewInfo.Shape)));
                            }

                            if (kernelInfo.BufferInfos[bid.Index].State.HasFlag(MicroKernelBufferInfo.BufferState.Write))
                            {
                                localBuilder.Tail(T.Memcopy(IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar ?? parentViewInfo.Buffer, viewInfo.LocalOffsets, viewInfo.Shape), viewInfo.ViewVar));
                            }
                        }
                    }

                    if (!_viewInfoMemo.TryGetValue(value, out var subViewMap))
                    {
                        subViewMap = new();
                        _viewInfoMemo.Add(value, subViewMap);
                    }

                    subViewMap[bid] = viewInfo;
                }
            }

            if (ci < loopVars.Length)
            {
                cntBuilder.Body(loopBuilders[ci]);
                cntBuilder = loopBuilders[ci];
            }
            else
            {
            }
        }

        for (int i = 0; i < value.Children.Length; i++)
        {
            var childBuilder = childBuilders[i];
            value.Children[i].Accept(this, new(childBuilder, forwardOffsets[^1], forwardExtents));
            cntBuilder.Body(childBuilder);
        }

        return default;
    }

    public Unit Visit(OpNode value, Context context)
    {
        var (parentbuilder, parentOffsets, parentExtents) = context;
        var parentDomain = ISLUtility.ToParametricDomain(parentExtents, out var paramVarMap);
        var paramDimMap = paramVarMap.Select(p => (p.Key.Name, p.Value)).Concat(parentExtents.Select((d, i) => ($"d{i}", d))).ToDictionary();

        var currentDomain = parentDomain.apply(value.DomainRelation.ToMap());
        var currentRanges = value.DomainRelation.Map.Apply(parentOffsets, parentExtents);
        var currentOffsets = currentRanges.Select(r => r.Start).ToArray();
        var currentExtents = currentRanges.Select(r => r.Stop).ToArray();
        var kernelInfo = value.GetKernelInfo(TargetOptions);

        var bufferParentViewInfos = new ViewInfo[value.BufferShapes.Length];
        var bufferViews = new Expr[value.BufferShapes.Length];
        for (int i = 0; i < value.BufferShapes.Length; i++)
        {
            var bid = new BufferIdentity(value.Wrapped, i);
            var shape = PartialShapeFromDomain(parentDomain, value.DomainRelation, currentDomain, value.Grid.AccessMaps[i], (uint)currentDomain.dim(Isl.dim_type.set), paramDimMap);
            if (!TryGetParentViewInfo(value, bid, out var parentViewInfo))
            {
                throw new InvalidOperationException($"can't find parent view info for {bid} at OpNode {value}!");
            }

            var bufferOffsets = OpNodeMemo[value].Maps[i].Apply(currentOffsets, Enumerable.Repeat<Dimension>(0L, currentOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
            var offsets = new Dimension[bufferOffsets.Length];
            for (int j = 0; j < offsets.Length; j++)
            {
                var x = bufferOffsets[j] - parentViewInfo.GlobalOffsets[j];
                offsets[j] = x;
            }

            offsets = ISLUtility.RoundTrip(offsets);
            bufferParentViewInfos[i] = parentViewInfo;
            bufferViews[i] = IR.F.Buffer.BufferSubview(parentViewInfo.ViewVar, offsets, shape);
        }

        var bodyVarReplaces = new Dictionary<BaseExpr, BaseExpr>();
        for (int i = 0; i < value.Grid.BodyParameters.Length; i++)
        {
            bodyVarReplaces.Add(value.Grid.BodyParameters[i], bufferViews[i]);
        }

        var domain = new IR.Tuple(currentOffsets.Select(off => new IR.Tuple(IR.F.Shapes.AsTensor(off), (Expr)0L)).ToArray());
        bodyVarReplaces.Add(value.Grid.DomainParameter, domain);
        var nestBody = new ReplacingExprCloner(bodyVarReplaces).Clone(value.Grid.Body, default);
        nestBody = new AffineIOTileOffsetRebaser(currentOffsets).Clone(nestBody, default);
        parentbuilder.Body(nestBody);
        return default;
    }

    public long ScheduleBuffers()
    {
        var maxAlign = 0L;
        foreach (var (level, nodeBufferInfos) in LevelNodeBufferInfos)
        {
            var memoryLevel = TargetOptions.GetRequired(level, "Auto tiling buffer scheduling");
            if (!memoryLevel.IsAddressable)
            {
                foreach (var (key, info) in nodeBufferInfos)
                {
                    maxAlign = Math.Max(maxAlign, key.Id.Node.GetBufferElemSize(key.Id.Index));
                    info.Offset = 0;
                }

                continue;
            }

            var model = new CpModel();
            var rectangles = new Dictionary<NodeWithBuffer, (IntervalVar XInterval, IntervalVar YInterval)>();
            int count = 0;
            var cons = model.AddNoOverlap2D();
            foreach (var (key, info) in nodeBufferInfos)
            {
                if (info.Size > 0)
                {
                    var x = model.NewFixedSizeIntervalVar(info.Liveness.Item1, info.Liveness.Item2 - info.Liveness.Item1, $"x{count}");
                    var ystart = model.NewIntVar(0, TargetOptions.MemoryCapacities[level] - info.Size, $"ystart{count}");
                    var align = key.Id.Node.GetBufferElemSize(key.Id.Index);
                    if (ModuleKind == "xpu")
                    {
                        align = 128;
                    }

                    maxAlign = Math.Max(maxAlign, align);
                    model.AddModuloEquality(0, ystart, align);
                    var y = model.NewFixedSizeIntervalVar(ystart, info.Size, $"y{count}");
                    cons.AddRectangle(x, y);
                    rectangles.Add(key, (x, y));
                    count++;
                }
            }

            var solver = new CpSolver();
            var status = solver.Solve(model);
            if (status is not CpSolverStatus.Optimal)
            {
                throw new InvalidOperationException("can't schedule buffers!");
            }

            foreach (var (k, (_, y)) in rectangles)
            {
                nodeBufferInfos[k].Offset = (ulong)solver.Value(y.StartExpr());
            }
        }

        return maxAlign;
    }

    private TensorType GetBufferTensorType(Expr expr)
    {
        TensorType GetTensorType(IRType type) => type switch
        {
            TensorType t => t,
            DistributedType dt => Utilities.DistributedUtility.GetDividedTensorType(dt),
            _ => throw new NotSupportedException(),
        };

        return expr switch
        {
            IR.Buffers.BufferOf bufof => GetTensorType(bufof.Input.CheckedType),
            Call { Target: IR.Buffers.Uninitialized } c => GetTensorType(c.CheckedType),
            _ => throw new NotSupportedException(),
        };
    }

    private DistributedType? GetBufferDistributedType(Expr expr)
    {
        DistributedType? GetTensorType(IRType type) => type switch
        {
            TensorType => null!,
            DistributedType dt => dt,
            _ => throw new NotSupportedException(),
        };

        return expr switch
        {
            IR.Buffers.BufferOf bufof => GetTensorType(bufof.Input.CheckedType),
            Call { Target: IR.Buffers.Uninitialized } c => GetTensorType(c.CheckedType),
            _ => throw new NotSupportedException(),
        };
    }

    private bool TryGetParentViewInfo(ITreeNode node, BufferIdentity bid, [MaybeNullWhen(false)] out ViewInfo parentViewInfo)
    {
        var cbid = bid;
        var parentNode = node.Parent;
        parentViewInfo = null;
        while (parentNode is TileNode parentTileNode && parentTileNode.OpId != -1)
        {
            var pbid = TileNodeMemo[parentTileNode].GetByChildBuffer(cbid);
            if (_viewInfoMemo.TryGetValue(parentTileNode, out var viewMap) && viewMap.TryGetValue(pbid, out var viewInfo))
            {
                parentViewInfo = viewInfo;
                return true;
            }

            parentNode = parentTileNode.Parent;
            cbid = pbid;
        }

        return false;
    }

    private ViewInfo GetViewInfo(int storeLevel, TileNode node, BufferIdentity bid, AffineMap map, Dimension[] forwardOffsets, RankedShape shape)
    {
        TIR.Buffer AllocateBuffer(TileNode tileNode, BufferIdentity bid)
        {
            var expr = bid.Node.Grid.Buffers[bid.Index];
            var distributedType = GetBufferDistributedType(expr);
            var tensorType = distributedType?.TensorType ?? GetBufferTensorType(expr);
            var info = LevelNodeBufferInfos[storeLevel][new NodeWithBuffer(tileNode, bid)];
            var alignment = tensorType.DType.SizeInBytes;
            var strides = info.Strides.Select(i => (Dimension)i).ToArray(); // using fixed strides.
            var memoryLevel = TargetOptions.GetRequired(storeLevel, "Auto tiling buffer allocation");
            var storage = memoryLevel.ToTileStorage(BufferUsage.Temp, storeLevel);
            var physicalBuffer = memoryLevel.IsAddressable
                ? new PhysicalBuffer(alignment, Tensor.FromPointer(info.Offset, tensorType.DType), info.Size, storage)
                : new PhysicalBuffer(alignment, info.Size, storage);
            var dimensions = distributedType is null
                ? shape.Dimensions.ToArray()
                : ((RankedShape)distributedType.TensorType.Shape).Dimensions.ToArray();
            IRType bufferType = distributedType is null ? tensorType : distributedType;
            return new TIR.Buffer($"{bid}", bufferType, new MemSpan(physicalBuffer), dimensions, strides);
        }

        Expr GetViewExpr(ViewInfo? parentInfo, Expr buffer, RankedShape forwardOffsets, RankedShape relatedOffsets, RankedShape shape)
        {
            return parentInfo switch
            {
                null => buffer switch
                {
                    TIR.Buffer buf => IR.F.Buffer.AllocateBufferView(buf),
                    Var ivar => TargetOptions.UnifiedMemoryArch switch
                    {
                        true => IR.F.Buffer.BufferSubview(ivar, relatedOffsets, shape),
                        false => ivar,
                    },
                    _ => throw new NotSupportedException(),
                },
                ViewInfo info => TargetOptions.UnifiedMemoryArch switch
                {
                    true => IR.F.Buffer.BufferSubview(info.ViewVar, relatedOffsets, shape),
                    false => buffer switch
                    {
                        TIR.Buffer buf => IR.F.Buffer.AllocateBufferView(buf),
                        _ => throw new NotSupportedException(),
                    },
                },
            };
        }

        var bufferOffsets = map.Apply(forwardOffsets, Enumerable.Repeat<Dimension>(0L, forwardOffsets.Length).ToArray()).Select(i => i.Start).ToArray();
        var offsets = new RankedShape(bufferOffsets);
        if (TryGetParentViewInfo(node, bid, out var parentViewInfo))
        {
            var viewOffset = new Dimension[bufferOffsets.Length];
            for (int j = 0; j < viewOffset.Length; j++)
            {
                var x = bufferOffsets[j] - parentViewInfo.GlobalOffsets[j];
                viewOffset[j] = x;
            }

            offsets = ISLUtility.RoundTrip(viewOffset);
            Expr buffer;
            if (TargetOptions.UnifiedMemoryArch)
            {
                buffer = parentViewInfo.Buffer;
            }
            else
            {
                buffer = AllocateBuffer(node, bid);
            }

            var view = GetViewExpr(parentViewInfo, buffer, bufferOffsets, offsets, shape);
            var viewVar = new Var($"{bid}_L{node.Level}", AnyType.Default);
            return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, offsets, shape);
        }
        else
        {
            parentViewInfo = null;
            Expr buffer = null!;
            var fromExternal = Inputs.Contains(bid) || Outputs.Contains(bid);

            if (TargetOptions.UnifiedMemoryArch && fromExternal)
            {
                buffer = InputOutputVars[bid];
            }
            else
            {
                buffer = AllocateBuffer(node, bid);
            }

            if (!TargetOptions.UnifiedMemoryArch && fromExternal)
            {
                parentViewInfo = new ViewInfo(null, null!, null!, InputOutputVars[bid], new RankedShape(bufferOffsets.Select(i => 0).ToArray()), new RankedShape(bufferOffsets.Select(i => 0).ToArray()), shape);
            }

            var view = GetViewExpr(null, buffer, bufferOffsets, offsets, shape);
            var viewVar = new Var($"{bid}_L{node.Level}", AnyType.Default);
            return new ViewInfo(parentViewInfo, view, viewVar, buffer, bufferOffsets, fromExternal ? bufferOffsets : new RankedShape(bufferOffsets.Select(i => 0).ToArray()), shape);
        }
    }

    private int FetchBidOwnerIndex(TileNode node, BufferIdentity bid)
    {
        for (int i = 0; i < node.Children.Length; i++)
        {
            var child = node.Children[i];
            if (child is TileNode tilenode)
            {
                if (TileNodeMemo[tilenode].BufferInfoMap.ContainsKey(bid))
                {
                    return i;
                }
            }
            else if (child is OpNode opnode)
            {
                if (bid.Node.OpId == opnode.OpId)
                {
                    return i;
                }
            }
        }

        return -1;
    }

    private sealed class AffineIOTileOffsetRebaser : ExprCloner<Unit>
    {
        private const BindingFlags AttributeFlags = BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic;
        private readonly Dimension[] _domainOffsets;

        public AffineIOTileOffsetRebaser(IReadOnlyList<Dimension> domainOffsets)
        {
            CloneUnmutated = false;
            _domainOffsets = domainOffsets.ToArray();
        }

        protected override BaseExpr VisitLeafCall(Call expr, Unit context)
        {
            var target = Clone(expr.Target, context);
            var arguments = CloneArray(expr.Arguments, context);
            if (TryRebaseTarget(target, out var rebasedTarget))
            {
                return new Call(rebasedTarget, arguments).InheritMetaData(expr);
            }

            if (!ReferenceEquals(target, expr.Target) || !AreSameArguments(arguments, expr.Arguments))
            {
                return expr.With(target: target, arguments: arguments);
            }

            return expr;
        }

        private bool TryRebaseTarget(Expr target, [MaybeNullWhen(false)] out Expr rebasedTarget)
        {
            rebasedTarget = null;
            var type = target.GetType();
            var relationProperty = type.GetProperty(nameof(AffineRelation), AttributeFlags)
                ?? type.GetProperty("Relation", AttributeFlags);
            var symbolsProperty = type.GetProperty("Symbols", AttributeFlags);
            if (relationProperty is null ||
                symbolsProperty is null ||
                relationProperty.PropertyType != typeof(AffineRelation) ||
                symbolsProperty.PropertyType != typeof(RankedShape))
            {
                return false;
            }

            var relation = (AffineRelation?)relationProperty.GetValue(target)
                ?? throw new InvalidOperationException($"{type.Name}.Relation is null.");
            var symbols = (RankedShape?)symbolsProperty.GetValue(target)
                ?? throw new InvalidOperationException($"{type.Name}.Symbols is null.");

            if (relation.Domains.Length > _domainOffsets.Length)
            {
                throw new InvalidOperationException(
                    $"{type.Name} relation rank {relation.Domains.Length} exceeds tiled domain rank {_domainOffsets.Length}.");
            }

            ValidateCanonicalRelation(relation, type.Name);

            var domainOffsetSymbols = new Dictionary<int, int>();
            var appendedSymbolValues = new List<Dimension>();
            for (int i = 0; i < relation.Domains.Length; i++)
            {
                var offset = _domainOffsets[i];
                if (IsZero(offset))
                {
                    continue;
                }

                domainOffsetSymbols.Add(i, relation.Symbols.Length + appendedSymbolValues.Count);
                appendedSymbolValues.Add(offset);
            }

            if (appendedSymbolValues.Count == 0)
            {
                return false;
            }

            var relationSymbols = relation.Symbols.ToArray();
            if (symbols.Rank != relationSymbols.Length)
            {
                throw new InvalidOperationException(
                    $"{type.Name} symbol payload rank {symbols.Rank} does not match relation symbol rank {relationSymbols.Length}.");
            }

            var newRelationSymbols = relationSymbols
                .Concat(appendedSymbolValues.Select((offset, i) =>
                {
                    var symbol = new AffineSymbol(relationSymbols.Length + i)
                    {
                        Metadata = { Range = offset.Metadata.Range },
                    };
                    return symbol;
                }))
                .ToArray();
            var newSymbolValues = new RankedShape(symbols.Dimensions.ToArray().Concat(appendedSymbolValues).ToArray());
            var newResults = relation.Results.ToArray()
                .Select(result => RebaseAffineExpr(result, domainOffsetSymbols))
                .ToArray();
            var newConstraint = (LogicalExpr)new DomainConstraintRebaser(_domainOffsets, relation.Domains.Length)
                .Clone(relation.Constraint, default);
            var newRelation = relation.With(symbols: newRelationSymbols, results: newResults, constraint: newConstraint);
            rebasedTarget = RecreateTarget(target, newRelation, newSymbolValues);
            return true;
        }

        private bool IsZero(Dimension dim) => dim.IsFixed && dim.FixedValue == 0;

        private void ValidateCanonicalRelation(AffineRelation relation, string opName)
        {
            for (int i = 0; i < relation.Domains.Length; i++)
            {
                if (relation.Domains[i].Position != i)
                {
                    throw new InvalidOperationException(
                        $"{opName} relation domain at index {i} has non-canonical position {relation.Domains[i].Position}.");
                }
            }

            for (int i = 0; i < relation.Symbols.Length; i++)
            {
                if (relation.Symbols[i].Position != i)
                {
                    throw new InvalidOperationException(
                        $"{opName} relation symbol at index {i} has non-canonical position {relation.Symbols[i].Position}.");
                }
            }
        }

        private AffineExpr RebaseAffineExpr(AffineExpr expr, IReadOnlyDictionary<int, int> domainOffsetSymbols)
        {
            return expr switch
            {
                AffineDim dim when domainOffsetSymbols.TryGetValue(dim.Position, out var symbolPosition) => dim + new AffineSymbol(symbolPosition),
                AffineDim or AffineExtent or AffineSymbol or AffineConstant => expr,
                AffineAddBinary add => RebaseAffineExpr(add.Lhs, domainOffsetSymbols) + RebaseAffineExpr(add.Rhs, domainOffsetSymbols),
                AffineMulBinary mul => new AffineMulBinary(RebaseAffineExpr(mul.Lhs, domainOffsetSymbols), RebaseAffineExpr(mul.Rhs, domainOffsetSymbols)),
                AffineDivBinary div => new AffineDivBinary(div.BinaryOp, RebaseAffineExpr(div.Lhs, domainOffsetSymbols), RebaseAffineExpr(div.Rhs, domainOffsetSymbols)),
                _ => throw new NotSupportedException($"Unsupported affine expression node {expr.GetType().Name}."),
            };
        }

        private bool AreSameArguments(IReadOnlyList<BaseExpr> lhs, ReadOnlySpan<BaseExpr> rhs)
        {
            if (lhs.Count != rhs.Length)
            {
                return false;
            }

            for (int i = 0; i < lhs.Count; i++)
            {
                if (!ReferenceEquals(lhs[i], rhs[i]))
                {
                    return false;
                }
            }

            return true;
        }

        private Expr RecreateTarget(Expr target, AffineRelation relation, RankedShape symbols)
        {
            var type = target.GetType();
            foreach (var ctor in type.GetConstructors(AttributeFlags))
            {
                var args = TryBuildConstructorArguments(target, ctor, relation, symbols);
                if (args is not null)
                {
                    return (Expr)ctor.Invoke(args);
                }
            }

            throw new InvalidOperationException($"Cannot recreate affine IO target {type.FullName} with rebased relation.");
        }

        private object?[]? TryBuildConstructorArguments(Expr target, ConstructorInfo constructor, AffineRelation relation, RankedShape symbols)
        {
            var type = target.GetType();
            var parameters = constructor.GetParameters();
            var args = new object?[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                var parameter = parameters[i];
                if (parameter.ParameterType == typeof(AffineRelation))
                {
                    args[i] = relation;
                    continue;
                }

                if (parameter.ParameterType == typeof(RankedShape))
                {
                    args[i] = symbols;
                    continue;
                }

                var property = type.GetProperties(AttributeFlags)
                    .FirstOrDefault(prop =>
                        prop.GetIndexParameters().Length == 0 &&
                        string.Equals(prop.Name, parameter.Name, StringComparison.OrdinalIgnoreCase) &&
                        parameter.ParameterType.IsAssignableFrom(prop.PropertyType));
                if (property is null)
                {
                    return null;
                }

                args[i] = property.GetValue(target);
            }

            return args;
        }

        private sealed class DomainConstraintRebaser : ExprCloner<Unit>
        {
            private readonly IReadOnlyList<Dimension> _domainOffsets;
            private readonly int _domainRank;

            public DomainConstraintRebaser(IReadOnlyList<Dimension> domainOffsets, int domainRank)
            {
                CloneUnmutated = false;
                _domainOffsets = domainOffsets;
                _domainRank = domainRank;
            }

            protected override BaseExpr VisitLeafDimVar(DimVar expr, Unit context)
            {
                if (TryGetDomainIndex(expr, out var index))
                {
                    var offset = _domainOffsets[index];
                    return IsZero(offset) ? expr : expr + offset;
                }

                return expr;
            }

            private bool IsZero(Dimension dim) => dim.IsFixed && dim.FixedValue == 0;

            private bool TryGetDomainIndex(DimVar dimVar, out int index)
            {
                if (dimVar.Name.Length > 1 &&
                    dimVar.Name[0] == 'd' &&
                    int.TryParse(dimVar.Name[1..], out index) &&
                    index >= 0 &&
                    index < _domainRank)
                {
                    return true;
                }

                index = -1;
                return false;
            }
        }
    }

    public sealed record Context(ISequentialBuilder<Expr> ParentBuilder, Dimension[] ForwardOffsets, Dimension[] ForwardExtents)
    {
    }
}
