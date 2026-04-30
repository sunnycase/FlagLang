// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.IR.Logics;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tiling;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes;

public sealed class NTTTIRSelectionPass : TIRSelectionPass
{
    private readonly CompileOptions _compileOptions;

    public NTTTIRSelectionPass(CompileOptions compileOptions, string moduleKind = CPUTarget.Kind)
        : base(moduleKind)
    {
        _compileOptions = compileOptions;
    }

    protected override bool TrySelectBlock(IRBlock block, bool isEntry, out Sequential body, out IReadOnlyList<Var> outputBuffers)
    {
        if (TryLowerBlockLocalSmemAffineBlock(block, out body))
        {
            outputBuffers = Array.Empty<Var>();
            return true;
        }

        if (TryLowerRegisterDirectAffineBlock(block, out body))
        {
            outputBuffers = Array.Empty<Var>();
            return true;
        }

        outputBuffers = Array.Empty<Var>();
        return false;
    }

    protected override Expr SelectCall(Call call, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        var op = call.Target;
        switch (op)
        {
            case IR.Math.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.CustomNTT.Unary unary:
                return GenerateUnary(unary.UnaryOp, arguments, output);
            case IR.Math.Clamp clamp:
                return GenerateClamp(call, arguments, output);
            case IR.Distributed.Boxing boxing:
                return GenerateBoxing(call, boxing, arguments, ref output);
            case IR.Distributed.ForceBoxing forceBoxing:
                return T.Memcopy(output, (Expr)arguments[0]);
            case IR.Math.Binary binary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, None.Default, binary.BinaryOp, Array.Empty<int>(), Array.Empty<Dimension>(), Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.Tensors.Bitcast bitcast:
                return GenerateBitcast((Expr)arguments[0], ref output, bitcast.NewType);
            case IR.Tensors.Pack pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, pack.Lanes, pack.Axes);
            case IR.Tensors.VectorizeMask pack:
                return TIR.F.NTT.Pack((Expr)arguments[0], output, new[] { pack.Lanes }, new[] { pack.Axis });
            case IR.Tensors.Unpack unpack:
                return TIR.F.NTT.Unpack((Expr)arguments[0], output, unpack.Lanes, unpack.Axes);
            case IR.NTT.VectorizedBinary vectorizedBinary:
                return TIR.F.NTT.VectorizedBinary((Expr)arguments[0], (Expr)arguments[1], output, (Expr)arguments[2], vectorizedBinary.BinaryOp, vectorizedBinary.LhsVectorizedAxes, vectorizedBinary.LhsPadedNums, vectorizedBinary.RhsVectorizedAxes, vectorizedBinary.RhsPadedNums);
            case IR.NTT.VectorizedMatMul vectorizedMatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                ValidateLegacyAxisPolicySelection(dta, "VectorizedMatMul lhs");
                ValidateLegacyAxisPolicySelection(dtb, "VectorizedMatMul rhs");
                ValidateLegacyAxisPolicySelection(output, "VectorizedMatMul output");
                var dinfo = vectorizedMatMul.GetDimInfo(dta.TensorType.Shape.Rank, dtb.TensorType.Shape.Rank);
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[dinfo.Lk] == dtb.AxisPolicies[dinfo.Rn] &&
                    dta.AxisPolicies[dinfo.Lm] == dtb.AxisPolicies[dinfo.Rk])
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.VectorizedMatMul.Scale], None.Default, vectorizedMatMul.LhsVectorizedAxes, vectorizedMatMul.RhsVectorizedAxes, vectorizedMatMul.TransposeA, vectorizedMatMul.TransposeB, vectorizedMatMul.FusedReduce);
                }

            case IR.Math.MatMul when GetArgumentType(arguments[0]) is DistributedType dta && GetArgumentType(arguments[1]) is DistributedType dtb:
                ValidateLegacyAxisPolicySelection(dta, "MatMul lhs");
                ValidateLegacyAxisPolicySelection(dtb, "MatMul rhs");
                ValidateLegacyAxisPolicySelection(output, "MatMul output");
                if (dta.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dtb.AxisPolicies[^2..].AsValueEnumerable().All(x => x is SBPSplit) &&
                    dta.AxisPolicies[^2] == dtb.AxisPolicies[^2] &&
                    dta.AxisPolicies[^1] == dtb.AxisPolicies[^1])
                {
                    return TIR.F.NTT.SUMMA((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }
                else
                {
                    return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.Math.MatMul.Scale]);
                }

            case IR.CustomNTT.MatMul matmul:
                return TIR.F.NTT.Matmul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.CustomNTT.MatMul.Scale], (Expr)arguments[3], matmul.LhsVectorizedAxes, matmul.RhsVectorizedAxes, matmul.TransposeA, matmul.TransposeB, false, matmul.CSourcePath, matmul.FuncName);
            case IR.NTT.PackedMatMul matmul:
                return TIR.F.NTT.PackedMatMul((Expr)arguments[0], (Expr)arguments[1], output, None.Default, (Expr)call[IR.NTT.PackedMatMul.Scale], matmul.FusedReduce);
            case IR.NN.Conv2D conv:
                {
                    var input = call[IR.NN.Conv2D.Input];
                    var weights = call[IR.NN.Conv2D.Weights];
                    var bias = call[IR.NN.Conv2D.Bias];
                    var strides = ((RankedShape)call[IR.NN.Conv2D.Stride]).ToValueArray();
                    var padding = Tensor.From(((Paddings)call[IR.NN.Conv2D.Padding]).ToValueArray()).ToArray();
                    var dilation = ((RankedShape)call[IR.NN.Conv2D.Dilation]).ToValueArray();
                    var groups = ((Dimension)call[IR.NN.Conv2D.Groups]).FixedValue;
                    var fusedClamp = ((TensorConst)call[IR.NN.Conv2D.FusedClamp]).Value.ToArray<float>();
                    var wShape = weights.CheckedShape.ToValueArray();
                    var outShape = call.CheckedShape.ToValueArray();
                    if (fusedClamp[0] != float.NegativeInfinity || fusedClamp[1] != float.PositiveInfinity || conv.PadMode != PadMode.Constant)
                    {
                        throw new NotSupportedException("not support this conv2d");
                    }

                    return TIR.F.NTT.Conv2D((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, strides, padding, dilation, groups, conv.PadMode, call.CheckedType is DistributedType dt_conv ? dt_conv : null!);
                }

            case IR.NTT.Im2col im2col:
                return TIR.F.NTT.Im2col((Expr)arguments[0], output, im2col.Kernel, im2col.Stride, im2col.Padding, im2col.VectorizedAxes, im2col.PadedNums);
            case IR.NTT.VectorizedRoPE rope:
                return TIR.F.NTT.RoPE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Imaging.ResizeImage resize:
                if ((call[IR.Imaging.ResizeImage.Roi] is not None && ((RankedShape)call[IR.Imaging.ResizeImage.Roi].CheckedShape).Size != 0) || resize.IsTFResize)
                {
                    throw new NotSupportedException("not support tf resize");
                }

                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, Array.Empty<int>(), Array.Empty<Dimension>(), ((RankedShape)call[IR.Imaging.ResizeImage.NewSize]).ToValueArray().ToInts(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.NTT.ResizeImage resize:
                return TIR.F.NTT.ResizeImage((Expr)arguments[0], output, resize.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.ResizeImage.PadedNums]).Dimensions.ToArray(), resize.NewSize.ToArray(), resize.ResizeMode, resize.TransformationMode, resize.NearestMode);
            case IR.Tensors.Slice slice:
                return TIR.F.NTT.Slice((Expr)arguments[0], (RankedShape)arguments[1], (RankedShape)arguments[2], output, ((RankedShape)call[IR.Tensors.Slice.Axes]).ToValueArray().ToInts(), ((RankedShape)call[IR.Tensors.Slice.Strides]).ToValueArray().ToInts());
            case IR.Tensors.Concat concat:
                return TIR.F.NTT.Concat(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, concat.Axis);
            case IR.Tensors.Transpose trans:
                return TIR.F.NTT.Transpose((Expr)arguments[0], output, ((RankedShape)call[IR.Tensors.Transpose.Perm]).ToValueArray().ToInts());
            case IR.NN.Swish swish:
                return TIR.F.NTT.Swish((Expr)arguments[0], output, ((TensorConst)call[IR.NN.Swish.Beta]).Value.ToScalar<float>());
            case IR.Tensors.Gather gather:
                return TIR.F.NTT.Gather((Expr)arguments[0], (Expr)arguments[1], output, gather.Axis);
            case IR.Affine.Gather affineGather:
                return TIR.F.NTT.AffineGather((Expr)arguments[0], (Expr)arguments[1], output, affineGather.Relation, affineGather.Symbols, affineGather.Shape);
            case IR.NN.Pad pad:
                var paddings = (Paddings)call[IR.NN.Pad.Pads];
                var actualPadAxes = Enumerable.Range(0, paddings.Count).Where(i => !(paddings[i] is { IsFixed: true } pad && pad.Sum() == 0)).ToArray();
                return TIR.F.NTT.Pad((Expr)arguments[0], output, paddings, ((TensorConst)call[IR.NN.Pad.Value]).Value.ToArray<float>()[0], actualPadAxes);
            case IR.Math.Reduce reduce:
                var reduceInputType = call[IR.Math.Reduce.Input].CheckedTensorType;
                var reduceAxes = ((RankedShape)call[IR.Math.Reduce.Axes])
                    .ToValueArray()
                    .Select(x => Util.PositiveIndex(x, reduceInputType))
                    .OrderBy(a => a)
                    .ToArray()
                    .ToInts();
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, Array.Empty<int>(), Array.Empty<Dimension>(), reduceAxes, ((TensorConst)call[IR.Math.Reduce.KeepDims]).Value.ToArray<bool>()[0], reduce.ReduceOp);
            case IR.Math.ReduceArg reduceArg:
                return TIR.F.NTT.ReduceArg((Expr)arguments[0], output, (int)((DimConst)call[IR.Math.ReduceArg.Axis]).FixedValue, ((TensorConst)call[IR.Math.ReduceArg.KeepDims]).Value.ToArray<bool>()[0], ((TensorConst)call[IR.Math.ReduceArg.SelectLastIndex]).Value.ToArray<bool>()[0], reduceArg.ReduceArgOp, reduceArg.DestType);
            case IR.Tensors.Cast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, Array.Empty<int>(), None.Default);
            case IR.NTT.VectorizedCast cast:
                return TIR.F.NTT.Cast((Expr)arguments[0], output, cast.NewType, cast.CastMode, cast.VectorizeAxes, (Expr)arguments[1]);
            case IR.Tensors.Where where:
                return TIR.F.NTT.Where((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Tensors.Broadcast:
                return TIR.F.NTT.Expand((Expr)arguments[0], output);
            case IR.Tensors.Expand expand:
                return TIR.F.NTT.Expand((Expr)arguments[0], output);
            case IR.NN.Erf erf:
                return TIR.F.NTT.Erf((Expr)arguments[0], output);
            case IR.NTT.VectorizedReduce pr:
                return TIR.F.NTT.Reduce((Expr)arguments[0], output, false, pr.VectorizedAxes.ToArray(), ((RankedShape)call[IR.NTT.VectorizedReduce.PadedNums]).Dimensions.ToArray(), pr.Axes, pr.KeepDims, pr.ReduceOp);
            case IR.Math.Compare compare:
                return TIR.F.NTT.Compare(compare.CompareOp, (Expr)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.GetItem getItem:
                return TIR.F.NTT.GetItem((Expr)arguments[0], arguments[1], output);
            case IR.Tensors.Reshape:
                return GenerateReshape((Expr)arguments[0], ref output);
            case IR.Tensors.ScatterND scatterND:
                return TIR.F.NTT.ScatterND((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Affine.Scatter affineScatter:
                return TIR.F.NTT.AffineScatter((Expr)arguments[0], (Expr)arguments[1], affineScatter.Relation, affineScatter.Symbols);
            case IR.Tensors.Stack stack:
                return TIR.F.NTT.Stack(((IR.Tuple)arguments[0]).Fields.AsValueEnumerable().Select(x => (Expr)x).ToArray(), output, ((TensorConst)call[IR.Tensors.Stack.Axis]).Value.ToScalar<int>());
            case IR.Tensors.Unsqueeze:
                return GenerateReshape((Expr)arguments[0], ref output);
            case IR.NN.UpdatePagedAttentionKVCache upkv:
                output = (Expr)arguments[1];
                return TIR.F.NTT.UpdatePagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], upkv.CacheKind, upkv.LayerId, upkv.Layout);
            case IR.NN.GatherPagedAttentionKVCache gakv:
                return TIR.F.NTT.GatherPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], output);
            case IR.NN.CreatePagedAttentionKVCache ctkv:
                return TIR.F.NTT.CreatePagedAttentionKVCache(ctkv.Config, (Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], output);
            case IR.NN.IdentityPagedAttentionKVCache ctkv:
                output = (Expr)arguments[0];
                return TIR.F.NTT.IdentityPagedAttentionKVCache((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8]);
            case IR.NN.PagedAttention pgat:
                return TIR.F.NTT.PagedAttention((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], pgat.LayerId, output, pgat.Layout, pgat.HiddenSize);
            case IR.Tensors.ConstantOfShape constantOfShape:
                return TIR.F.NTT.ConstantOfShape((Shape)arguments[0], (Expr)arguments[1], output);
            case IR.Tensors.Range range:
                return TIR.F.NTT.Range((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output);
            case IR.Buffers.Uninitialized uninitialized:
                return T.Nop();
            case IR.Shapes.AsTensor asTensor:
                output = call;
                return call;
            case IR.NN.GetPositionIds getPositionIds:
                return TIR.F.NTT.GetPositionIds((Expr)arguments[1], output, (DistributedType)call.CheckedType);
            case IR.NN.LayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, Array.Empty<int>(), Array.Empty<Dimension>());
            case IR.NTT.VectorizedLayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.VectorizedAxes, ((RankedShape)call[IR.NTT.VectorizedLayerNorm.PadedNums]).Dimensions.ToArray());
            case IR.CustomNTT.LayerNorm ln:
                return TIR.F.NTT.VectorizedLayerNorm((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], output, ln.Axis, ln.Epsilon, ln.UseMean, ln.VectorizedAxes, Array.Empty<Dimension>(), (Expr)call[IR.CustomNTT.LayerNorm.PostScale], ln.CSourcePath, ln.FuncName);
            case IR.NN.Softmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, (int)((DimConst)call[IR.NN.Softmax.Axis]).FixedValue, Array.Empty<int>());
            case IR.NTT.VectorizedSoftmax softmax:
                return TIR.F.NTT.VectorizedSoftmax((Expr)arguments[0], output, softmax.Axis, softmax.VectorizedAxes);
            case IR.NN.Qwen3MoE moe:
                return TIR.F.NTT.Qwen3MoE((Expr)arguments[0], (Expr)arguments[1], (Expr)arguments[2], (Expr)arguments[3], (Expr)arguments[4], (Expr)arguments[5], (Expr)arguments[6], (Expr)arguments[7], (Expr)arguments[8], (Expr)arguments[9], (Expr)arguments[10], output, moe.LayerId, moe.HiddenSize, moe.IntermediateSize, moe.MoEIntermediateSize, moe.NumExpert, moe.NumTopK, moe.IsNormTopkProb);
            default:
                throw new NotSupportedException($"Not supported: {op}");
        }
    }

    private bool TryLowerBlockLocalSmemAffineBlock(IRBlock block, out Sequential body)
    {
        body = null!;
        var blockBody = UnwrapNestedBlock(block.Body);
        if (!ContainsBlockLocalSmemDecision(blockBody))
        {
            return false;
        }

        var groups = BuildBlockLocalSmemGroups(GetBlockLocalSmemFields(blockBody));
        var buffers = new Dictionary<Call, TIR.Buffer>(ReferenceEqualityComparer.Instance);
        var loweredFields = new List<Expr>();
        for (var i = 0; i < groups.Count; i++)
        {
            var group = groups[i];
            var sourceBuffer = EnsureBlockLocalSmemGatherLowered(group.GatherCall, buffers, loweredFields);
            loweredFields.Add(BuildBlockLocalSmemScatterLoop(group.ScatterCalls, group.GatherCall, sourceBuffer));

            if (i + 1 < groups.Count)
            {
                loweredFields.Add(TIR.F.NTT.SynchronizeThreads());
            }
        }

        loweredFields.Add(T.Return());
        body = new Sequential(loweredFields.ToArray());
        return true;
    }

    private List<(Call GatherCall, List<Call> ScatterCalls)> BuildBlockLocalSmemGroups(Expr[] fields)
    {
        var groups = new Dictionary<Call, (Call GatherCall, List<Call> ScatterCalls)>(ReferenceEqualityComparer.Instance);
        var orderedGroups = new List<(Call GatherCall, List<Call> ScatterCalls)>();
        foreach (var field in fields)
        {
            if (field is Call { Target: IR.Affine.Gather } gatherCall && IsBlockLocalSmem(gatherCall))
            {
                EnsureBlockLocalSmemGroup(gatherCall, groups, orderedGroups);
                continue;
            }

            if (field is Call { Target: IR.Affine.Scatter } scatterCall &&
                TryGetBlockLocalSmemGatherSource(scatterCall, out var sourceGather))
            {
                EnsureBlockLocalSmemGroup(sourceGather, groups, orderedGroups).ScatterCalls.Add(scatterCall);
                continue;
            }

            if (ContainsBlockLocalSmemDecision(field))
            {
                throw new NotSupportedException($"Unsupported block-local SMem affine field {field.GetType().Name}. Lowering currently supports shared Affine.Gather producers consumed directly by multiple Affine.Scatter calls.");
            }

            throw new NotSupportedException($"Block-local SMem affine lowering cannot preserve unrelated block field {field.GetType().Name}; split the block before tiling.");
        }

        foreach (var group in orderedGroups)
        {
            if (group.ScatterCalls.Count < 2)
            {
                throw new InvalidOperationException($"Block-local SMem affine lowering requires every Affine.Gather to have at least two direct Affine.Scatter consumers to justify shared storage, got {group.ScatterCalls.Count}.");
            }
        }

        return orderedGroups;
    }

    private (Call GatherCall, List<Call> ScatterCalls) EnsureBlockLocalSmemGroup(
        Call gatherCall,
        IDictionary<Call, (Call GatherCall, List<Call> ScatterCalls)> groups,
        IList<(Call GatherCall, List<Call> ScatterCalls)> orderedGroups)
    {
        if (groups.TryGetValue(gatherCall, out var existing))
        {
            return existing;
        }

        var group = (gatherCall, new List<Call>());
        groups.Add(gatherCall, group);
        orderedGroups.Add(group);
        return group;
    }

    private Expr[] GetBlockLocalSmemFields(BaseExpr body) => body switch
    {
        Sequential sequential => sequential.Fields.ToArray(),
        IR.Tuple tuple => tuple.Fields.ToArray().Select(field => (Expr)field).ToArray(),
        Expr expr => [expr],
        _ => throw new NotSupportedException($"Block-local SMem affine lowering requires an expression block, got {body.GetType().Name}."),
    };

    private TIR.Buffer EnsureBlockLocalSmemGatherLowered(Call gatherCall, IDictionary<Call, TIR.Buffer> buffers, IList<Expr> loweredFields)
    {
        if (buffers.TryGetValue(gatherCall, out var existing))
        {
            return existing;
        }

        var gather = (IR.Affine.Gather)gatherCall.Target;
        var decision = TileDecisionMetadata.Require(gatherCall, "Block-local SMem affine TIR selection");
        ValidateBlockLocalSmemDecision(decision, "Affine.Gather");
        var output = CreateBlockLocalSmemBuffer(gatherCall, decision, buffers.Count);
        buffers.Add(gatherCall, output);
        loweredFields.Add(TIR.F.NTT.AffineGather(
            (Expr)gatherCall[IR.Affine.Gather.Source],
            (Expr)gatherCall[IR.Affine.Gather.DefaultValue],
            output,
            gather.Relation,
            gather.Symbols,
            gather.Shape).InheritMetaData(gatherCall));
        loweredFields.Add(TIR.F.NTT.SynchronizeThreads());
        return output;
    }

    private bool ContainsBlockLocalSmemDecision(BaseExpr expr) =>
        ExprCollector.Collect(expr)
            .OfType<Call>()
            .Any(IsBlockLocalSmem);

    private bool IsBlockLocalSmem(Call call) =>
        TileDecisionMetadata.TryGet(call, out var decision) &&
        decision.Storage is { Scope: BufferScope.BlockLocal, PhysicalLocation: PhysicalMemorySpace.SMem };

    private bool TryGetBlockLocalSmemGatherSource(Call scatterCall, [MaybeNullWhen(false)] out Call sourceGather)
    {
        sourceGather = null;
        if (scatterCall.Target is not IR.Affine.Scatter)
        {
            return false;
        }

        var sources = new List<Call>();
        if (!TryCollectBlockLocalSmemGatherSources((Expr)scatterCall[IR.Affine.Scatter.Source], sources, new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance)) ||
            sources.Count == 0)
        {
            return false;
        }

        var uniqueSources = sources.Distinct(new ReferenceEqualityComparer<Call>()).ToArray();
        if (uniqueSources.Length != 1)
        {
            throw new NotSupportedException($"Block-local SMem affine lowering currently supports one shared Affine.Gather per scatter expression, got {uniqueSources.Length}.");
        }

        sourceGather = uniqueSources[0];
        return true;
    }

    private bool TryCollectBlockLocalSmemGatherSources(Expr expr, List<Call> sources, ISet<BaseExpr> visited)
    {
        if (expr is TensorConst { CheckedType: TensorType { IsScalar: true } })
        {
            return true;
        }

        if (expr is not Call call)
        {
            return false;
        }

        if (!visited.Add(call))
        {
            return true;
        }

        switch (call.Target)
        {
            case IR.Affine.Gather when IsBlockLocalSmem(call):
                sources.Add(call);
                return true;
            case IR.Math.Unary:
            case IR.Math.Binary:
            case IR.Tensors.Cast:
                return call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectBlockLocalSmemGatherSources(arg, sources, visited));
            case IR.Tensors.Where where when !where.IsTfWhere:
                return call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectBlockLocalSmemGatherSources(arg, sources, visited));
            case PrimFunctionWrapper { Target: PrimFunction primFunction }:
                _ = GetPrimWrapperBinaryOp(primFunction);
                return call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectBlockLocalSmemGatherSources(arg, sources, visited));
            default:
                return false;
        }
    }

    private TIR.Buffer CreateBlockLocalSmemBuffer(Call gatherCall, TileDecision decision, int bufferIndex)
    {
        var (tensorType, distributedType) = GetTensorAndDistributedType(gatherCall.CheckedType, "Block-local SMem affine gather");
        var bufferDistributedType = CreateBlockLocalSmemDistributedType(gatherCall, tensorType, distributedType, decision);
        var tileType = new TensorType(tensorType.DType, decision.StorageLayout.LogicalShape);
        return T.CreateBuffer(tileType, decision.Storage, out _, $"smem_tile_{bufferIndex}", bufferDistributedType);
    }

    private DistributedType? CreateBlockLocalSmemDistributedType(Call gatherCall, TensorType tensorType, DistributedType? distributedType, TileDecision decision)
    {
        if (decision.DistributionLayout is null)
        {
            return null;
        }

        if (decision.StorageLayout.ViewMap is null)
        {
            throw new NotSupportedException("Distributed block-local SMem lowering requires a storage layout view map from owner-local coordinates to shared storage coordinates.");
        }

        var gather = (IR.Affine.Gather)gatherCall.Target;
        var placement = distributedType?.Placement ?? gather.Placement;
        var axisPolicies = distributedType?.AxisPolicies ?? gather.NdSBP;
        var partial = distributedType?.Partial ?? false;
        return DistributedType.FromLayouts(
            tensorType,
            placement,
            decision.DistributionLayout,
            decision.StorageLayout,
            axisPolicies,
            partial);
    }

    private (TensorType TensorType, DistributedType? DistributedType) GetTensorAndDistributedType(IRType type, string context) => type switch
    {
        TensorType tensorType => (tensorType, null),
        DistributedType distributedType => (distributedType.TensorType, distributedType),
        _ => throw new NotSupportedException($"{context} requires a tensor output, got {type}."),
    };

    private void ValidateBlockLocalSmemDecision(TileDecision decision, string opKind)
    {
        if (decision.Storage is not { Scope: BufferScope.BlockLocal, PhysicalLocation: PhysicalMemorySpace.SMem })
        {
            throw new InvalidOperationException($"{opKind} block-local lowering requires SMem tile storage, got {decision.Storage}.");
        }

        if (decision.StorageLayout.LogicalShape.IsUnranked ||
            decision.StorageLayout.LogicalShape.Rank != 1 ||
            !decision.StorageLayout.LogicalShape[0].IsFixed)
        {
            throw new NotSupportedException($"{opKind} block-local SMem lowering requires a fixed rank-1 storage shape, got {decision.StorageLayout.LogicalShape}.");
        }
    }

    private Expr BuildBlockLocalSmemScatterLoop(IReadOnlyList<Call> scatterCalls, Call sourceGather, TIR.Buffer sourceBuffer)
    {
        if (scatterCalls.Count == 0)
        {
            throw new InvalidOperationException("Block-local SMem affine lowering requires at least one Affine.Scatter consumer.");
        }

        var firstScatterCall = scatterCalls[0];
        var scatter = (IR.Affine.Scatter)firstScatterCall.Target;
        var gather = (IR.Affine.Gather)sourceGather.Target;
        var gatherDecision = TileDecisionMetadata.Require(sourceGather, "Block-local SMem affine expression TIR selection");
        ValidateBlockLocalSmemDecision(gatherDecision, "Affine.Gather");
        foreach (var scatterCall in scatterCalls)
        {
            ValidateBlockLocalSmemAffineChain(scatterCall, (IR.Affine.Scatter)scatterCall.Target, sourceGather, gather);
        }

        var globalExtent = sourceBuffer.Dimensions[0];
        var iterationExtents = AffineIOLayoutEvaluator.GetIterationExtents(sourceBuffer);
        if (iterationExtents.Length != 1)
        {
            throw new NotSupportedException($"Block-local SMem affine lowering requires a rank-1 source buffer, got {iterationExtents.Length} iteration axes.");
        }

        var symbolMap = BuildSymbolMap(scatter.Relation, scatter.Symbols);
        return T.Serial(out var lane, new TIR.Range(Dimension.Zero, iterationExtents[0], Dimension.One), "d0")
            .Body(BuildBlockLocalSmemStoreBody(scatterCalls, scatter, sourceGather, sourceBuffer, globalExtent, symbolMap, lane))
            .Build()
            .InheritMetaData(firstScatterCall);
    }

    private Expr BuildBlockLocalSmemStoreBody(
        IReadOnlyList<Call> scatterCalls,
        IR.Affine.Scatter scatter,
        Call sourceGather,
        TIR.Buffer sourceBuffer,
        Dimension globalExtent,
        IReadOnlyDictionary<int, Dimension>? symbolMap,
        DimVar lane)
    {
        var loopVars = new[] { lane };
        var extents = new[] { globalExtent };
        var domainValues = AffineIOLayoutEvaluator.GetDomainValues(sourceBuffer, loopVars, extents);
        var address = EvaluateAddress(scatter.Relation, domainValues, extents, symbolMap);
        var loaded = T.BufferLoad(sourceBuffer, AffineIOLayoutEvaluator.GetStorageIndices(sourceBuffer, loopVars, domainValues));
        return T.Let(out var smemValue, loaded).Body(BuildStore(smemValue)).Build();

        Expr BuildStore(Expr sharedValue)
        {
            var stores = scatterCalls.Select(scatterCall => BuildBlockLocalSmemStore(scatterCall, sourceGather, sharedValue, address)).ToArray();
            return scatter.Relation.Constraint == LogicalExpr.True
                ? new Sequential(stores)
                : T.If(EvaluateConstraint(scatter.Relation.Constraint, domainValues)).Then(stores.Cast<object>().ToArray()).Build();
        }
    }

    private Expr BuildBlockLocalSmemStore(Call scatterCall, Call sourceGather, Expr sharedValue, Dimension address)
    {
        var source = (Expr)scatterCall[IR.Affine.Scatter.Source];
        var dest = (Expr)scatterCall[IR.Affine.Scatter.Dest];
        var value = BuildBlockLocalSmemScalarExpr(source, sourceGather, sharedValue);
        return T.Store(dest, address, value);
    }

    private Expr BuildBlockLocalSmemScalarExpr(Expr expr, Call sourceGather, Expr sharedValue)
    {
        if (expr is TensorConst { CheckedType: TensorType { IsScalar: true } })
        {
            return expr;
        }

        if (ReferenceEquals(expr, sourceGather))
        {
            return sharedValue;
        }

        if (expr is not Call call)
        {
            throw new NotSupportedException($"Unsupported block-local SMem scalar expression {expr.GetType().Name}.");
        }

        return call.Target switch
        {
            IR.Affine.Gather when IsBlockLocalSmem(call) => throw new NotSupportedException("Block-local SMem scalar expression contains a second shared Affine.Gather that was not selected as the source value."),
            IR.Math.Unary unary => IR.F.Math.Unary(unary.UnaryOp, BuildBlockLocalSmemScalarExpr((Expr)call[IR.Math.Unary.Input], sourceGather, sharedValue)),
            IR.Math.Binary binary => IR.F.Math.Binary(
                binary.BinaryOp,
                BuildBlockLocalSmemScalarExpr((Expr)call[IR.Math.Binary.Lhs], sourceGather, sharedValue),
                BuildBlockLocalSmemScalarExpr((Expr)call[IR.Math.Binary.Rhs], sourceGather, sharedValue)),
            IR.Tensors.Cast cast => IR.F.Tensors.Cast(BuildBlockLocalSmemScalarExpr((Expr)call[IR.Tensors.Cast.Input], sourceGather, sharedValue), cast.NewType, cast.CastMode),
            IR.Tensors.Where where when !where.IsTfWhere => IR.F.Math.Select(
                BuildBlockLocalSmemScalarExpr((Expr)call[IR.Tensors.Where.Cond], sourceGather, sharedValue),
                BuildBlockLocalSmemScalarExpr((Expr)call[IR.Tensors.Where.X], sourceGather, sharedValue),
                BuildBlockLocalSmemScalarExpr((Expr)call[IR.Tensors.Where.Y], sourceGather, sharedValue)),
            PrimFunctionWrapper { Target: PrimFunction primFunction } => IR.F.Math.Binary(
                GetPrimWrapperBinaryOp(primFunction),
                BuildBlockLocalSmemScalarExpr((Expr)call.Arguments[0], sourceGather, sharedValue),
                BuildBlockLocalSmemScalarExpr((Expr)call.Arguments[1], sourceGather, sharedValue)),
            _ => throw new NotSupportedException($"Unsupported block-local SMem scalar call target {call.Target}."),
        };
    }

    private void ValidateBlockLocalSmemAffineChain(Call scatterCall, IR.Affine.Scatter scatter, Call gatherCall, IR.Affine.Gather gather)
    {
        if (gatherCall[IR.Affine.Gather.DefaultValue] is not None)
        {
            throw new NotSupportedException("Block-local SMem affine expression lowering only supports masked gathers whose default is None and whose value is consumed under the scatter guard.");
        }

        if (!IsSameAffineRelation(gather.Relation, scatter.Relation))
        {
            throw new NotSupportedException($"Block-local SMem affine expression lowering requires gather/scatter relation and symbols to match. Gather={gather.Relation}, Scatter={scatter.Relation}.");
        }

        if (!IsSameSymbolPayload(gather.Symbols, scatter.Symbols))
        {
            throw new NotSupportedException($"Block-local SMem affine expression lowering requires gather/scatter symbol payloads to match. Gather={gather.Symbols}, Scatter={scatter.Symbols}.");
        }

        if (scatterCall.CheckedType != TupleType.Void)
        {
            throw new NotSupportedException($"Block-local SMem affine scatter must be void, got {scatterCall.CheckedType}.");
        }
    }

    private bool TryLowerRegisterDirectAffineBlock(IRBlock block, out Sequential body)
    {
        body = null!;
        var blockBody = UnwrapNestedBlock(block.Body);
        var fields = GetRegisterBlockFields(blockBody);
        if (fields.Length == 0)
        {
            return false;
        }

        var lowerings = new List<RegisterScatterLowering>();
        foreach (var field in fields)
        {
            if (TryBuildRegisterScatterLowering(field, out var lowering))
            {
                lowerings.Add(lowering);
            }
        }

        if (lowerings.Count == 0)
        {
            return false;
        }

        var covered = new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance);
        foreach (var lowering in lowerings)
        {
            covered.Add(lowering.ScatterCall);
            foreach (var expr in lowering.CoveredCalls)
            {
                covered.Add(expr);
            }
        }

        foreach (var field in fields)
        {
            if (!IsCoveredRegisterBlockField(field, covered))
            {
                throw new NotSupportedException($"Register direct affine block lowering cannot preserve unrelated field {field.GetType().Name}. Split the block or add a supported lowering for this field.");
            }
        }

        var lowered = lowerings.Select(BuildRegisterScatterLoop).Append(T.Return()).ToArray();
        body = new Sequential(lowered);
        return true;
    }

    private bool TryBuildRegisterScatterLowering(Expr field, [MaybeNullWhen(false)] out RegisterScatterLowering lowering)
    {
        lowering = null;
        if (field is not Call { Target: IR.Affine.Scatter scatter } scatterCall)
        {
            return false;
        }

        if (!TileDecisionMetadata.TryGet(scatterCall, out var scatterDecision))
        {
            return false;
        }

        if (scatterDecision.Storage.PhysicalLocation is not PhysicalMemorySpace.Register)
        {
            return false;
        }

        var source = (Expr)scatterCall[IR.Affine.Scatter.Source];
        var gatherCalls = new List<Call>();
        var coveredCalls = new List<Call>();
        if (!TryCollectRegisterChain(source, coveredCalls, gatherCalls, new HashSet<BaseExpr>(ReferenceEqualityComparer.Instance)) ||
            gatherCalls.Count == 0)
        {
            throw new InvalidOperationException($"Register direct affine scatter requires a supported fused elementwise producer, got {source.GetType().Name}.");
        }

        ValidateRegisterAffineChain(scatterCall, scatter, gatherCalls, coveredCalls, scatterDecision);
        lowering = new RegisterScatterLowering(scatterCall, scatter, scatterDecision, source, gatherCalls, coveredCalls);
        return true;
    }

    private Expr BuildRegisterScatterLoop(RegisterScatterLowering lowering)
    {
        var sourceType = GetRegisterChainDistributedType(lowering.Source, lowering.GatherCalls);
        var globalExtent = sourceType?.TensorType.Shape[0] ?? lowering.Decision.TileShape[0];
        var tileExtent = lowering.Decision.TileShape[0];
        var symbolMap = BuildSymbolMap(lowering.Scatter.Relation, lowering.Scatter.Symbols);
        var dest = (Expr)lowering.ScatterCall[IR.Affine.Scatter.Dest];
        return T.Serial(out var lane, new TIR.Range(Dimension.Zero, tileExtent, Dimension.One), "d0")
            .Body(BuildRegisterStoreBody(lowering.Source, lowering.Scatter, lowering.Decision, sourceType, globalExtent, symbolMap, dest, lane))
            .Build();
    }

    private bool IsCoveredRegisterBlockField(Expr field, ISet<BaseExpr> covered)
    {
        if (covered.Contains(field))
        {
            return true;
        }

        var calls = ExprCollector.Collect(field).OfType<Call>().ToArray();
        return calls.Length > 0 && calls.All(covered.Contains);
    }

    private Expr BuildRegisterStoreBody(
        Expr source,
        IR.Affine.Scatter scatter,
        TileDecision scatterDecision,
        DistributedType? sourceType,
        Dimension globalExtent,
        IReadOnlyDictionary<int, Dimension>? symbolMap,
        Expr dest,
        DimVar lane)
    {
        var domainValue = GetRegisterDomainValue(scatterDecision, sourceType, lane, globalExtent);
        var domainValues = new[] { domainValue };
        var extents = new[] { globalExtent };
        var address = EvaluateAddress(scatter.Relation, domainValues, extents, symbolMap);
        var value = BuildRegisterScalarExpr(source, address);
        var store = T.Store(dest, address, value);
        return scatter.Relation.Constraint == LogicalExpr.True
            ? store
            : T.If(EvaluateConstraint(scatter.Relation.Constraint, domainValues)).Then(store).Build();
    }

    private bool TryCollectRegisterChain(Expr expr, List<Call> coveredCalls, List<Call> gatherCalls, ISet<BaseExpr> visited)
    {
        if (expr is TensorConst { CheckedType: TensorType { IsScalar: true } })
        {
            return true;
        }

        if (expr is not Call call)
        {
            return false;
        }

        if (!visited.Add(call))
        {
            return true;
        }

        switch (call.Target)
        {
            case IR.Affine.Gather:
                RequireRegisterDecision(call, "register direct affine gather");
                gatherCalls.Add(call);
                coveredCalls.Add(call);
                return true;
            case IR.Math.Unary:
            case IR.Math.Binary:
            case IR.Tensors.Cast:
                RequireRegisterDecision(call, "register direct affine elementwise producer");
                if (!call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectRegisterChain(arg, coveredCalls, gatherCalls, visited)))
                {
                    return false;
                }

                coveredCalls.Add(call);
                return true;
            case IR.Tensors.Where where:
                if (where.IsTfWhere)
                {
                    throw new NotSupportedException("Register direct affine lowering does not support TensorFlow-style Where.");
                }

                RequireRegisterDecision(call, "register direct affine where producer");
                if (!call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectRegisterChain(arg, coveredCalls, gatherCalls, visited)))
                {
                    return false;
                }

                coveredCalls.Add(call);
                return true;
            case PrimFunctionWrapper { Target: PrimFunction primFunction }:
                RequireRegisterDecision(call, "register direct affine prim wrapper producer");
                _ = GetPrimWrapperBinaryOp(primFunction);
                if (!call.Arguments.ToArray().OfType<Expr>().All(arg => TryCollectRegisterChain(arg, coveredCalls, gatherCalls, visited)))
                {
                    return false;
                }

                coveredCalls.Add(call);
                return true;
            default:
                return false;
        }
    }

    private Expr BuildRegisterScalarExpr(Expr expr, Dimension address)
    {
        if (expr is TensorConst { CheckedType: TensorType { IsScalar: true } })
        {
            return expr;
        }

        if (expr is not Call call)
        {
            throw new NotSupportedException($"Unsupported register direct affine scalar expression {expr.GetType().Name}.");
        }

        return call.Target switch
        {
            IR.Affine.Gather => T.Load((Expr)call[IR.Affine.Gather.Source], address),
            IR.Math.Unary unary => IR.F.Math.Unary(unary.UnaryOp, BuildRegisterScalarExpr((Expr)call[IR.Math.Unary.Input], address)),
            IR.Math.Binary binary => IR.F.Math.Binary(
                binary.BinaryOp,
                BuildRegisterScalarExpr((Expr)call[IR.Math.Binary.Lhs], address),
                BuildRegisterScalarExpr((Expr)call[IR.Math.Binary.Rhs], address)),
            IR.Tensors.Cast cast => IR.F.Tensors.Cast(BuildRegisterScalarExpr((Expr)call[IR.Tensors.Cast.Input], address), cast.NewType, cast.CastMode),
            IR.Tensors.Where where when !where.IsTfWhere => IR.F.Math.Select(
                BuildRegisterScalarExpr((Expr)call[IR.Tensors.Where.Cond], address),
                BuildRegisterScalarExpr((Expr)call[IR.Tensors.Where.X], address),
                BuildRegisterScalarExpr((Expr)call[IR.Tensors.Where.Y], address)),
            PrimFunctionWrapper { Target: PrimFunction primFunction } => IR.F.Math.Binary(
                GetPrimWrapperBinaryOp(primFunction),
                BuildRegisterScalarExpr((Expr)call.Arguments[0], address),
                BuildRegisterScalarExpr((Expr)call.Arguments[1], address)),
            _ => throw new NotSupportedException($"Unsupported register direct affine scalar call target {call.Target}."),
        };
    }

    private BinaryOp GetPrimWrapperBinaryOp(PrimFunction primFunction)
    {
        var vectorizedBinaryCalls = ExprCollector.Collect(primFunction.Body)
            .OfType<Call>()
            .Where(call => call.Target is TIR.NTT.VectorizedBinary)
            .ToArray();
        if (vectorizedBinaryCalls.Length != 1)
        {
            throw new NotSupportedException($"Register direct affine prim wrapper {primFunction.Name} expects exactly one VectorizedBinary op, got {vectorizedBinaryCalls.Length}.");
        }

        return ((TIR.NTT.VectorizedBinary)vectorizedBinaryCalls[0].Target).BinaryOp;
    }

    private void ValidateRegisterAffineChain(Call scatterCall, IR.Affine.Scatter scatter, IReadOnlyList<Call> gatherCalls, IReadOnlyList<Call> producerCalls, TileDecision scatterDecision)
    {
        ValidateOneDimensionalRegisterDecision(scatterDecision, "Affine.Scatter");
        foreach (var producerCall in producerCalls.Distinct(new ReferenceEqualityComparer<Call>()))
        {
            var producerDecision = TileDecisionMetadata.Require(producerCall, "Register direct affine producer/consumer layout validation");
            ValidateOneDimensionalRegisterDecision(producerDecision, producerCall.Target.GetType().Name);
            if (!IsSameDistributionLayout(producerDecision.DistributionLayout, scatterDecision.DistributionLayout))
            {
                throw new NotSupportedException(
                    "Register direct affine lowering requires producer/consumer distribution layouts to match for every tiled intermediate. " +
                    $"Producer={FormatRegisterDecision(producerCall, producerDecision)}, Consumer={FormatRegisterDecision(scatterCall, scatterDecision)}. " +
                    "Insert an explicit reshard/redistribute before consuming incompatible tile layouts.");
            }

            if (!IsSameStorageLayout(producerDecision.StorageLayout, scatterDecision.StorageLayout))
            {
                throw new NotSupportedException(
                    "Register direct affine lowering requires producer/consumer storage layouts to match for every tiled intermediate. " +
                    $"Producer={FormatRegisterDecision(producerCall, producerDecision)}, Consumer={FormatRegisterDecision(scatterCall, scatterDecision)}. " +
                    "Insert an explicit reshard/redistribute before consuming incompatible tile layouts.");
            }
        }

        foreach (var gatherCall in gatherCalls)
        {
            if (gatherCall.Target is not IR.Affine.Gather gather)
            {
                throw new InvalidOperationException("Register direct affine chain contains a non-gather input.");
            }

            var gatherDecision = TileDecisionMetadata.Require(gatherCall, "Register direct affine TIR selection");
            ValidateOneDimensionalRegisterDecision(gatherDecision, "Affine.Gather");
            if (gatherCall[IR.Affine.Gather.DefaultValue] is not None)
            {
                throw new NotSupportedException("Register direct affine lowering only supports masked gathers whose default is None and whose value is consumed under the scatter guard.");
            }

            if (!IsSameAffineRelation(gather.Relation, scatter.Relation))
            {
                throw new NotSupportedException($"Register direct affine lowering requires gather/scatter relation and symbols to match. Gather={gather.Relation}, Scatter={scatter.Relation}.");
            }

            if (!IsSameSymbolPayload(gather.Symbols, scatter.Symbols))
            {
                throw new NotSupportedException($"Register direct affine lowering requires gather/scatter symbol payloads to match. Gather={gather.Symbols}, Scatter={scatter.Symbols}.");
            }
        }

        if (scatterCall.CheckedType != TupleType.Void)
        {
            throw new NotSupportedException($"Register direct affine scatter must be void, got {scatterCall.CheckedType}.");
        }
    }

    private string FormatRegisterDecision(Call call, TileDecision decision) =>
        $"{call.Target.GetType().Name}/{decision.Id}, Op={decision.OpKind}, Shape={decision.TileShape}, Distribution={FormatDistributionLayout(decision.DistributionLayout)}, Storage={FormatStorageLayout(decision.StorageLayout)}";

    private string FormatDistributionLayout(DistributionLayout? layout) =>
        layout is null
            ? "<none>"
            : $"{layout.Kind}, LocalShape={layout.LocalShape}, OwnerDomain=[{string.Join("; ", layout.OwnerLocalToGlobal.InputDomain)}]";

    private string FormatStorageLayout(StorageLayout layout) =>
        $"{layout.Kind}, LogicalShape={layout.LogicalShape}, Map={layout.LogicalToPhysical}";

    private void ValidateOneDimensionalRegisterDecision(TileDecision decision, string opKind)
    {
        if (decision.Storage is not { Scope: BufferScope.ThreadLocal, PhysicalLocation: PhysicalMemorySpace.Register })
        {
            throw new InvalidOperationException($"{opKind} requires thread-local register tile storage, got {decision.Storage}.");
        }

        if (decision.TileShape.IsUnranked || decision.TileShape.Rank != 1 || !decision.TileShape[0].IsFixed)
        {
            throw new NotSupportedException($"{opKind} register lowering requires a fixed rank-1 tile shape, got {decision.TileShape}.");
        }
    }

    private void RequireRegisterDecision(Call call, string context)
    {
        var decision = TileDecisionMetadata.Require(call, context);
        if (decision.Storage is not { Scope: BufferScope.ThreadLocal, PhysicalLocation: PhysicalMemorySpace.Register })
        {
            throw new InvalidOperationException($"{context} requires thread-local register tile storage, got {decision.Storage}.");
        }
    }

    private BaseExpr UnwrapNestedBlock(BaseExpr body)
    {
        while (body is IRBlock nested)
        {
            body = nested.Body;
        }

        return body;
    }

    private Expr[] GetRegisterBlockFields(BaseExpr body)
    {
        return body switch
        {
            Sequential sequential => sequential.Fields.ToArray(),
            IR.Tuple tuple => tuple.Fields.ToArray().Select(field => (Expr)field).ToArray(),
            Expr expr => [expr],
            _ => throw new NotSupportedException($"Register direct affine lowering requires an expression block, got {body.GetType().Name}."),
        };
    }

    private DistributedType? GetRegisterChainDistributedType(Expr source, IReadOnlyList<Call> gatherCalls)
    {
        if (source.CheckedType is DistributedType sourceDistributed)
        {
            return sourceDistributed;
        }

        foreach (var gatherCall in gatherCalls)
        {
            if (gatherCall.CheckedType is DistributedType gatherDistributed)
            {
                return gatherDistributed;
            }
        }

        return null;
    }

    private bool IsSameAffineRelation(AffineRelation lhs, AffineRelation rhs)
    {
        return lhs.Domains.Length == rhs.Domains.Length &&
            lhs.Symbols.Length == rhs.Symbols.Length &&
            lhs.Results.Length == rhs.Results.Length &&
            lhs.Domains.ToArray().Zip(rhs.Domains.ToArray()).All(pair => pair.First.Position == pair.Second.Position) &&
            lhs.Symbols.ToArray().Zip(rhs.Symbols.ToArray()).All(pair => pair.First.Position == pair.Second.Position) &&
            lhs.Results.ToArray().Zip(rhs.Results.ToArray()).All(pair => IsSameAffineExpr(pair.First, pair.Second)) &&
            IsSameLogicalExpr(lhs.Constraint, rhs.Constraint);
    }

    private bool IsSameSymbolPayload(RankedShape lhs, RankedShape rhs)
    {
        if (lhs.Rank != rhs.Rank)
        {
            return false;
        }

        for (int i = 0; i < lhs.Rank; i++)
        {
            if (!IsSameDimension(lhs[i], rhs[i]))
            {
                return false;
            }
        }

        return true;
    }

    private bool IsSameDistributionLayout(DistributionLayout? lhs, DistributionLayout? rhs)
    {
        if (lhs is null || rhs is null)
        {
            return lhs is null && rhs is null;
        }

        return lhs.Kind == rhs.Kind &&
            lhs.LocalShape == rhs.LocalShape &&
            lhs.ValidPredicate == rhs.ValidPredicate &&
            IsSameStringArray(lhs.Attributes, rhs.Attributes) &&
            IsSameIndexMap(lhs.GlobalToOwnerLocal, rhs.GlobalToOwnerLocal) &&
            IsSameIndexMap(lhs.OwnerLocalToGlobal, rhs.OwnerLocalToGlobal);
    }

    private bool IsSameStorageLayout(StorageLayout lhs, StorageLayout rhs) =>
        lhs.Kind == rhs.Kind &&
        lhs.LogicalShape == rhs.LogicalShape &&
        lhs.ValidPredicate == rhs.ValidPredicate &&
        IsSameStringArray(lhs.Attributes, rhs.Attributes) &&
        IsSameIndexMap(lhs.LogicalToPhysical, rhs.LogicalToPhysical) &&
        IsSameNullableIndexMap(lhs.ViewMap, rhs.ViewMap);

    private bool IsSameNullableIndexMap(IndexMapDescriptor? lhs, IndexMapDescriptor? rhs)
    {
        if (lhs is null || rhs is null)
        {
            return lhs is null && rhs is null;
        }

        return IsSameIndexMap(lhs, rhs);
    }

    private bool IsSameIndexMap(IndexMapDescriptor lhs, IndexMapDescriptor rhs)
    {
        return lhs.Name == rhs.Name &&
            lhs.Predicate == rhs.Predicate &&
            lhs.Inverse == rhs.Inverse &&
            IsSameStringArray(lhs.Inputs, rhs.Inputs) &&
            IsSameStringArray(lhs.InputDomain, rhs.InputDomain) &&
            IsSameStringArray(lhs.OutputDomain, rhs.OutputDomain) &&
            lhs.Outputs.Count == rhs.Outputs.Count &&
            lhs.Outputs.ToArray().Zip(rhs.Outputs.ToArray()).All(pair =>
                pair.First.Name == pair.Second.Name && IsSameIndexExpr(pair.First.Expr, pair.Second.Expr));
    }

    private bool IsSameStringArray(IRArray<string>? lhs, IRArray<string>? rhs)
    {
        if (lhs is null || rhs is null)
        {
            return lhs is null && rhs is null;
        }

        return lhs.Value.SequenceEqual(rhs.Value);
    }

    private bool IsSameIndexExpr(IndexExpr lhs, IndexExpr rhs)
    {
        if (lhs.GetType() != rhs.GetType())
        {
            return false;
        }

        return (lhs, rhs) switch
        {
            (IndexVar l, IndexVar r) => l.Name == r.Name,
            (IndexConst l, IndexConst r) => l.Value == r.Value,
            (IndexAny, IndexAny) => true,
            (IndexAdd l, IndexAdd r) => l.Terms.Count == r.Terms.Count &&
                l.Terms.ToArray().Zip(r.Terms.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            (IndexMul l, IndexMul r) => l.Factors.Count == r.Factors.Count &&
                l.Factors.ToArray().Zip(r.Factors.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            (IndexFloorDiv l, IndexFloorDiv r) => IsSameIndexExpr(l.Value, r.Value) && IsSameIndexExpr(l.Divisor, r.Divisor),
            (IndexMod l, IndexMod r) => IsSameIndexExpr(l.Value, r.Value) && IsSameIndexExpr(l.Divisor, r.Divisor),
            (IndexNamedPrimitive l, IndexNamedPrimitive r) => l.Name == r.Name &&
                l.Arguments.Count == r.Arguments.Count &&
                l.Arguments.ToArray().Zip(r.Arguments.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            _ => false,
        };
    }

    private bool IsSameAffineExpr(AffineExpr lhs, AffineExpr rhs)
    {
        if (lhs.GetType() != rhs.GetType())
        {
            return false;
        }

        return (lhs, rhs) switch
        {
            (AffineConstant l, AffineConstant r) => l.Value == r.Value,
            (AffineDim l, AffineDim r) => l.Position == r.Position,
            (AffineExtent l, AffineExtent r) => l.Position == r.Position,
            (AffineSymbol l, AffineSymbol r) => l.Position == r.Position,
            (AffineAddBinary l, AffineAddBinary r) => IsSameAffineExpr(l.Lhs, r.Lhs) && IsSameAffineExpr(l.Rhs, r.Rhs),
            (AffineMulBinary l, AffineMulBinary r) => IsSameAffineExpr(l.Lhs, r.Lhs) && IsSameAffineExpr(l.Rhs, r.Rhs),
            (AffineDivBinary l, AffineDivBinary r) => l.BinaryOp == r.BinaryOp && IsSameAffineExpr(l.Lhs, r.Lhs) && IsSameAffineExpr(l.Rhs, r.Rhs),
            _ => false,
        };
    }

    private bool IsSameLogicalExpr(LogicalExpr lhs, LogicalExpr rhs)
    {
        if (lhs.GetType() != rhs.GetType())
        {
            return false;
        }

        return (lhs, rhs) switch
        {
            (LogicalConst l, LogicalConst r) => l.Value == r.Value,
            (DimCompare l, DimCompare r) => l.Op == r.Op && IsSameDimension(l.Lhs, r.Lhs) && IsSameDimension(l.Rhs, r.Rhs),
            (LogicalAnd l, LogicalAnd r) => l.Count == r.Count && l.Operands.ToArray().Zip(r.Operands.ToArray()).All(pair => IsSameLogicalExpr(pair.First, pair.Second)),
            (LogicalOr l, LogicalOr r) => l.Count == r.Count && l.Operands.ToArray().Zip(r.Operands.ToArray()).All(pair => IsSameLogicalExpr(pair.First, pair.Second)),
            _ => false,
        };
    }

    private bool IsSameDimension(Dimension lhs, Dimension rhs)
    {
        if (ReferenceEquals(lhs, rhs))
        {
            return true;
        }

        if (lhs.GetType() != rhs.GetType())
        {
            return false;
        }

        return (lhs, rhs) switch
        {
            (DimConst l, DimConst r) => l.Value == r.Value,
            (DimVar l, DimVar r) => l.Name == r.Name,
            (ThreadIdDim, ThreadIdDim) => true,
            (ProgramIdDim l, ProgramIdDim r) => l.Axis == r.Axis,
            (DimSum l, DimSum r) => l.Bias == r.Bias &&
                l.Operands.Length == r.Operands.Length &&
                l.Operands.ToArray().Zip(r.Operands.ToArray()).All(pair => IsSameDimension(pair.First, pair.Second)),
            (DimProduct l, DimProduct r) => l.Scale == r.Scale &&
                l.Operands.Length == r.Operands.Length &&
                l.Operands.ToArray().Zip(r.Operands.ToArray()).All(pair => IsSameDimension(pair.First, pair.Second)),
            (DimFraction l, DimFraction r) => l.DivMode == r.DivMode &&
                IsSameDimension(l.Numerator, r.Numerator) &&
                IsSameDimension(l.Denominator, r.Denominator),
            (DimRemainder l, DimRemainder r) => IsSameDimension(l.Numerator, r.Numerator) &&
                IsSameDimension(l.Denominator, r.Denominator),
            _ => false,
        };
    }

    private Dimension GetRegisterDomainValue(TileDecision decision, DistributedType? distributedType, DimVar lane, Dimension globalExtent)
    {
        if (decision.DistributionLayout is { Kind: "TritonBlocked" } tritonBlockedLayout)
        {
            return AffineIOLayoutEvaluator.GetTritonBlockedLocalDomainValue(tritonBlockedLayout, lane, "Register direct affine lowering");
        }

        if (decision.DistributionLayout is { Kind: not "SBP" } unsupportedLayout)
        {
            throw new NotSupportedException($"Register direct affine lowering cannot evaluate {unsupportedLayout.Kind} layouts. Add a DistributionLayout owner-local evaluator for this layout kind.");
        }

        if (distributedType is null)
        {
            return lane;
        }

        LayoutVerifier.Verify(distributedType, "Register direct affine TIR selection");

        if (distributedType.ExplicitDistributionLayout is not null && distributedType.DistributionLayout.Kind != "SBP")
        {
            throw new NotSupportedException($"Register direct affine lowering cannot yet evaluate explicit {distributedType.DistributionLayout.Kind} layouts. Add a DistributionLayout map evaluator instead of falling back to AxisPolicies.");
        }

        if (distributedType.Partial)
        {
            throw new NotSupportedException("Register direct affine lowering cannot lower partial distributed buffers. Resolve Partial before affine IO lowering.");
        }

        if (distributedType.AxisPolicies.Count != 1)
        {
            throw new NotSupportedException($"Register direct affine lowering currently supports rank-1 distribution only, got {distributedType.AxisPolicies.Count} axis policies.");
        }

        return distributedType.AxisPolicies[0] switch
        {
            SBPBroadCast => lane,
            SBPSplit split => lane + GetSplitShardOffset(distributedType, split, globalExtent),
            SBPPartial partial => throw new NotSupportedException($"Register direct affine lowering cannot directly lower partial shard policy {partial}."),
            SBP policy => throw new NotSupportedException($"Unsupported register direct affine shard policy {policy.GetType().Name}."),
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
            var name => throw new NotSupportedException($"Register direct affine lowering only supports thread ('t') and block ('b') mesh axes, got '{name}' in placement {placement}."),
        };
    }

    private Dimension EvaluateAddress(AffineRelation relation, IReadOnlyList<Dimension> domainValues, IReadOnlyList<Dimension> extents, IReadOnlyDictionary<int, Dimension>? symbolMap)
    {
        if (relation.Results.Length != 1)
        {
            throw new NotSupportedException($"Register direct affine lowering expects one address result, got {relation.Results.Length}.");
        }

        return EvaluateAffineExpr(relation.Results[0], domainValues, extents, symbolMap);
    }

    private LogicalExpr EvaluateConstraint(LogicalExpr constraint, IReadOnlyList<Dimension> domainValues)
    {
        return constraint switch
        {
            LogicalConst logicalConst => logicalConst,
            DimCompare compare => new DimCompare(compare.Op, EvaluateDimension(compare.Lhs, domainValues), EvaluateDimension(compare.Rhs, domainValues)),
            LogicalAnd logicalAnd => new LogicalAnd(logicalAnd.Operands.ToArray().Select(x => EvaluateConstraint(x, domainValues)).ToArray()),
            LogicalOr logicalOr => new LogicalOr(logicalOr.Operands.ToArray().Select(x => EvaluateConstraint(x, domainValues)).ToArray()),
            _ => throw new NotSupportedException($"Unsupported register direct affine constraint node {constraint.GetType().Name}."),
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
            _ => throw new NotSupportedException($"Unsupported register direct affine constraint dimension {dim.GetType().Name}."),
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
            AffineSymbol symbol => symbols is null ? throw new NotSupportedException("Symbolic register direct affine relations require a bound symbol map.") : symbols[symbol.Position],
            AffineAddBinary add => EvaluateAffineExpr(add.Lhs, dims, extents, symbols) + EvaluateAffineExpr(add.Rhs, dims, extents, symbols),
            AffineMulBinary mul => EvaluateAffineExpr(mul.Lhs, dims, extents, symbols) * EvaluateAffineExpr(mul.Rhs, dims, extents, symbols),
            AffineDivBinary div => ApplyDivBinary(div.BinaryOp, EvaluateAffineExpr(div.Lhs, dims, extents, symbols), EvaluateAffineExpr(div.Rhs, dims, extents, symbols)),
            _ => throw new NotSupportedException($"Unsupported register direct affine expression node {expr.GetType().Name}"),
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

    private Expr GenerateReshape(Expr input, ref Expr output, bool sequeeze = false)
    {
        if (input is not TIR.Buffer inBuffer)
        {
            throw new NotSupportedException("Reshape only support buffer input");
        }

        var outBuffer = (TIR.Buffer)output;

        var bitcast = outBuffer.DistributedType is DistributedType ? false : true;
        if (!bitcast)
        {
            var inDistributedType = inBuffer.DistributedType ?? throw new NotSupportedException("Reshape distributed output requires distributed buffer input.");
            var outDistributedType = outBuffer.DistributedType!;
            ValidateDistributedViewLayoutForReshape(inBuffer, outBuffer, inDistributedType, outDistributedType);
            if (inDistributedType.AxisPolicies.Where(sbp => sbp is not SBPBroadCast).ToArray().SequenceEqual(outDistributedType.AxisPolicies.Where(sbp => sbp is not SBPBroadCast).ToArray()))
            {
                bitcast = true;
            }
        }

        // If the size is not same, we cannot bitcast.
        if ((inBuffer.MemSpan.Size == outBuffer.MemSpan.Size) && (bitcast || sequeeze))
        {
            output = inBuffer.With(name: outBuffer.Name, elemType: outBuffer.ElemType, dimensions: outBuffer.Dimensions.ToArray(), strides: outBuffer.Strides.ToArray(), distributedType: outBuffer.DistributedType);
            return T.Nop();
        }
        else
        {
            return TIR.F.NTT.Reshape(input, output);
        }
    }

    private Expr GenerateBitcast(Expr input, ref Expr output, DataType newType)
    {
        if (input is not TIR.Buffer inBuffer)
        {
            throw new NotSupportedException("Bitcast only support buffer input");
        }

        var srcSize = inBuffer.ElemType.SizeInBytes;
        var destSize = newType.SizeInBytes;
        var newDimensions = inBuffer.Dimensions.ToArray();
        var newStrides = inBuffer.Strides.ToArray();

        if (srcSize != destSize)
        {
            if (newDimensions.Rank == 0)
            {
                newDimensions = [srcSize / destSize];
                newStrides = [1];
            }
            else
            {
                newDimensions[^1] = newDimensions[^1] * srcSize / destSize;
                if (newStrides.Length > 1)
                {
                    newStrides[^1] = 1;
                    for (var i = 0; i < newStrides.Length - 1; i++)
                    {
                        newStrides[i] = newStrides[i] * srcSize / destSize;
                    }
                }
            }
        }

        var distributedType = inBuffer.DistributedType is DistributedType dt
            ? dt with { TensorType = new TensorType(newType, newDimensions) }
            : null;
        output = inBuffer.With(name: ((TIR.Buffer)output).Name, elemType: newType, dimensions: newDimensions, strides: newStrides, distributedType: distributedType);
        return T.Nop();
    }

    private Expr GenerateUnary(UnaryOp unaryOp, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var input = (Expr)arguments[IR.Math.Unary.Input.Index];
        return TIR.F.NTT.Unary(unaryOp, input, output);
    }

    private Expr GenerateClamp(Call call, IReadOnlyList<BaseExpr> arguments, Expr output)
    {
        var min = ((TensorConst)call[IR.Math.Clamp.Min]).Value.ToScalar<float>();
        var max = ((TensorConst)call[IR.Math.Clamp.Max]).Value.ToScalar<float>();
        return TIR.F.NTT.Clamp((Expr)arguments[0], output, min, max);
    }

    private Expr GenerateBoxing(Call call, IR.Distributed.Boxing boxing, IReadOnlyList<BaseExpr> arguments, ref Expr output)
    {
        switch (call[IR.Distributed.Boxing.Input].CheckedType, boxing.NewType)
        {
            case (TensorType, DistributedType distTensorType):
                ValidateLegacyAxisPolicySelection(distTensorType, "TensorLoad output");
                return TIR.F.NTT.TensorLoad(output, (Expr)arguments[0], distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType distTensorType, TensorType):
                ValidateLegacyAxisPolicySelection(distTensorType, "TensorStore input");
                return TIR.F.NTT.TensorStore((Expr)arguments[0], output, distTensorType.AxisPolicies, distTensorType.Placement);
            case (DistributedType inType, DistributedType outType):
                return GenerateReshard((Expr)arguments[0], ref output, inType, outType);
            default:
                throw new NotSupportedException();
        }
    }

    private Expr GenerateReshard(Expr input, ref Expr output, DistributedType inType, DistributedType outType)
    {
        // FIXME: re-balance issue.
#if false
        if (input is TIR.Buffer inBuffer)
        {
            if (TryGenerateGatherThreadsReshard(inBuffer, ref output, inType, outType, out var newCall))
            {
                return newCall;
            }
            else if (TryGenerateSplitThreadsReshard(inBuffer, ref output, inType, outType, out newCall))
            {
                return newCall;
            }
        }
#endif

        return TIR.F.NTT.GatherReduceScatter(input, output, inType, outType);
    }

    private void ValidateDistributedViewLayoutForReshape(TIR.Buffer input, TIR.Buffer output, DistributedType inputType, DistributedType outputType)
    {
        if (!inputType.HasExplicitLayout && !outputType.HasExplicitLayout)
        {
            return;
        }

        LayoutVerifier.VerifyEquivalentForView(
            inputType,
            outputType,
            $"NTT GenerateReshape bitcast {input.Name}->{output.Name}, InputShape={FormatBufferShape(input)}, OutputShape={FormatBufferShape(output)}");
    }

    private string FormatBufferShape(TIR.Buffer buffer) =>
        $"[{string.Join(",", buffer.Dimensions.ToArray().Select(dimension => dimension.ToString()))}]";

    private void ValidateLegacyAxisPolicySelection(Expr expr, string context)
    {
        if (expr is TIR.Buffer { DistributedType: { } distributedType })
        {
            ValidateLegacyAxisPolicySelection(distributedType, context);
        }
        else if (expr.CheckedType is DistributedType checkedDistributedType)
        {
            ValidateLegacyAxisPolicySelection(checkedDistributedType, context);
        }
    }

    private void ValidateLegacyAxisPolicySelection(DistributedType distributedType, string context)
    {
        LayoutVerifier.VerifyEquivalentToLegacyAxisPolicies(
            distributedType,
            $"NTT TIR selection {context}");
    }

    private bool TryGenerateGatherThreadsReshard(TIR.Buffer inBuffer, ref Expr output, DistributedType inType, DistributedType outType, [MaybeNullWhen(false)] out Expr newCall)
    {
        var threadAxis = inType.Placement.Rank - 1;
        PhysicalBuffer? oldPhysicalBuffer = null;

        // S -> B
        var reducedInPolices = inType.AxisPolicies.Select(sbp => sbp is SBPSplit split && split.Axes.Contains(threadAxis) ? (split.Axes.Count == 1 ? (SBP)SBP.B : SBP.S(split.Axes.Except([threadAxis]).ToArray())) : sbp);
        if (reducedInPolices.ToArray().SequenceEqual(outType.AxisPolicies.ToArray()))
        {
            oldPhysicalBuffer = inBuffer.MemSpan.Buffer;
        }

        var oldOutputBuffer = (TIR.Buffer)output;
        if (oldPhysicalBuffer is not null)
        {
            var threads = inType.Placement.Hierarchy[threadAxis];
            var newPhysicalBuffer = oldPhysicalBuffer.With(
                size: oldOutputBuffer.MemSpan.Size,
                location: MemoryLocation.BlockLocalData);

            // 1. Replace all uses of old buffer to new buffer.
            var userBuffers = (from memSpan in oldPhysicalBuffer.Users.OfType<TIR.MemSpan>()
                               from userBuffer in memSpan.Users.OfType<TIR.Buffer>()
                               select userBuffer).ToArray();

            foreach (var userBuffer in userBuffers)
            {
                var userType = userBuffer.DistributedType!;
                var threadAxisDim = userType.AxisPolicies.ToArray().IndexOf(x => x is SBPSplit split && split.Axes.Contains(threadAxis));
                if (threadAxisDim >= 0)
                {
                    var newStrides = userBuffer.Strides.ToArray();
                    for (int i = 0; i < userType.AxisPolicies.Count; i++)
                    {
                        if (i < threadAxisDim)
                        {
                            newStrides[i] *= threads;
                        }

                        if (i == threadAxisDim)
                        {
                            var newStart = userBuffer.MemSpan.Start;
                            if (newStart != Dimension.Zero)
                            {
                                // We don't support sliced buffer.
                                newCall = null;
                                return false;
                            }

                            var dividedType = DistributedUtility.GetDividedTensorType(userType);
                            newStart = dividedType.Shape[i] * newStrides[i] * userBuffer.CheckedDataType.SizeInBytes * IR.F.Distributed.ThreadId();
                            var newSize = TensorUtilities.GetMaxSize(dividedType.Shape, newStrides.Select(x => x.FixedValue).ToArray(), userBuffer.CheckedDataType.SizeInBytes);
                            var newBuffer = userBuffer.With(
                                memSpan: userBuffer.MemSpan.With(
                                    buffer: newPhysicalBuffer,
                                    start: newStart,
                                    size: newSize),
                                strides: newStrides);
                            ReplaceUtility.ReplaceAllUsesWith(userBuffer, newBuffer);
                            break;
                        }
                    }
                }
            }

            // 2. Create new output buffer.
            output = oldOutputBuffer.With(
                memSpan: oldOutputBuffer.MemSpan.With(buffer: newPhysicalBuffer));
            newCall = TIR.F.NTT.SynchronizeThreads();
            return true;
        }

        newCall = null;
        return false;
    }

    private bool TryGenerateSplitThreadsReshard(TIR.Buffer inBuffer, ref Expr output, DistributedType inType, DistributedType outType, [MaybeNullWhen(false)] out Expr newCall)
    {
        // Avoid P -> B -> S
        if (inType.Partial)
        {
            newCall = null;
            return false;
        }

        var threadAxis = inType.Placement.Rank - 1;
        PhysicalBuffer? oldPhysicalBuffer = null;
        var reducedOutPolices = outType.AxisPolicies.Select(sbp => sbp is SBPSplit split && split.Axes.Contains(threadAxis) ? (split.Axes.Count == 1 ? (SBP)SBP.B : SBP.S(split.Axes.Except([threadAxis]).ToArray())) : sbp);
        int splitAxis = -1;

        // B -> S
        if (reducedOutPolices.ToArray().SequenceEqual(inType.AxisPolicies.ToArray()))
        {
            oldPhysicalBuffer = inBuffer.MemSpan.Buffer;
            splitAxis = outType.AxisPolicies.ToArray().IndexOf(x => x is SBPSplit split && split.Axes.Contains(threadAxis));
        }

        var oldOutputBuffer = (TIR.Buffer)output;
        if (oldPhysicalBuffer is not null)
        {
            var threads = inType.Placement.Hierarchy[threadAxis];
            var newPhysicalBuffer = oldPhysicalBuffer.With(
                location: MemoryLocation.BlockLocalData);

            // 1. Replace all uses of old mem span to new mem span.
            var userMemSpans = oldPhysicalBuffer.Users.OfType<TIR.MemSpan>().ToArray();

            foreach (var userMemSpan in userMemSpans)
            {
                var newMemSpan = userMemSpan.With(buffer: newPhysicalBuffer);
                ReplaceUtility.ReplaceAllUsesWith(userMemSpan, newMemSpan);
            }

            // 2. Create new output buffer.
            var newStart = oldOutputBuffer.MemSpan.Start;
            if (newStart != Dimension.Zero)
            {
                // We don't support sliced buffer.
                newCall = null;
                return false;
            }

            var dividedType = DistributedUtility.GetDividedTensorType(outType);
            newStart = dividedType.Shape[splitAxis] * oldOutputBuffer.Strides[splitAxis] * oldOutputBuffer.CheckedDataType.SizeInBytes * IR.F.Distributed.ThreadId();

            output = oldOutputBuffer.With(
                memSpan: oldOutputBuffer.MemSpan.With(buffer: newPhysicalBuffer, start: newStart), strides: inBuffer.Strides.ToArray());
            newCall = TIR.F.NTT.SynchronizeThreads();
            return true;
        }

        newCall = null;
        return false;
    }

    private sealed record RegisterScatterLowering(
        Call ScatterCall,
        IR.Affine.Scatter Scatter,
        TileDecision Decision,
        Expr Source,
        IReadOnlyList<Call> GatherCalls,
        IReadOnlyList<Call> CoveredCalls);
}
