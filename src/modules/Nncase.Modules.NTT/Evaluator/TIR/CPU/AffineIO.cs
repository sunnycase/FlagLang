// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class AffineGatherEvaluator : ITypeInferencer<AffineGather>, IKernelInfoEvaluator<AffineGather>
{
    public IRType Visit(ITypeInferenceContext context, AffineGather target)
    {
        _ = context.CheckArgumentType<IRType>(target, AffineGather.Source);
        _ = context.CheckArgumentType<IRType>(target, AffineGather.DefaultValue);
        _ = context.CheckArgumentType<TensorType>(target, AffineGather.Output);
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(AffineGather op, MicroKernelContext context)
    {
        var bufferInfos = AffineIOKernelInfo.CreateBufferInfos(context, MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(AffineIOKernelInfo.CreateTileBounds(context), bufferInfos, AffineIOKernelInfo.GetComputeCycle);
    }
}

public sealed class AffineScatterEvaluator : ITypeInferencer<AffineScatter>, IKernelInfoEvaluator<AffineScatter>
{
    public IRType Visit(ITypeInferenceContext context, AffineScatter target)
    {
        _ = context.CheckArgumentType<TensorType>(target, AffineScatter.Source);
        _ = context.CheckArgumentType<IRType>(target, AffineScatter.Dest);
        return TupleType.Void;
    }

    public MicroKernelInfo Visit(AffineScatter op, MicroKernelContext context)
    {
        var bufferInfos = AffineIOKernelInfo.CreateBufferInfos(context, MicroKernelBufferInfo.BufferState.Write);
        return new MicroKernelInfo(AffineIOKernelInfo.CreateTileBounds(context), bufferInfos, AffineIOKernelInfo.GetComputeCycle);
    }
}

internal static class AffineIOKernelInfo
{
    public static ValueRange<long>[] CreateTileBounds(MicroKernelContext context)
    {
        var domain = context.AccessMaps[0].Domains;
        return Enumerable.Repeat(new ValueRange<long>(1, int.MaxValue), domain.Length).ToArray();
    }

    public static MicroKernelBufferInfo[] CreateBufferInfos(MicroKernelContext context, MicroKernelBufferInfo.BufferState lastBufferState)
    {
        var opt = (INTTTargetOptions)context.TargetOptions;
        var bufferInfos = new MicroKernelBufferInfo[context.BufferShapes.Length];
        for (int i = 0; i < bufferInfos.Length; i++)
        {
            var state = i == bufferInfos.Length - 1 ? lastBufferState : MicroKernelBufferInfo.BufferState.Read;
            bufferInfos[i] = new(opt.MemoryBandWidths[1], opt.MemoryBandWidths[1], state);
        }

        return bufferInfos;
    }

    public static Google.OrTools.ConstraintSolver.IntExpr GetComputeCycle(Google.OrTools.ConstraintSolver.IntExpr[][] bufferShapes, Google.OrTools.ConstraintSolver.Solver solver, MicroKernelContext context)
    {
        return solver.MakeIntConst(1);
    }
}
