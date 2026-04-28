// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public sealed class AffineGatherEvaluator : ITypeInferencer<AffineGather>
{
    public IRType Visit(ITypeInferenceContext context, AffineGather target)
    {
        _ = context.CheckArgumentType<IRType>(target, AffineGather.Source);
        _ = context.CheckArgumentType<IRType>(target, AffineGather.DefaultValue);
        _ = context.CheckArgumentType<TensorType>(target, AffineGather.Output);
        return TupleType.Void;
    }
}

public sealed class AffineScatterEvaluator : ITypeInferencer<AffineScatter>
{
    public IRType Visit(ITypeInferenceContext context, AffineScatter target)
    {
        _ = context.CheckArgumentType<TensorType>(target, AffineScatter.Source);
        _ = context.CheckArgumentType<IRType>(target, AffineScatter.Dest);
        return TupleType.Void;
    }
}
