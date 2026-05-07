// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR.NTT;

namespace Nncase.Evaluator.TIR.NTT;

public class TensorLoadEvaluator : ITypeInferencer<TensorLoad>
{
    public IRType Visit(ITypeInferenceContext context, TensorLoad target)
    {
        var destType = context.GetArgumentType(target, TensorLoad.Dest);
        if (destType is not TensorType and not DistributedType)
        {
            throw new TypeInferenceInterruptException(new InvalidType($"TensorLoad.dest must be TensorType or DistributedType, got {destType}."));
        }

        _ = context.CheckArgumentType<IRType>(target, TensorLoad.Src);
        return TupleType.Void;
    }
}
