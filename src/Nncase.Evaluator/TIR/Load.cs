// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Evaluator.TIR;

/// <summary>
/// Evaluator for <see cref="Load"/>.
/// </summary>
public class LoadEvaluator : ITypeInferencer<Load>, IOpPrinter<Load>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Load target)
    {
        var handle = context.CheckArgumentType<TensorType>(target, Load.Handle);
        var index = context.GetArgumentType(target, Load.Index);
        return Visit(target, handle, index);
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Load target)
    {
        var lhs = context.GetArgument(target, Load.Handle);
        var rhs = context.GetArgument(target, Load.Index);
        return $"{lhs}[{rhs}]";
    }

    private IRType Visit(Load target, TensorType handle, IRType index)
    {
        if (handle is not TensorType { DType: PointerType { } p })
        {
            return new InvalidType("handle must be pointer type!");
        }

        var validIndex = index is DimensionType
            || (index is TensorType { Shape.IsScalar: true, DType: var indexDtype } && indexDtype.IsIntegral());
        if (!validIndex)
        {
            return new InvalidType("load index must be a dimension or scalar integral tensor.");
        }

        return TensorType.Scalar(p.ElemType);
    }
}
