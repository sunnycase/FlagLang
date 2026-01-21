// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Triton;

namespace Nncase.Evaluator.Triton;

/// <summary>
/// Evaluator for <see cref="Load"/>.
/// </summary>
[TypeInferGenerator]
public partial class LoadEvaluator : ITypeInferencer<Load>, IOpPrinter<Load>
{
    public string Visit(IPrintOpContext context, Load target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"{context.GetArgument(target, Load.Ptr)}";
        }

        return context.GetDefault(target);
    }

    private IRType Visit(TensorType ptr)
    {
        if (ptr.DType is not PointerType pointerType)
        {
            return new InvalidType("ptr is not pointer type!");
        }

        return ptr with { DType = pointerType.ElemType };
    }
}
