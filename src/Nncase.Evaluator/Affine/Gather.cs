// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Evaluator.Affine;

/// <summary>
/// Evaluator for <see cref="Gather"/>.
/// </summary>
[TypeInferGenerator]
public partial class GatherEvaluator : ITypeInferencer<Gather>, IOpPrinter<Gather>
{
    public string Visit(IPrintOpContext context, Gather target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"{context.GetArgument(target, Gather.Source)}";
        }

        return context.GetDefault(target);
    }

    private IRType Visit(Gather target, TensorType source)
    {
        if (source.DType is not PointerType pointerType)
        {
            return new InvalidType("source is not pointer type!");
        }

        return new TensorType(pointerType.ElemType, target.Shape);
    }
}
