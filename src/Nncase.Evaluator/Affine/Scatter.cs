// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Evaluator.Affine;

/// <summary>
/// Evaluator for <see cref="Scatter"/>.
/// </summary>
[TypeInferGenerator]
public partial class ScatterEvaluator : ITypeInferencer<Scatter>, IOpPrinter<Scatter>, ICostEvaluator<Scatter>
{
    public string Visit(IPrintOpContext context, Scatter target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"{context.GetArgument(target, Scatter.Source)}";
        }

        return context.GetDefault(target);
    }

    public Cost Visit(ICostEvaluateContext context, Scatter target)
    {
        var sourceType = context.GetArgumentType<TensorType>(target, Scatter.Source);
        var bytes = CostUtility.GetMemoryAccess(sourceType);
        return new()
        {
            [CostFactorNames.MemoryLoad] = bytes,
            [CostFactorNames.MemoryStore] = bytes,
            [CostFactorNames.CPUCycles] = bytes,
        };
    }

    private IRType Visit(Scatter target, TensorType source, TensorType dest)
    {
        if (dest.DType is not PointerType)
        {
            return new InvalidType("dest is not pointer type!");
        }

        return TupleType.Void;
    }
}
