// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Buffers;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for BufferOf.
/// </summary>
public partial class BufferLoadEvaluator : ITypeInferencer<BufferLoad>, IOpPrinter<BufferLoad>
{
    public IRType Visit(ITypeInferenceContext context, BufferLoad target)
    {
        var input = context.CheckArgumentType<IRType>(target, BufferLoad.Input);
        var indices = context.CheckArgumentType<TupleType>(target, BufferLoad.Indices);
        return input switch
        {
            TensorType tensorType => Visit(tensorType, indices),
            DistributedType distributedType => Visit(distributedType, indices),
            _ => new InvalidType($"BufferLoad input must be TensorType or DistributedType, got {input}."),
        };
    }

    public string Visit(IPrintOpContext context, BufferLoad target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"{context.GetArgument(target, BufferLoad.Input)}[{context.GetArgument(target, BufferLoad.Indices)}]";
        }

        return context.GetDefault(target);
    }

    private IRType Visit(TensorType input, TupleType indices)
    {
        if (indices.Count != input.Shape.Rank)
        {
            return new InvalidType($"the input buffer rank {input.Shape.Rank} != indices.Count {indices.Count}");
        }

        foreach (var item in indices)
        {
            if (item is not TensorType { IsScalar: true, DType: var dtype } || dtype != DataTypes.Int32)
            {
                return new InvalidType("indices is not int32 type!");
            }
        }

        return TensorType.Scalar(input.DType);
    }

    private IRType Visit(DistributedType input, TupleType indices)
    {
        return Visit(input.TensorType, indices);
    }
}
