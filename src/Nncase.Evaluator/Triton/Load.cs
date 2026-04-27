// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Runtime.InteropServices;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Triton;

namespace Nncase.Evaluator.Triton;

/// <summary>
/// Evaluator for <see cref="Load"/>.
/// </summary>
[TypeInferGenerator]
public partial class LoadEvaluator : ITypeInferencer<Load>, IOpPrinter<Load>, IEvaluator<Load>, ICostEvaluator<Load>
{
    public string Visit(IPrintOpContext context, Load target)
    {
        if (context.Flags.HasFlag(PrinterFlags.Inline) || context.Flags.HasFlag(PrinterFlags.Script))
        {
            return $"{context.GetArgument(target, Load.Ptr)}";
        }

        return context.GetDefault(target);
    }

    public unsafe IValue Visit(IEvaluateContext context, Load target)
    {
        var ptrTensor = context.GetArgumentValue(target, Load.Ptr).AsTensor();
        var other = context.GetArgumentValue(target, Load.Other).AsTensor();
        var maskValue = context.GetArgumentValue(target, Load.Mask);
        if (ptrTensor.ElementType is not PointerType pointerType)
        {
            throw new InvalidOperationException("Load ptr must be pointer type");
        }

        if (context.GetReturnType() is not TensorType resultType)
        {
            throw new InvalidOperationException("Load return type must be tensor");
        }

        var destination = other.CastElementTo(resultType.DType);
        Tensor<bool>? maskTensor = maskValue is NoneValue ? null : maskValue.AsTensor().Cast<bool>();
        var elemSize = pointerType.ElemType.SizeInBytes;
        for (long idx = 0; idx < destination.Length; idx++)
        {
            if (maskTensor is not null && !maskTensor.GetValue(idx))
            {
                continue;
            }

            var pointerValue = ptrTensor.BytesBuffer.Slice((int)(idx * sizeof(ulong)), sizeof(ulong));
            var address = MemoryMarshal.Read<ulong>(pointerValue);
            var span = new Span<byte>((void*)address, elemSize);
            span.CopyTo(destination.BytesBuffer.Slice((int)(idx * elemSize), elemSize));
        }

        return Value.FromTensor(destination);
    }

    public Cost Visit(ICostEvaluateContext context, Load target)
    {
        var resultType = context.GetReturnType<TensorType>();
        var bytes = CostUtility.GetMemoryAccess(resultType);
        return new()
        {
            [CostFactorNames.MemoryLoad] = bytes,
            [CostFactorNames.MemoryStore] = 0,
            [CostFactorNames.CPUCycles] = bytes,
        };
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
