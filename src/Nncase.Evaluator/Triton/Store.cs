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
/// Evaluator for <see cref="Store"/>.
/// </summary>
public class StoreEvaluator : ITypeInferencer<Store>, IOpPrinter<Store>, IEvaluator<Store>, ICostEvaluator<Store>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Store target)
    {
        var ptr = context.CheckArgumentType<TensorType>(target, Store.Ptr);
        var value = context.CheckArgumentType<TensorType>(target, Store.Value);
        return Visit(target, ptr, value);
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Store target)
    {
        var ptr = context.GetArgument(target, Store.Ptr);
        var value = context.GetArgument(target, Store.Value);
        return $"*({ptr}) = {value}";
    }

    public unsafe IValue Visit(IEvaluateContext context, Store target)
    {
        var ptrTensor = context.GetArgumentValue(target, Store.Ptr).AsTensor();
        var valueTensor = context.GetArgumentValue(target, Store.Value).AsTensor();
        var maskValue = context.GetArgumentValue(target, Store.Mask);
        if (ptrTensor.ElementType is not PointerType pointerType)
        {
            throw new InvalidOperationException("Store ptr must be pointer type");
        }

        var elemSize = pointerType.ElemType.SizeInBytes;
        var source = valueTensor.CastElementTo(pointerType.ElemType);
        Tensor<bool>? maskTensor = maskValue is NoneValue ? null : maskValue.AsTensor().Cast<bool>();

        if (source.Length != ptrTensor.Length)
        {
            throw new InvalidOperationException("Store ptr/value length mismatch");
        }

        for (long idx = 0; idx < source.Length; idx++)
        {
            if (maskTensor is not null && !maskTensor.GetValue(idx))
            {
                continue;
            }

            var pointerSpan = ptrTensor.BytesBuffer.Slice((int)(idx * sizeof(ulong)), sizeof(ulong));
            var address = MemoryMarshal.Read<ulong>(pointerSpan);
            if (address == 0)
            {
                throw new InvalidOperationException("Pointer value is zero");
            }

            var dest = new Span<byte>((void*)address, elemSize);
            var src = source.BytesBuffer.Slice((int)(idx * elemSize), elemSize);
            src.CopyTo(dest);
        }

        return Value.None;
    }

    public Cost Visit(ICostEvaluateContext context, Store target)
    {
        var valueType = context.GetArgumentType<TensorType>(target, Store.Value);
        var bytes = CostUtility.GetMemoryAccess(valueType);
        return new()
        {
            [CostFactorNames.MemoryLoad] = bytes,
            [CostFactorNames.MemoryStore] = bytes,
            [CostFactorNames.CPUCycles] = bytes,
        };
    }

    private IRType Visit(Store target, TensorType ptr, TensorType value)
    {
        if (ptr.DType is not PointerType { ElemType: DataType elemType }
            || elemType != value.DType
            || ptr.Shape != value.Shape)
        {
            return new InvalidType($"Cannot store {value.DType} to {ptr.DType}");
        }

        return TupleType.Void;
    }
}
