// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Triton;

namespace Nncase.Evaluator.Triton;

/// <summary>
/// Evaluator for <see cref="Store"/>.
/// </summary>
public class StoreEvaluator : ITypeInferencer<Store>, IOpPrinter<Store>
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
