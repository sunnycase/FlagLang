// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// buffer.
/// </summary>
public sealed class Buffer : Expr
{
    public Buffer(string name, IRType type, MemSpan memSpan, Dimension[] dimensions, Dimension[] strides)
        : base(new BaseExpr[] { memSpan }.Concat(dimensions).Concat(strides))
    {
        var tensorType = GetTensorType(type, name);
        if (tensorType.Shape is not RankedShape rankedShape)
        {
            throw new InvalidOperationException($"TIR buffer {name} requires a ranked tensor type, got {tensorType.Shape}.");
        }

        if (rankedShape.Rank != dimensions.Length)
        {
            throw new InvalidOperationException($"TIR buffer {name} type rank {rankedShape.Rank} does not match buffer rank {dimensions.Length}.");
        }

        if (strides.Length != dimensions.Length)
        {
            throw new InvalidOperationException($"TIR buffer {name} stride rank {strides.Length} does not match buffer rank {dimensions.Length}.");
        }

        Name = name;
        Type = type;
        TensorType = tensorType;
        Rank = dimensions.Length;
    }

    public string Name { get; }

    public IRType Type { get; }

    public TensorType TensorType { get; }

    public DataType ElemType => TensorType.DType;

    /// <summary>
    /// Gets rank of the tensor: number of dimensions.
    /// </summary>
    public int Rank { get; }

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public MemSpan MemSpan => (MemSpan)Operands[0];

    /// <summary>
    /// Gets the storage model of the backing physical buffer.
    /// </summary>
    public BufferStorage Storage => MemSpan.Buffer.Storage;

    /// <summary>
    /// Gets the shape.
    /// </summary>
    public ReadOnlySpan<Dimension> Dimensions => SpanUtility.UnsafeCast<BaseExpr, Dimension>(Operands[1..(1 + Rank)]);

    /// <summary>
    /// Gets the strides.
    /// <remarks>
    /// This Strides is by elements not by bytes!
    /// </remarks>
    /// </summary>
    public ReadOnlySpan<Dimension> Strides => SpanUtility.UnsafeCast<BaseExpr, Dimension>(Operands[(1 + Rank)..(1 + Rank + Rank)]);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitBuffer(this, context);

    public Buffer With(string? name = null, IRType? type = null, MemSpan? memSpan = null, Dimension[]? dimensions = null, Dimension[]? strides = null, Expr[]? globalShape = null)
        => new Buffer(name ?? Name, type ?? Type, memSpan ?? MemSpan, dimensions ?? Dimensions.ToArray(), strides ?? Strides.ToArray());

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is TIR.Buffer other && GetHashCode() == other.GetHashCode() && Name == other.Name && Type == other.Type && Rank == other.Rank && Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Name, Type, Rank, base.GetHashCodeCore());

    private static TensorType GetTensorType(IRType type, string name) => type switch
    {
        TensorType tensorType => tensorType,
        DistributedType distributedType => distributedType.TensorType,
        _ => throw new InvalidOperationException($"TIR buffer {name} requires TensorType or DistributedType, got {type}."),
    };
}
