// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;

namespace Nncase.TIR;

public sealed class PhysicalBuffer : BaseExpr
{
    public PhysicalBuffer(int alignment, Dimension size, BufferStorage storage)
        : this(alignment, None.Default, size, storage with { Alignment = alignment })
    {
    }

    public PhysicalBuffer(int alignment, Expr start, Dimension size, BufferStorage storage)
        : base([start, size])
    {
        if (!storage.IsAddressable && start is not None)
        {
            throw new InvalidOperationException(
                $"Storage {storage} is not addressable and must not have a physical start address. " +
                "Lower it as a logical register tile and let codegen materialize typed variables.");
        }

        Alignment = alignment;
        Storage = storage.Alignment == alignment ? storage : storage with { Alignment = alignment };
    }

    /// <summary>
    /// Gets the start.
    /// </summary>
    public Expr Start => (Expr)Operands[0];

    /// <summary>
    /// Gets the size of bytes.
    /// </summary>
    public Dimension Size => (Dimension)Operands[1];

    /// <summary>
    /// Gets the alignment.
    /// </summary>
    public int Alignment { get; }

    /// <summary>
    /// Gets the memory hierarchy.
    /// </summary>
    public int Hierarchy => Storage.Hierarchy;

    /// <summary>
    /// Gets the orthogonal storage description.
    /// </summary>
    public BufferStorage Storage { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPhysicalBuffer(this, context);

    public PhysicalBuffer With(int? alignment = null, Expr? start = null, Dimension? size = null, int? hierarchy = null, BufferStorage? storage = null)
    {
        var nextAlignment = alignment ?? Alignment;
        var nextHierarchy = hierarchy ?? storage?.Hierarchy ?? Hierarchy;
        var nextStorage = storage ?? Storage;
        nextStorage = nextStorage with { Hierarchy = nextHierarchy, Alignment = nextAlignment };
        return new PhysicalBuffer(nextAlignment, start ?? Start, size ?? Size, nextStorage);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is PhysicalBuffer other && GetHashCode() == other.GetHashCode() && Storage == other.Storage && Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Storage, base.GetHashCodeCore());
}
