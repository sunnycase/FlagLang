// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// the memory type.
/// </summary>
[Flags]
public enum MemoryLocation
{
    /// <summary>
    /// input.
    /// </summary>
    Input = 1 << 1,

    /// <summary>
    /// output.
    /// </summary>
    Output = 1 << 2,

    /// <summary>
    /// constant data.
    /// </summary>
    Rdata = 1 << 3,

    /// <summary>
    /// thread local constant data.
    /// </summary>
    ThreadLocalRdata = 1 << 5,

    /// <summary>
    /// lane local constant data.
    /// </summary>
    WarpLocalRdata = 1 << 4,

    /// <summary>
    /// block local constant data.
    /// </summary>
    BlockLocalRdata = 1 << 6,

    /// <summary>
    /// compute temp data.
    /// </summary>
    Data = 1 << 7,

    /// <summary>
    /// warp local data.
    /// </summary>
    WarpLocalData = 1 << 8,

    /// <summary>
    /// block local data.
    /// </summary>
    BlockLocalData = 1 << 9,

    /// <summary>
    /// cache.
    /// </summary>
    Cache = 1 << 10,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 1 << 11,
}

public sealed class PhysicalBuffer : BaseExpr
{
    public PhysicalBuffer(int alignment, Dimension size, MemoryLocation location, int hierarchy = 0)
        : this(alignment, None.Default, size, location, hierarchy, BufferStorage.FromLegacy(location, hierarchy))
    {
    }

    public PhysicalBuffer(int alignment, Expr start, Dimension size, MemoryLocation location, int hierarchy = 0)
        : this(alignment, start, size, location, hierarchy, BufferStorage.FromLegacy(location, hierarchy))
    {
    }

    public PhysicalBuffer(int alignment, Dimension size, BufferStorage storage)
        : this(alignment, None.Default, size, ToAddressableLegacyMemoryLocation(storage), storage.Hierarchy, storage with { Alignment = alignment })
    {
    }

    public PhysicalBuffer(int alignment, Expr start, Dimension size, BufferStorage storage)
        : this(alignment, start, size, ToAddressableLegacyMemoryLocation(storage), storage.Hierarchy, storage with { Alignment = alignment })
    {
    }

    private PhysicalBuffer(int alignment, Expr start, Dimension size, MemoryLocation location, int hierarchy, BufferStorage storage)
        : base([start, size])
    {
        storage.ValidateAddressable();
        Alignment = alignment;
        Location = location;
        Hierarchy = hierarchy;
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
    /// Gets the memory location.
    /// </summary>
    public MemoryLocation Location { get; }

    /// <summary>
    /// Gets the memory hierarchy.
    /// </summary>
    public int Hierarchy { get; }

    /// <summary>
    /// Gets the orthogonal storage description.
    /// </summary>
    public BufferStorage Storage { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPhysicalBuffer(this, context);

    public PhysicalBuffer With(int? alignment = null, Expr? start = null, Dimension? size = null, MemoryLocation? location = null, int? hierarchy = null, BufferStorage? storage = null)
    {
        var nextAlignment = alignment ?? Alignment;
        var nextHierarchy = hierarchy ?? storage?.Hierarchy ?? Hierarchy;
        var nextStorage = storage ?? (location.HasValue || hierarchy.HasValue
            ? BufferStorage.FromLegacy(location ?? Location, nextHierarchy)
            : Storage);
        nextStorage = nextStorage with { Hierarchy = nextHierarchy, Alignment = nextAlignment };
        var storageLocation = ToAddressableLegacyMemoryLocation(nextStorage);
        var nextLocation = location ?? storageLocation;
        if (storage is not null && location.HasValue && location.Value != storageLocation)
        {
            throw new InvalidOperationException($"Storage {nextStorage} maps to {storageLocation}, but requested legacy location {location.Value}.");
        }

        return new PhysicalBuffer(nextAlignment, start ?? Start, size ?? Size, nextLocation, nextHierarchy, nextStorage);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (ReferenceEquals(this, obj))
        {
            return true;
        }

        return obj is PhysicalBuffer other && GetHashCode() == other.GetHashCode() && Location == other.Location && Storage == other.Storage && Operands.SequenceEqual(other.Operands);
    }

    protected override int GetHashCodeCore() => HashCode.Combine(Location, Storage, base.GetHashCodeCore());

    private static MemoryLocation ToAddressableLegacyMemoryLocation(BufferStorage storage)
    {
        storage.ValidateAddressable();
        return storage.ToLegacyMemoryLocation();
    }
}
