// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;

namespace Nncase.TIR;

/// <summary>
/// Semantic role of a buffer.
/// </summary>
public enum BufferUsage : byte
{
    Input,
    Output,
    Const,
    Temp,
    Scratch,
    Staging,
    Accumulator,
    Metadata,
    PrivateBase,
}

/// <summary>
/// Sharing and visibility scope of a buffer instance.
/// </summary>
public enum BufferScope : byte
{
    ThreadLocal,
    WarpLocal,
    BlockLocal,
    ClusterLocal,
    Grid,
    Device,
    Host,
}

/// <summary>
/// Physical memory space used by a buffer.
/// </summary>
public enum PhysicalMemorySpace : byte
{
    Register,
    L1,
    L2,
    SMem,
    TMem,
    GMem,
    ConstMem,
    LocalAddressable,
    Stack,
    Heap,
    DRAM,
}

/// <summary>
/// Orthogonal storage description for buffers.
/// </summary>
public sealed record BufferStorage(
    BufferUsage Usage,
    BufferScope Scope,
    PhysicalMemorySpace PhysicalLocation,
    int Hierarchy = 0,
    int Alignment = 0)
{
    public bool IsAddressable => PhysicalLocation is not PhysicalMemorySpace.Register;

    public BufferStorage WithoutAlignment() => this with { Alignment = 0 };

    public static BufferStorage FromLegacy(MemoryLocation location, int hierarchy = 0) => location switch
    {
        MemoryLocation.Input => new(BufferUsage.Input, BufferScope.Device, PhysicalMemorySpace.GMem, hierarchy),
        MemoryLocation.Output => new(BufferUsage.Output, BufferScope.Device, PhysicalMemorySpace.GMem, hierarchy),
        MemoryLocation.Rdata => new(BufferUsage.Const, BufferScope.Device, PhysicalMemorySpace.ConstMem, hierarchy),
        MemoryLocation.ThreadLocalRdata => new(BufferUsage.Const, BufferScope.ThreadLocal, PhysicalMemorySpace.ConstMem, hierarchy),
        MemoryLocation.WarpLocalRdata => new(BufferUsage.Const, BufferScope.WarpLocal, PhysicalMemorySpace.ConstMem, hierarchy),
        MemoryLocation.BlockLocalRdata => new(BufferUsage.Const, BufferScope.BlockLocal, PhysicalMemorySpace.ConstMem, hierarchy),
        MemoryLocation.Data => new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable, hierarchy),
        MemoryLocation.WarpLocalData => new(BufferUsage.Temp, BufferScope.WarpLocal, PhysicalMemorySpace.LocalAddressable, hierarchy),
        MemoryLocation.BlockLocalData => new(BufferUsage.Temp, BufferScope.BlockLocal, PhysicalMemorySpace.SMem, hierarchy),
        MemoryLocation.Cache => new(BufferUsage.Scratch, BufferScope.ThreadLocal, PhysicalMemorySpace.L1, hierarchy),
        MemoryLocation.PrivateBase => new(BufferUsage.PrivateBase, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable, hierarchy),
        _ => throw new NotSupportedException($"Unsupported legacy memory location: {location}"),
    };

    public MemoryLocation ToLegacyMemoryLocation() => (Usage, Scope, PhysicalLocation) switch
    {
        (BufferUsage.Input, BufferScope.Device, PhysicalMemorySpace.GMem) => MemoryLocation.Input,
        (BufferUsage.Output, BufferScope.Device, PhysicalMemorySpace.GMem) => MemoryLocation.Output,
        (BufferUsage.Const, BufferScope.ThreadLocal, PhysicalMemorySpace.ConstMem) => MemoryLocation.ThreadLocalRdata,
        (BufferUsage.Const, BufferScope.WarpLocal, PhysicalMemorySpace.ConstMem) => MemoryLocation.WarpLocalRdata,
        (BufferUsage.Const, BufferScope.BlockLocal, PhysicalMemorySpace.ConstMem) => MemoryLocation.BlockLocalRdata,
        (BufferUsage.Const, BufferScope.Device, PhysicalMemorySpace.ConstMem) => MemoryLocation.Rdata,
        (BufferUsage.Temp, BufferScope.WarpLocal, PhysicalMemorySpace.LocalAddressable) => MemoryLocation.WarpLocalData,
        (BufferUsage.Temp, BufferScope.BlockLocal, PhysicalMemorySpace.SMem) => MemoryLocation.BlockLocalData,
        (BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable) => MemoryLocation.Data,
        (BufferUsage.Scratch, BufferScope.ThreadLocal, PhysicalMemorySpace.L1) => MemoryLocation.Cache,
        (BufferUsage.PrivateBase, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable) => MemoryLocation.PrivateBase,
        _ => throw new NotSupportedException($"Storage {this} cannot be represented by legacy MemoryLocation."),
    };

    public void ValidateAddressable()
    {
        if (!IsAddressable)
        {
            throw new InvalidOperationException($"Storage {this} is not addressable and must not be materialized as a PhysicalBuffer.");
        }
    }

    public override string ToString() => $"Usage={Usage}, Scope={Scope}, Location={PhysicalLocation}, Hierarchy={Hierarchy}, Alignment={Alignment}";
}
