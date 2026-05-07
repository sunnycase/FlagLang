// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

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

    public static BufferStorage GlobalInput(int hierarchy = 0) => new(BufferUsage.Input, BufferScope.Device, PhysicalMemorySpace.GMem, hierarchy);

    public static BufferStorage GlobalOutput(int hierarchy = 0) => new(BufferUsage.Output, BufferScope.Device, PhysicalMemorySpace.GMem, hierarchy);

    public static BufferStorage DeviceConst(int hierarchy = 0) => new(BufferUsage.Const, BufferScope.Device, PhysicalMemorySpace.ConstMem, hierarchy);

    public static BufferStorage ThreadLocalConst(int hierarchy = 0) => new(BufferUsage.Const, BufferScope.ThreadLocal, PhysicalMemorySpace.ConstMem, hierarchy);

    public static BufferStorage WarpLocalConst(int hierarchy = 0) => new(BufferUsage.Const, BufferScope.WarpLocal, PhysicalMemorySpace.ConstMem, hierarchy);

    public static BufferStorage BlockLocalConst(int hierarchy = 0) => new(BufferUsage.Const, BufferScope.BlockLocal, PhysicalMemorySpace.ConstMem, hierarchy);

    public static BufferStorage ThreadLocalTemp(int hierarchy = 0) => new(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable, hierarchy);

    public static BufferStorage WarpLocalTemp(int hierarchy = 0) => new(BufferUsage.Temp, BufferScope.WarpLocal, PhysicalMemorySpace.LocalAddressable, hierarchy);

    public static BufferStorage BlockLocalSMem(int hierarchy = 0) => new(BufferUsage.Temp, BufferScope.BlockLocal, PhysicalMemorySpace.SMem, hierarchy);

    public static BufferStorage PrivateBase(int hierarchy = 0) => new(BufferUsage.PrivateBase, BufferScope.ThreadLocal, PhysicalMemorySpace.LocalAddressable, hierarchy);

    public void ValidateAddressable()
    {
        if (!IsAddressable)
        {
            throw new InvalidOperationException($"Storage {this} is not addressable and must not be materialized as a PhysicalBuffer.");
        }
    }

    public override string ToString() => $"Usage={Usage}, Scope={Scope}, Location={PhysicalLocation}, Hierarchy={Hierarchy}, Alignment={Alignment}";
}
