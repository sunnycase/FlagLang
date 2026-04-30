// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Buffers;

namespace Nncase.IR.F;

/// <summary>
/// Random functional helper.
/// </summary>
public static class Buffer
{
    /// <summary>
    /// the placeholder for this expr's ddr pointer.
    /// </summary>
    public static Call AddressOf(Expr input) =>
        new Call(new AddressOf(), input);

    public static BufferOf BufferOf(Expr input) => new BufferOf(input);

    /// <summary>
    /// the placeholder for this expr's index.
    /// </summary>
    public static Call BufferIndexOf(Expr input) =>
        new Call(new BufferIndexOf(), input);

    /// <summary>
    /// the placeholder for the expr's basement value.
    /// </summary>
    public static Call BaseMentOf(Expr input) =>
        new Call(new BaseMentOf(), input);

    /// <summary>
    /// the placeholder for the expr's strides.
    /// </summary>
    public static Call StrideOf(Expr input) => new Call(new StrideOf(), input);

    /// <summary>
    /// create the uninitialized buffer.
    /// </summary>
    public static Call Uninitialized(DataType dataType, TIR.MemoryLocation memoryLocation, Shape shape) => Uninitialized(dataType, TIR.BufferStorage.FromLegacy(memoryLocation), shape);

    public static Call Uninitialized(DataType dataType, TIR.BufferStorage storage, Shape shape) => new Call(new Uninitialized(dataType, storage, new IRArray<SBP>(), new Placement(new IRArray<int>(), string.Empty)), shape);

    public static Call Uninitialized(DataType dataType, TIR.MemoryLocation memoryLocation, Shape shape, IRArray<SBP> ndsbp, Placement placement) => Uninitialized(dataType, TIR.BufferStorage.FromLegacy(memoryLocation), shape, ndsbp, placement);

    public static Call Uninitialized(DataType dataType, TIR.BufferStorage storage, Shape shape, IRArray<SBP> ndsbp, Placement placement) => new Call(new Uninitialized(dataType, storage, ndsbp, placement), shape);

    public static Call Allocate(Expr size, DataType dataType, TIR.MemoryLocation location, bool malloc = true) => new Call(new Allocate(dataType, location, malloc), size);

    public static Call AllocateBufferView(Expr buffer) => new Call(new AllocateBufferView(), buffer);

    public static Call BufferSubview(Expr buffer, Shape offset, Shape shape) => new Call(new BufferSubview(), buffer, offset, shape);
}
