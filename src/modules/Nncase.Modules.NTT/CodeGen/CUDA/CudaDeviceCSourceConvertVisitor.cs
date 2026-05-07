// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.CodeGen.NTT;

public class CudaDeviceCSourceConvertVisitor : DeviceCSourceConvertVisitor
{
    private readonly Dictionary<PhysicalBuffer, RegisterStorageDecl> _registerStorageDecls = new(ReferenceEqualityComparer.Instance);
    private int _nextRegisterStorageId;

    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.Storage.PhysicalLocation is not PhysicalMemorySpace.Register)
        {
            return base.VisitPhysicalBuffer(expr);
        }

        throw new InvalidOperationException(
            $"CUDA register buffer storage {expr.Storage} requires a typed MemSpan context for codegen. " +
            "Materialize register storage through TIR.Buffer or a typed reinterpret cast.");
    }

    protected override CSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.Buffer.Storage.PhysicalLocation is not PhysicalMemorySpace.Register)
        {
            return base.VisitMemSpan(expr);
        }

        var elemType = ResolveRegisterElementType(expr);
        var decl = EnsureRegisterStorage(expr.Buffer, elemType);
        var start = GetFixedElementCount(expr.Start, decl.ElementBytes, "start");
        var extent = GetFixedElementCount(expr.Size, decl.ElementBytes, "size");
        if (start < 0 || extent <= 0 || start + extent > decl.ElementCount)
        {
            throw new InvalidOperationException(
                $"CUDA register memspan [{start}, {start + extent}) exceeds register storage {decl.Name} with {decl.ElementCount} elements.");
        }

        var span = $"ntt::span<{decl.ElementType}, {decl.ElementCount}>({decl.Name}, {decl.ElementCount})";
        var str = start == 0 && extent == decl.ElementCount
            ? span
            : $"make_subspan({span}, {start}_dim, {extent}_dim)";
        symbol = new($"ntt::span<{decl.ElementType}, {extent}>", str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    private RegisterStorageDecl EnsureRegisterStorage(PhysicalBuffer buffer, DataType elemType)
    {
        if (_registerStorageDecls.TryGetValue(buffer, out var decl))
        {
            if (decl.ElementType != elemType.ToC())
            {
                throw new InvalidOperationException(
                    $"CUDA register storage {decl.Name} was declared as {decl.ElementType}, but a {elemType.ToC()} view was requested.");
            }

            return decl;
        }

        if (buffer.Storage.Scope is not BufferScope.ThreadLocal)
        {
            throw new NotSupportedException($"CUDA register buffer storage must be thread-local, got {buffer.Storage}.");
        }

        if (buffer.Start is not None)
        {
            throw new InvalidOperationException($"CUDA register buffer storage must not have an addressable start: {buffer.Start}.");
        }

        var sizeBytes = GetFixedByteCount(buffer);
        var elemBytes = elemType.SizeInBytes;
        if (sizeBytes % elemBytes != 0)
        {
            throw new InvalidOperationException(
                $"CUDA register buffer byte size {sizeBytes} is not divisible by element size {elemBytes} for {elemType}.");
        }

        var elemCount = sizeBytes / elemBytes;
        var dtype = elemType.ToC();
        var name = $"register_storage_{_nextRegisterStorageId++}";
        var alignment = buffer.Alignment > 1 ? $"alignas({buffer.Alignment}) " : string.Empty;
        IndentScope.Writer.IndWrite($"{alignment}{dtype} {name}[{elemCount}];\n");

        decl = new RegisterStorageDecl(name, dtype, elemBytes, elemCount);
        _registerStorageDecls.Add(buffer, decl);
        return decl;
    }

    private DataType ResolveRegisterElementType(MemSpan expr)
    {
        var candidates = expr.Users.Select(user => user switch
        {
            TIR.Buffer buffer => buffer.ElemType,
            Call { Target: IR.Tensors.Cast { CastMode: CastMode.Reinterpret, NewType: PointerType { ElemType: DataType elemType } } } => elemType,
            _ => null,
        }).Where(type => type is not null).Cast<DataType>().ToArray();

        if (candidates.Length == 0)
        {
            throw new InvalidOperationException($"CUDA register memspan {expr} is not owned by a typed buffer or reinterpret cast.");
        }

        var elemType = candidates[0];
        foreach (var candidate in candidates[1..])
        {
            if (!Equals(elemType, candidate))
            {
                throw new InvalidOperationException(
                    $"CUDA register memspan {expr} has incompatible typed users: {elemType} and {candidate}.");
            }
        }

        return elemType;
    }

    private long GetFixedElementCount(Dimension bytes, int elemBytes, string fieldName)
    {
        if (!bytes.IsFixed)
        {
            throw new NotSupportedException($"CUDA register memspan {fieldName} requires a fixed byte count, got {bytes}.");
        }

        var byteCount = bytes.FixedValue;
        if (byteCount % elemBytes != 0)
        {
            throw new InvalidOperationException(
                $"CUDA register memspan {fieldName} byte count {byteCount} is not divisible by element size {elemBytes}.");
        }

        return byteCount / elemBytes;
    }

    private long GetFixedByteCount(PhysicalBuffer buffer)
    {
        if (!buffer.Size.IsFixed)
        {
            throw new NotSupportedException($"CUDA register buffer storage requires a fixed byte size, got {buffer.Size}.");
        }

        if (buffer.Size.FixedValue <= 0)
        {
            throw new InvalidOperationException($"CUDA register buffer storage requires a positive byte size, got {buffer.Size.FixedValue}.");
        }

        return buffer.Size.FixedValue;
    }

    private sealed record RegisterStorageDecl(string Name, string ElementType, int ElementBytes, long ElementCount);
}
