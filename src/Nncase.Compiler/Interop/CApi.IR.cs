// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using CommunityToolkit.HighPerformance;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Compiler.Interop;

public static unsafe partial class CApi
{
    [UnmanagedCallersOnly]
    private static IntPtr FileLocationCreate(byte* filePathPtr, nuint filePathLength, int startLine, int startColumn, int endLine, int endColumn)
    {
        var filePath = ToString(filePathPtr, filePathLength);
        var location = new Diagnostics.FileLocation(filePath!, startLine, startColumn, endLine, endColumn);
        return GCHandle.ToIntPtr(GCHandle.Alloc(location));
    }

    [UnmanagedCallersOnly]
    private static IntPtr NameLocationCreate(byte* namePtr, nuint nameLength, IntPtr childLocationPtr)
    {
        var name = ToString(namePtr, nameLength);
        var childLocation = GetNullable<Diagnostics.Location>(childLocationPtr);
        var location = new Diagnostics.NameLocation(name!, childLocation);
        return GCHandle.ToIntPtr(GCHandle.Alloc(location));
    }

    [UnmanagedCallersOnly]
    private static IntPtr IRModuleCreate()
    {
        var module = new IR.IRModule();
        return GCHandle.ToIntPtr(GCHandle.Alloc(module));
    }

    [UnmanagedCallersOnly]
    private static void IRModuleAdd(IntPtr modulePtr, IntPtr functionPtr)
    {
        var module = Get<IR.IRModule>(modulePtr);
        var function = Get<IR.BaseFunction>(functionPtr);
        module.Add(function);
    }

    [UnmanagedCallersOnly]
    private static IntPtr IRModuleGetFunctionByName(IntPtr modulePtr, byte* namePtr, nuint nameLength)
    {
        var module = Get<IR.IRModule>(modulePtr);
        var name = ToString(namePtr, nameLength);
        var function = module.Functions.IndexOf(f => f.Name == name);
        return function == -1 ? IntPtr.Zero : GCHandle.ToIntPtr(GCHandle.Alloc(module.Functions[function]));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetBoolean()
    {
        var dt = DataTypes.Boolean;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetInt8()
    {
        var dt = DataTypes.Int8;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetInt16()
    {
        var dt = DataTypes.Int16;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetInt32()
    {
        var dt = DataTypes.Int32;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetInt64()
    {
        var dt = DataTypes.Int64;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetUInt8()
    {
        var dt = DataTypes.UInt8;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetUInt16()
    {
        var dt = DataTypes.UInt16;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetUInt32()
    {
        var dt = DataTypes.UInt32;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetUInt64()
    {
        var dt = DataTypes.UInt64;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetFloat16()
    {
        var dt = DataTypes.Float16;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetBFloat16()
    {
        var dt = DataTypes.BFloat16;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetFloat32()
    {
        var dt = DataTypes.Float32;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr DataTypesGetFloat64()
    {
        var dt = DataTypes.Float64;
        return GCHandle.ToIntPtr(GCHandle.Alloc(dt));
    }

    [UnmanagedCallersOnly]
    private static IntPtr PointerTypeCreate(IntPtr elemTypePtr, int addressSpace)
    {
        var elemType = Get<DataType>(elemTypePtr);
        var pointerType = new PointerType(elemType, addressSpace);
        return GCHandle.ToIntPtr(GCHandle.Alloc(pointerType));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CallableTypeCreate(IntPtr retTypePtr, IntPtr* paramTypePtrs, nuint paramCount)
    {
        var retType = Get<IRType>(retTypePtr);
        var paramTypes = new IRType[paramCount];
        for (nuint i = 0; i < paramCount; i++)
        {
            paramTypes[i] = Get<IRType>(paramTypePtrs[i]);
        }

        var callableType = new CallableType(retType, paramTypes);
        return GCHandle.ToIntPtr(GCHandle.Alloc(callableType));
    }

    [UnmanagedCallersOnly]
    private static IntPtr VoidTypeGet()
    {
        var voidType = TupleType.Void;
        return GCHandle.ToIntPtr(GCHandle.Alloc(voidType));
    }

    [UnmanagedCallersOnly]
    private static IntPtr TupleTypeCreate(IntPtr* fieldTypePtrs, nuint fieldCount)
    {
        var fieldTypes = new DataType[fieldCount];
        for (nuint i = 0; i < fieldCount; i++)
        {
            fieldTypes[i] = Get<DataType>(fieldTypePtrs[i]);
        }

        var tupleType = new IR.TupleType(fieldTypes);
        return GCHandle.ToIntPtr(GCHandle.Alloc(tupleType));
    }

    [UnmanagedCallersOnly]
    private static IntPtr TensorTypeCreate(IntPtr dataTypePtr, IntPtr shapePtr)
    {
        var dataType = Get<DataType>(dataTypePtr);
        var shape = Get<IR.Shape>(shapePtr);
        var tensorType = new IR.TensorType(dataType, shape);
        return GCHandle.ToIntPtr(GCHandle.Alloc(tensorType));
    }

    [UnmanagedCallersOnly]
    private static IntPtr TensorTypeGetShape(IntPtr tensorTypePtr)
    {
        var tensorType = Get<IR.TensorType>(tensorTypePtr);
        var shape = tensorType.Shape;
        return GCHandle.ToIntPtr(GCHandle.Alloc(shape));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ShapeCreateFixed(nint* dimsPtr, nuint dimCount)
    {
        var dims = new long[dimCount];
        for (nuint i = 0; i < dimCount; i++)
        {
            dims[i] = dimsPtr[i];
        }

        var shape = new IR.RankedShape(dims);
        return GCHandle.ToIntPtr(GCHandle.Alloc(shape));
    }

    [UnmanagedCallersOnly]
    private static void BaseExprSetInt32Attribute(IntPtr exprPtr, byte* namePtr, nuint nameLength, int value)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var name = ToString(namePtr, nameLength);
        expr.Metadata.Attributes[name] = value;
    }

    [UnmanagedCallersOnly]
    private static IntPtr BaseExprGetLocation(IntPtr exprPtr)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var location = expr.Metadata.Location;
        return GCHandle.ToIntPtr(GCHandle.Alloc(location));
    }

    [UnmanagedCallersOnly]
    private static void BaseExprSetLocation(IntPtr exprPtr, IntPtr locationPtr)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var location = Get<Diagnostics.Location>(locationPtr);
        expr.Metadata.Location = location;
    }

    [UnmanagedCallersOnly]
    private static IntPtr BaseExprGetShape(IntPtr exprPtr)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var type = expr.CheckedShape;
        return GCHandle.ToIntPtr(GCHandle.Alloc(type));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Math_Binary(BinaryOp binaryOp, IntPtr aPtr, IntPtr bPtr)
    {
        var a = Get<IR.Expr>(aPtr);
        var b = Get<IR.Expr>(bPtr);
        var compare = IR.F.Math.Binary(binaryOp, a, b);
        return GCHandle.ToIntPtr(GCHandle.Alloc(compare));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Math_Compare(CompareOp compareOp, IntPtr aPtr, IntPtr bPtr)
    {
        var a = Get<IR.Expr>(aPtr);
        var b = Get<IR.Expr>(bPtr);
        var compare = IR.F.Math.Compare(compareOp, a, b);
        return GCHandle.ToIntPtr(GCHandle.Alloc(compare));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Shapes_AsTensor(IntPtr dimensionPtr)
    {
        var dimension = Get<IR.Dimension>(dimensionPtr);
        var value = IR.F.Shapes.AsTensor(dimension);
        return GCHandle.ToIntPtr(GCHandle.Alloc(value));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Tensors_Broadcast(IntPtr valuePtr, IntPtr shapePtr)
    {
        var value = Get<IR.Expr>(valuePtr);
        var shape = Get<IR.Shape>(shapePtr);
        var broadcasted = IR.F.Tensors.Broadcast(value, shape);
        return GCHandle.ToIntPtr(GCHandle.Alloc(broadcasted));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Tensors_Range(IntPtr beginPtr, IntPtr endPtr, IntPtr stepPtr)
    {
        var begin = Get<IR.Expr>(beginPtr);
        var end = Get<IR.Expr>(endPtr);
        var step = Get<IR.Expr>(stepPtr);
        var range = IR.F.Tensors.Range(begin, end, step);
        return GCHandle.ToIntPtr(GCHandle.Alloc(range));
    }

    [UnmanagedCallersOnly]
    private static IntPtr TIR_Return(IntPtr* valuePtrs, nuint valueCount)
    {
        var values = new IR.Expr[valueCount];
        for (nuint i = 0; i < valueCount; i++)
        {
            values[i] = Get<IR.Expr>(valuePtrs[i]);
        }

        var ret = new TIR.Return(values);
        return GCHandle.ToIntPtr(GCHandle.Alloc(ret));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Triton_Load(IntPtr ptrPtr, IntPtr maskPtr, IntPtr otherPtr, IR.Triton.CacheModifier cacheModifier, IR.Triton.EvictionPolicy evictionPolicy)
    {
        var ptr = Get<IR.Expr>(ptrPtr);
        var mask = GetNullable<IR.Expr>(maskPtr);
        var other = GetNullable<IR.Expr>(otherPtr);
        var load = IR.F.Triton.Load(ptr, mask, other, cacheModifier, evictionPolicy);
        return GCHandle.ToIntPtr(GCHandle.Alloc(load));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Triton_Store(IntPtr ptrPtr, IntPtr valuePtr, IntPtr maskPtr, IR.Triton.CacheModifier cacheModifier, IR.Triton.EvictionPolicy evictionPolicy)
    {
        var ptr = Get<IR.Expr>(ptrPtr);
        var value = Get<IR.Expr>(valuePtr);
        var mask = GetNullable<IR.Expr>(maskPtr);
        var store = IR.F.Triton.Store(ptr, value, mask, cacheModifier, evictionPolicy);
        return GCHandle.ToIntPtr(GCHandle.Alloc(store));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ProgramId(int axis)
    {
        var pid = IR.F.Distributed.ProgramId(axis);
        return GCHandle.ToIntPtr(GCHandle.Alloc(pid));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ScalarInt32(int value)
    {
        var scalar = (TensorConst)Tensor.FromScalar(value);
        return GCHandle.ToIntPtr(GCHandle.Alloc(scalar));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ScalarInt64(long value)
    {
        var scalar = (TensorConst)Tensor.FromScalar(value);
        return GCHandle.ToIntPtr(GCHandle.Alloc(scalar));
    }

    [UnmanagedCallersOnly]
    private static IntPtr Cast(IntPtr valuePtr, IntPtr targetTypePtr, CastMode mode)
    {
        var value = Get<IR.Expr>(valuePtr);
        var targetDType = IRType.GetDataType(Get<IRType>(targetTypePtr));
        var casted = IR.F.Tensors.Cast(value, targetDType, mode);
        return GCHandle.ToIntPtr(GCHandle.Alloc(casted));
    }

    [UnmanagedCallersOnly]
    private static nuint SequentialGetParametersCount(IntPtr blockPtr)
    {
        var block = Get<TIR.Sequential>(blockPtr);
        return (nuint)block.Parameters.Length;
    }

    [UnmanagedCallersOnly]
    private static nuint SequentialGetFieldsCount(IntPtr blockPtr)
    {
        var block = Get<TIR.Sequential>(blockPtr);
        return (nuint)block.Fields.Length;
    }

    [UnmanagedCallersOnly]
    private static IntPtr SequentialGetParameter(IntPtr blockPtr, nuint index)
    {
        var block = Get<TIR.Sequential>(blockPtr);
        var parameter = block.Parameters[(int)index];
        return GCHandle.ToIntPtr(GCHandle.Alloc(parameter));
    }

    [UnmanagedCallersOnly]
    private static byte SequentialHasTerminator(IntPtr blockPtr)
    {
        var block = Get<TIR.Sequential>(blockPtr);
        return (byte)(block.HasTerminator ? 1 : 0);
    }

    [UnmanagedCallersOnly]
    private static void SequentialInsertAt(IntPtr blockPtr, int index, IntPtr exprPtr)
    {
        var block = Get<TIR.Sequential>(blockPtr);
        var expr = Get<IR.Expr>(exprPtr);
        block.InsertAt(index, expr);
    }

    [UnmanagedCallersOnly]
    private static IntPtr PrimFunctionCreate(byte* namePtr, nuint nameLength, IntPtr callableTypePtr)
    {
        TensorType CanonicalizeParamType(IRType t) => t switch
        {
            DataType dt => IR.TensorType.Scalar(dt),
            IR.TensorType tt => tt,
            _ => throw new NotSupportedException($"Unsupported parameter type: {t}"),
        };

        var name = ToString(namePtr, nameLength);
        var callableType = Get<IR.CallableType>(callableTypePtr);
        var parameters = callableType.Parameters
            .Select((t, i) => new IR.Var(
                $"param_{i}",
                CanonicalizeParamType(t)))
            .ToArray();
        var body = new TIR.Sequential([], parameters);
        var function = new TIR.PrimFunction(name, body);
        return GCHandle.ToIntPtr(GCHandle.Alloc(function));
    }

    [UnmanagedCallersOnly]
    private static IntPtr PrimFunctionAddBody(IntPtr functionPtr)
    {
        var function = Get<TIR.PrimFunction>(functionPtr);
        var newBody = function.Body.With(fields: []);
        ReplaceUtility.ReplaceAllUsesWith(function.Body, newBody);
        return GCHandle.ToIntPtr(GCHandle.Alloc(newBody));
    }

    [UnmanagedCallersOnly]
    private static IntPtr PrimFunctionGetBody(IntPtr functionPtr)
    {
        var function = Get<TIR.PrimFunction>(functionPtr);
        return GCHandle.ToIntPtr(GCHandle.Alloc(function.Body));
    }

    [UnmanagedCallersOnly]
    private static IntPtr TupleCreate(IntPtr* fieldPtrs, nuint fieldCount)
    {
        IR.ITuple tuple;
        if (fieldCount == 0)
        {
            tuple = IR.Tuple.Void;
        }
        else
        {
            var fields = new IR.Expr[fieldCount];
            for (nuint i = 0; i < fieldCount; i++)
            {
                fields[i] = Get<IR.Expr>(fieldPtrs[i]);
            }

            tuple = new IR.Tuple(fields);
        }

        return GCHandle.ToIntPtr(GCHandle.Alloc(tuple));
    }
}
