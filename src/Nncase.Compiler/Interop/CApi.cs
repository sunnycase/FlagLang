// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.IR;

namespace Nncase.Compiler.Interop;

public enum ArrayElementKind
{
    /// <summary>
    /// <see cref="Runtime.Interop.RTValue"/>.
    /// </summary>
    RTValue = 0,
    Var = 1,
    Object = 2,
}

/// <summary>
/// Compiler C Api method table.
/// </summary>
[StructLayout(LayoutKind.Sequential)]

[SuppressMessage("StyleCop.CSharp.NamingRules", "SA1310:Field names should not contain underscore", Justification = "Method table", Scope = "member")]
public unsafe struct CApiMT
{
    // CLR functions.
    public delegate* unmanaged<ArrayElementKind, IntPtr*, nuint, IntPtr> ArrayCreatePtr;
    public delegate* unmanaged<IntPtr, nuint, IntPtr> ArrayGetItemPtr;
    public delegate* unmanaged<IntPtr, nuint> ArrayGetLengthPtr;
    public delegate* unmanaged<IntPtr, void> ClrHandleDisposePtr;
    public delegate* unmanaged<IntPtr, IntPtr> ClrHandleDuplicatePtr;
    public delegate* unmanaged<IntPtr, void> ClrHandleFreePtr;
    public delegate* unmanaged<CStreamMT*, IntPtr, IntPtr> StreamCreatePtr;
    public delegate* unmanaged<byte*, nuint, nuint> LastErrorGetPtr;

    // Hosting functions.
    public delegate* unmanaged<void> CompilerInitializePtr;
    public delegate* unmanaged<byte*, nuint, IntPtr> TargetCreatePtr;

    // Compile functions.
    public delegate* unmanaged<IntPtr> CompileOptionsCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> CompileSessionCreatePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, IntPtr> CompileSessionCreatePassManagerPtr;

    public delegate* unmanaged<IntPtr, byte> CompilerServices_InferenceTypePtr;

    public delegate* unmanaged<IntPtr, int, byte> PassManagerAddOptimizeTTIRPtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> PassManagerRunPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, IntPtr> IRModuleCompileToCubinPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> NativeCudaCompileResultGetJsonPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> NativeCudaCompileResultGetCubinPtr;

    // IR functions.
    public delegate* unmanaged<byte*, nuint, int, int, int, int, IntPtr> FileLocationCreatePtr;
    public delegate* unmanaged<byte*, nuint, IntPtr, IntPtr> NameLocationCreatePtr;

    public delegate* unmanaged<IntPtr> IRModuleCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> IRModuleAddPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> IRModuleSetEntryPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, IntPtr> IRModuleGetFunctionByNamePtr;

    public delegate* unmanaged<IntPtr> DataTypesGetBooleanPtr;
    public delegate* unmanaged<IntPtr> DataTypesGetInt8Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetInt16Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetInt32Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetInt64Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetUInt8Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetUInt16Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetUInt32Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetUInt64Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetFloat16Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetBFloat16Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetFloat32Ptr;
    public delegate* unmanaged<IntPtr> DataTypesGetFloat64Ptr;
    public delegate* unmanaged<IntPtr, int, IntPtr> PointerTypeCreatePtr;

    public delegate* unmanaged<IntPtr, IntPtr*, nuint, IntPtr> CallableTypeCreatePtr;
    public delegate* unmanaged<IntPtr> VoidTypeGetPtr;
    public delegate* unmanaged<IntPtr*, nuint, IntPtr> TupleTypeCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> TensorTypeCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr> TensorTypeGetShapePtr;

    public delegate* unmanaged<nint*, nuint, IntPtr> ShapeCreateFixedPtr;

    public delegate* unmanaged<IntPtr, byte*, nuint, int, void> BaseExprSetInt32AttributePtr;
    public delegate* unmanaged<IntPtr, IntPtr> BaseExprGetLocationPtr;
    public delegate* unmanaged<IntPtr, IntPtr, void> BaseExprSetLocationPtr;
    public delegate* unmanaged<IntPtr, IntPtr> BaseExprGetShapePtr;

    public delegate* unmanaged<int, IntPtr> ProgramIdPtr;
    public delegate* unmanaged<int, IntPtr> ScalarInt32Ptr;
    public delegate* unmanaged<long, IntPtr> ScalarInt64Ptr;
    public delegate* unmanaged<float, IntPtr> ScalarFloat16Ptr;
    public delegate* unmanaged<float, IntPtr> ScalarFloat32Ptr;
    public delegate* unmanaged<double, IntPtr> ScalarFloat64Ptr;
    public delegate* unmanaged<IntPtr, IntPtr, CastMode, IntPtr> CastPtr;

    public delegate* unmanaged<BinaryOp, IntPtr, IntPtr, IntPtr> Math_BinaryPtr;
    public delegate* unmanaged<CompareOp, IntPtr, IntPtr, IntPtr> Math_ComparePtr;

    public delegate* unmanaged<IntPtr, IntPtr> Shapes_AsTensorPtr;

    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> Tensors_BroadcastPtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IntPtr> Tensors_RangePtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr> Tensors_UnsqueezePtr;

    public delegate* unmanaged<IntPtr*, nuint, IntPtr> TIR_ReturnPtr;

    public delegate* unmanaged<IntPtr, IntPtr*, nuint, IntPtr> CallCreatePtr;
    public delegate* unmanaged<IntPtr, nuint> CallGetNumResultsPtr;
    public delegate* unmanaged<IntPtr, nuint, IntPtr> CallGetResultPtr;

    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IR.Triton.CacheModifier, IR.Triton.EvictionPolicy, IntPtr> Triton_LoadPtr;
    public delegate* unmanaged<IntPtr, IntPtr, IntPtr, IR.Triton.CacheModifier, IR.Triton.EvictionPolicy, IntPtr> Triton_StorePtr;

    public delegate* unmanaged<IntPtr, nuint> SequentialGetParametersCountPtr;
    public delegate* unmanaged<IntPtr, nuint> SequentialGetFieldsCountPtr;
    public delegate* unmanaged<IntPtr, nuint, IntPtr> SequentialGetParameterPtr;
    public delegate* unmanaged<IntPtr, byte> SequentialHasTerminatorPtr;
    public delegate* unmanaged<IntPtr, int, IntPtr, void> SequentialInsertAtPtr;

    public delegate* unmanaged<byte*, nuint, IntPtr, IntPtr> PrimFunctionCreatePtr;
    public delegate* unmanaged<IntPtr, IntPtr> PrimFunctionAddBodyPtr;
    public delegate* unmanaged<IntPtr, IntPtr> PrimFunctionGetBodyPtr;

    public delegate* unmanaged<IntPtr*, nuint, IntPtr> TupleCreatePtr;

    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> BaseExprPrintPtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> IRModuleGetEntryNamePtr;
    public delegate* unmanaged<IntPtr, byte*, nuint, nuint> IRModuleDescribeVectorAddPtr;
}

/// <summary>
/// Compiler C Api.
/// </summary>
public static unsafe partial class CApi
{
    [ThreadStatic]
    private static string? _lastError;

    [UnmanagedCallersOnly]
    public static void Initialize(CApiMT* mt)
    {
        // CLR functions.
        mt->ArrayCreatePtr = &ArrayCreate;
        mt->ArrayGetItemPtr = &ArrayGetItem;
        mt->ArrayGetLengthPtr = &ArrayGetLength;
        mt->ClrHandleDisposePtr = &ClrHandleDispose;
        mt->ClrHandleDuplicatePtr = &ClrHandleDuplicate;
        mt->ClrHandleFreePtr = &ClrHandleFree;
        mt->StreamCreatePtr = &StreamCreate;
        mt->LastErrorGetPtr = &LastErrorGet;

        // Hosting functions.
        mt->CompilerInitializePtr = &CompilerInitialize;
        mt->TargetCreatePtr = &TargetCreate;

        // Compile functions.
        mt->CompileOptionsCreatePtr = &CompileOptionsCreate;
        mt->CompileSessionCreatePtr = &CompileSessionCreate;
        mt->CompileSessionCreatePassManagerPtr = &CompileSessionCreatePassManager;

        mt->CompilerServices_InferenceTypePtr = &CompilerServices_InferenceType;

        mt->PassManagerAddOptimizeTTIRPtr = &PassManagerAddOptimizeTTIR;
        mt->PassManagerRunPtr = &PassManagerRun;
        mt->IRModuleCompileToCubinPtr = &IRModuleCompileToCubin;
        mt->NativeCudaCompileResultGetJsonPtr = &NativeCudaCompileResultGetJson;
        mt->NativeCudaCompileResultGetCubinPtr = &NativeCudaCompileResultGetCubin;

        // IR functions.
        mt->FileLocationCreatePtr = &FileLocationCreate;
        mt->NameLocationCreatePtr = &NameLocationCreate;

        mt->IRModuleCreatePtr = &IRModuleCreate;
        mt->IRModuleAddPtr = &IRModuleAdd;
        mt->IRModuleSetEntryPtr = &IRModuleSetEntry;
        mt->IRModuleGetFunctionByNamePtr = &IRModuleGetFunctionByName;

        mt->DataTypesGetBooleanPtr = &DataTypesGetBoolean;
        mt->DataTypesGetInt8Ptr = &DataTypesGetInt8;
        mt->DataTypesGetInt16Ptr = &DataTypesGetInt16;
        mt->DataTypesGetInt32Ptr = &DataTypesGetInt32;
        mt->DataTypesGetInt64Ptr = &DataTypesGetInt64;
        mt->DataTypesGetUInt8Ptr = &DataTypesGetUInt8;
        mt->DataTypesGetUInt16Ptr = &DataTypesGetUInt16;
        mt->DataTypesGetUInt32Ptr = &DataTypesGetUInt32;
        mt->DataTypesGetUInt64Ptr = &DataTypesGetUInt64;
        mt->DataTypesGetFloat16Ptr = &DataTypesGetFloat16;
        mt->DataTypesGetBFloat16Ptr = &DataTypesGetBFloat16;
        mt->DataTypesGetFloat32Ptr = &DataTypesGetFloat32;
        mt->DataTypesGetFloat64Ptr = &DataTypesGetFloat64;
        mt->PointerTypeCreatePtr = &PointerTypeCreate;

        mt->CallableTypeCreatePtr = &CallableTypeCreate;
        mt->VoidTypeGetPtr = &VoidTypeGet;
        mt->TupleTypeCreatePtr = &TupleTypeCreate;
        mt->TensorTypeCreatePtr = &TensorTypeCreate;
        mt->TensorTypeGetShapePtr = &TensorTypeGetShape;

        mt->ShapeCreateFixedPtr = &ShapeCreateFixed;

        mt->BaseExprSetInt32AttributePtr = &BaseExprSetInt32Attribute;
        mt->BaseExprGetLocationPtr = &BaseExprGetLocation;
        mt->BaseExprSetLocationPtr = &BaseExprSetLocation;
        mt->BaseExprGetShapePtr = &BaseExprGetShape;

        mt->ProgramIdPtr = &ProgramId;
        mt->ScalarInt32Ptr = &ScalarInt32;
        mt->ScalarInt64Ptr = &ScalarInt64;
        mt->ScalarFloat16Ptr = &ScalarFloat16;
        mt->ScalarFloat32Ptr = &ScalarFloat32;
        mt->ScalarFloat64Ptr = &ScalarFloat64;
        mt->CastPtr = &Cast;

        mt->Math_BinaryPtr = &Math_Binary;
        mt->Math_ComparePtr = &Math_Compare;

        mt->Shapes_AsTensorPtr = &Shapes_AsTensor;

        mt->Tensors_BroadcastPtr = &Tensors_Broadcast;
        mt->Tensors_RangePtr = &Tensors_Range;
        mt->Tensors_UnsqueezePtr = &Tensors_Unsqueeze;

        mt->TIR_ReturnPtr = &TIR_Return;

        mt->CallCreatePtr = &CallCreate;
        mt->CallGetNumResultsPtr = &CallGetNumResults;
        mt->CallGetResultPtr = &CallGetResult;

        mt->Triton_LoadPtr = &Triton_Load;
        mt->Triton_StorePtr = &Triton_Store;

        mt->SequentialGetParametersCountPtr = &SequentialGetParametersCount;
        mt->SequentialGetFieldsCountPtr = &SequentialGetFieldsCount;
        mt->SequentialGetParameterPtr = &SequentialGetParameter;
        mt->SequentialHasTerminatorPtr = &SequentialHasTerminator;
        mt->SequentialInsertAtPtr = &SequentialInsertAt;

        mt->PrimFunctionCreatePtr = &PrimFunctionCreate;
        mt->PrimFunctionAddBodyPtr = &PrimFunctionAddBody;
        mt->PrimFunctionGetBodyPtr = &PrimFunctionGetBody;

        mt->TupleCreatePtr = &TupleCreate;

        mt->BaseExprPrintPtr = &BaseExprPrint;
        mt->IRModuleGetEntryNamePtr = &IRModuleGetEntryName;
        mt->IRModuleDescribeVectorAddPtr = &IRModuleDescribeVectorAdd;
    }

    [UnmanagedCallersOnly]
    private static void CompilerInitialize()
    {
        var host = Host.CreateDefaultBuilder()
            .ConfigureCompiler()
            .Build();
        CompilerServices.Configure(host.Services);
    }

    [UnmanagedCallersOnly]
    private static nuint LastErrorGet(byte* buffer, nuint bufferLength) =>
        WriteUtf8(_lastError ?? string.Empty, buffer, bufferLength);

    private static void ClearLastError() => _lastError = null;

    private static byte SetLastError(Exception ex)
    {
        _lastError = ex.ToString();
        return 0;
    }

    [UnmanagedCallersOnly]
    private static IntPtr TargetCreate(byte* targetNamePtr, nuint targetNameLength)
    {
        var targetName = ToString(targetNamePtr, targetNameLength);
        return GCHandle.ToIntPtr(GCHandle.Alloc(CompilerServices.GetTarget(targetName)));
    }

    private static T Get<T>(IntPtr handle)
    {
        return (T)(GCHandle.FromIntPtr(handle).Target ?? throw new ArgumentNullException(nameof(handle)));
    }

    private static T? GetNullable<T>(IntPtr handle)
    {
        if (handle == IntPtr.Zero)
        {
            return default;
        }

        return (T?)GCHandle.FromIntPtr(handle).Target;
    }

    private static string ToString(byte* bytes, nuint length) =>
        Encoding.UTF8.GetString(bytes, (int)length);

    private static T[] To1DArray<T>(T* value, nuint shape0)
        where T : unmanaged
    {
        var arr = new T[shape0];
        for (nuint i = 0; i < shape0; i++)
        {
            arr[i] = value[i];
        }

        return arr;
    }

    private static T[][] To2DArray<T>(T* value, nuint shape0, nuint* shape1)
        where T : unmanaged
    {
        var arr = new T[shape0][];
        int count = 0;
        for (nuint i = 0; i < shape0; i++)
        {
            arr[i] = new T[shape1[i]];
            for (nuint j = 0; j < shape1[i]; j++)
            {
                arr[i][j] = value[count++];
            }
        }

        return arr;
    }

    private static int[] StringToArrayInt32(string value)
    {
        var data = value.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(",");
        return Array.ConvertAll(data, int.Parse);
    }

    private static float[] StringToArrayFloat(string value)
    {
        var data = value.Replace(" ", string.Empty, StringComparison.OrdinalIgnoreCase).Split(',');
        return Array.ConvertAll(data, float.Parse);
    }
}
