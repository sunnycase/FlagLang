// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using CommunityToolkit.HighPerformance;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.IR.Logics;
using Nncase.IR.Math;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Compiler.Interop;

public static unsafe partial class CApi
{
    public static string DescribeVectorAddModuleForDiagnostics(IR.IRModule module) =>
        JsonSerializer.Serialize(DescribeVectorAddModule(module));

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
    private static nuint BaseExprPrint(IntPtr exprPtr, byte* buffer, nuint bufferLength)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var text = expr is IR.IRModule module
            ? PrintModule(module)
            : SafePrint(expr, PrintFlagsFor(expr));
        return WriteUtf8(text, buffer, bufferLength);
    }

    [UnmanagedCallersOnly]
    private static nuint IRModuleGetEntryName(IntPtr modulePtr, byte* buffer, nuint bufferLength)
    {
        var module = Get<IR.IRModule>(modulePtr);
        var name = SelectEntryBaseFunction(module)?.Name ?? string.Empty;
        return WriteUtf8(name, buffer, bufferLength);
    }

    [UnmanagedCallersOnly]
    private static nuint IRModuleDescribeVectorAdd(IntPtr modulePtr, byte* buffer, nuint bufferLength)
    {
        var module = Get<IR.IRModule>(modulePtr);
        var result = DescribeVectorAddModule(module);
        return WriteUtf8(JsonSerializer.Serialize(result), buffer, bufferLength);
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

    private static nuint WriteUtf8(string text, byte* buffer, nuint bufferLength)
    {
        var bytes = Encoding.UTF8.GetBytes(text);
        if (buffer != null && bufferLength > 0)
        {
            var copyLength = Math.Min((int)bufferLength, bytes.Length);
            bytes.AsSpan(0, copyLength).CopyTo(new Span<byte>(buffer, copyLength));
        }

        return (nuint)bytes.Length;
    }

    private static Dictionary<string, object?> DescribeVectorAddModule(IR.IRModule module)
    {
        try
        {
            var descriptor = BuildVectorAddDescriptor(module);
            return new()
            {
                ["valid"] = true,
                ["descriptor"] = descriptor,
                ["text"] = SafePrint(module, Diagnostics.PrinterFlags.Script | Diagnostics.PrinterFlags.Normal),
            };
        }
        catch (Exception ex)
        {
            return new()
            {
                ["valid"] = false,
                ["reason"] = ex.Message,
                ["text"] = SafePrint(module, Diagnostics.PrinterFlags.Script | Diagnostics.PrinterFlags.Normal),
            };
        }
    }

    private static string SafePrint(BaseExpr expr, Diagnostics.PrinterFlags flags)
    {
        try
        {
            return CompilerServices.Print(expr, flags);
        }
        catch (Exception ex)
        {
            return $"{expr.GetType().FullName}: print failed: {ex.Message}";
        }
    }

    private static string PrintModule(IR.IRModule module)
    {
        var functions = module.Functions.ToArray();
        if (functions.Length == 0)
        {
            return SafePrint(module, Diagnostics.PrinterFlags.Script | Diagnostics.PrinterFlags.Normal);
        }

        return string.Join(
            Environment.NewLine + Environment.NewLine,
            functions.Select(function => SafePrint(function, PrintFlagsFor(function))));
    }

    private static Diagnostics.PrinterFlags PrintFlagsFor(BaseExpr expr)
    {
        return expr is TIR.PrimFunction
            ? Diagnostics.PrinterFlags.Script | Diagnostics.PrinterFlags.Normal
            : Diagnostics.PrinterFlags.Detailed;
    }

    private static Dictionary<string, object?> BuildVectorAddDescriptor(IR.IRModule module)
    {
        var function = SelectEntryFunction(module)
            ?? throw new InvalidOperationException("Module does not contain a lowered Function entry.");
        var parameters = function.Parameters.ToArray();
        if (parameters.Length != 4)
        {
            throw new InvalidOperationException($"Vector-add module must have four parameters, got {parameters.Length}.");
        }

        for (int i = 0; i < 3; i++)
        {
            RequirePointerF32(parameters[i], $"parameter {i}");
        }

        RequireProblemSize(parameters[3]);

        var calls = ExprCollector.Collect(function.Body.Body).OfType<Call>().ToArray();
        var gatherCalls = calls.Where(call => call.Target is IR.Affine.Gather).ToArray();
        if (gatherCalls.Length != 2)
        {
            throw new InvalidOperationException($"Vector-add module must contain exactly two affine gathers, got {gatherCalls.Length}.");
        }

        var xGather = RequireSingleGather(gatherCalls, parameters[0], "x");
        var yGather = RequireSingleGather(gatherCalls, parameters[1], "y");
        var addCalls = calls.Where(call => call.Target is Binary { BinaryOp: BinaryOp.Add }).ToArray();
        if (addCalls.Length != 1)
        {
            throw new InvalidOperationException($"Vector-add module must contain exactly one floating add, got {addCalls.Length}.");
        }

        var addCall = RequireVectorAdd(addCalls[0], xGather, yGather);
        var scatterCalls = calls.Where(call => call.Target is IR.Affine.Scatter).ToArray();
        if (scatterCalls.Length != 1)
        {
            throw new InvalidOperationException($"Vector-add module must contain exactly one affine scatter, got {scatterCalls.Length}.");
        }

        var scatter = RequireSingleScatter(scatterCalls[0], addCall, parameters[2]);

        var xGatherOp = (IR.Affine.Gather)xGather.Target;
        var yGatherOp = (IR.Affine.Gather)yGather.Target;
        var scatterOp = (IR.Affine.Scatter)scatter.Target;
        var blockSize = ValidateRelation(xGatherOp.Relation, xGatherOp.Symbols, xGatherOp.Shape, parameters[3], "x gather");
        var yBlockSize = ValidateRelation(yGatherOp.Relation, yGatherOp.Symbols, yGatherOp.Shape, parameters[3], "y gather");
        var scatterBlockSize = ValidateRelation(scatterOp.Relation, scatterOp.Symbols, null, parameters[3], "scatter");
        if (yBlockSize != blockSize || scatterBlockSize != blockSize)
        {
            throw new InvalidOperationException("Vector-add affine IO operations do not share the same lane block size.");
        }

        var xDefault = DescribeDefaultValue(xGather[IR.Affine.Gather.DefaultValue]);
        var yDefault = DescribeDefaultValue(yGather[IR.Affine.Gather.DefaultValue]);
        var parameterOrder = parameters.Select(p => p.Name).ToArray();
        var parameterTypes = parameters.Select(p => ((Expr)p).CheckedType.ToString()).ToArray();
        return new()
        {
            ["kind"] = "flaglang.vector_add",
            ["version"] = 1,
            ["ir_source"] = "post_ttir_native_module",
            ["entry_name"] = function.Name,
            ["parameter_order"] = parameterOrder,
            ["parameter_types"] = parameterTypes,
            ["pointers"] = parameterOrder.Take(3).ToArray(),
            ["n_elements_arg"] = parameterOrder[3],
            ["block_size"] = blockSize,
            ["dtype"] = "float32",
            ["element_size"] = 4,
            ["program_id_axis"] = 0,
            ["lane_domain"] = $"0 <= d0 < {blockSize}",
            ["relation"] = $"s0 * {blockSize} + d0",
            ["constraint"] = $"s0 * {blockSize} + d0 < s1",
            ["loads"] = new[]
            {
                new Dictionary<string, object?> { ["source"] = parameterOrder[0], ["default"] = xDefault },
                new Dictionary<string, object?> { ["source"] = parameterOrder[1], ["default"] = yDefault },
            },
            ["compute"] = "fadd",
            ["store"] = new Dictionary<string, object?> { ["dest"] = parameterOrder[2] },
        };
    }

    private static BaseFunction? SelectEntryBaseFunction(IR.IRModule module)
    {
        if (module.Entry is not null)
        {
            return module.Entry;
        }

        var functions = module.Functions.ToArray();
        return functions.Length == 1 ? functions[0] : null;
    }

    private static Function? SelectEntryFunction(IR.IRModule module)
    {
        if (module.Entry is Function entry)
        {
            return entry;
        }

        var functions = module.Functions.ToArray().OfType<Function>().ToArray();
        return functions.Length == 1 ? functions[0] : null;
    }

    private static void RequirePointerF32(IVar parameter, string role)
    {
        if (((Expr)parameter).CheckedType is not TensorType { DType: PointerType { ElemType: var elemType }, Shape.IsScalar: true }
            || !Equals(elemType, DataTypes.Float32))
        {
            throw new InvalidOperationException($"Vector-add {role} must be a scalar float32 pointer, got {((Expr)parameter).CheckedType}.");
        }
    }

    private static void RequireProblemSize(IVar parameter)
    {
        if (((Expr)parameter).CheckedType is not TensorType { Shape.IsScalar: true } tensorType
            || (!Equals(tensorType.DType, DataTypes.Int32) && !Equals(tensorType.DType, DataTypes.UInt32)))
        {
            throw new InvalidOperationException($"Vector-add problem-size parameter must be scalar i32/u32, got {((Expr)parameter).CheckedType}.");
        }
    }

    private static Call RequireSingleGather(IReadOnlyList<Call> gathers, IVar source, string role)
    {
        var matches = gathers.Where(call => ReferenceEquals(call[IR.Affine.Gather.Source], source)).ToArray();
        return matches.Length == 1
            ? matches[0]
            : throw new InvalidOperationException($"Vector-add module must contain one affine gather for {role} input, got {matches.Length}.");
    }

    private static Call RequireVectorAdd(Call addCall, Call xGather, Call yGather)
    {
        if (addCall.Target is Binary { BinaryOp: BinaryOp.Add }
            && ReferenceEquals(addCall[Binary.Lhs], xGather)
            && ReferenceEquals(addCall[Binary.Rhs], yGather))
        {
            return addCall;
        }

        throw new InvalidOperationException("Vector-add module must contain one floating add from the two affine gathers.");
    }

    private static Call RequireSingleScatter(Call scatter, Call addCall, IVar dest)
    {
        if (scatter.Target is IR.Affine.Scatter
            && ReferenceEquals(scatter[IR.Affine.Scatter.Source], addCall)
            && ReferenceEquals(scatter[IR.Affine.Scatter.Dest], dest))
        {
            return scatter;
        }

        throw new InvalidOperationException("Vector-add module must contain one affine scatter of the add result to the output pointer.");
    }

    private static int ValidateRelation(AffineRelation relation, RankedShape symbols, Shape? shape, IVar problemSize, string role)
    {
        if (relation.Domains.Length != 1 || relation.Symbols.Length != 2 || relation.Results.Length != 1)
        {
            throw new InvalidOperationException($"Vector-add {role} relation must have one domain, two symbols, and one result.");
        }

        if (relation.Domains[0].Metadata.Range is not { Min: 0, Max: >= 0 } range || range.Max % 1 != 0)
        {
            throw new InvalidOperationException($"Vector-add {role} lane domain must have range [0, BLOCK_SIZE - 1].");
        }

        var blockSize = checked((int)range.Max + 1);
        var expectedWithoutConstraint = $"(d0)[s0, s1] -> (((s0 * {blockSize}) + d0))";
        if (relation.With(constraint: LogicalExpr.True).ToString() != expectedWithoutConstraint)
        {
            throw new InvalidOperationException($"Vector-add {role} relation does not match program_id(0) * BLOCK_SIZE + d0.");
        }

        if (symbols.Count != 2
            || symbols[0] is not ProgramIdDim { Axis: 0 }
            || symbols[1] is not DimVar problemSymbol
            || problemSymbol.Name != problemSize.Name)
        {
            throw new InvalidOperationException($"Vector-add {role} symbols must be program_id(0) and the problem-size parameter.");
        }

        ValidateConstraint(relation.Constraint, symbols[0], symbols[1], blockSize, role);

        if (shape is not null)
        {
            if (shape is not RankedShape rankedShape
                || rankedShape.Rank != 1
                || !rankedShape[0].IsFixed
                || rankedShape[0].FixedValue != blockSize)
            {
                throw new InvalidOperationException($"Vector-add {role} shape must be a one-dimensional BLOCK_SIZE vector.");
            }
        }

        return blockSize;
    }

    private static void ValidateConstraint(LogicalExpr constraint, Dimension programId, Dimension problemSize, int blockSize, string role)
    {
        if (constraint is not DimCompare { Op: CompareOp.LowerThan } compare
            || !ReferenceEquals(compare.Rhs, problemSize)
            || !IsProgramBlockPlusLane(compare.Lhs, programId, blockSize))
        {
            throw new InvalidOperationException($"Vector-add {role} relation must be guarded exactly by program_id(0) * BLOCK_SIZE + d0 < problem-size.");
        }
    }

    private static bool IsProgramBlockPlusLane(Dimension value, Dimension programId, int blockSize)
    {
        var (terms, bias) = value is DimSum sum
            ? (sum.Operands.ToArray(), sum.Bias)
            : (new[] { value }, 0L);

        return bias == 0
            && terms.Length == 2
            && terms.Any(term => IsScaledProgramId(term, programId, blockSize))
            && terms.Any(IsLaneDomain);
    }

    private static bool IsScaledProgramId(Dimension value, Dimension programId, int blockSize)
    {
        if (blockSize == 1 && ReferenceEquals(value, programId))
        {
            return true;
        }

        return value is DimProduct { Count: 1, Scale: var scale, Operands: [var operand] }
            && scale == blockSize
            && ReferenceEquals(operand, programId);
    }

    private static bool IsLaneDomain(Dimension value)
    {
        return value is DimVar { Name: "d0", Metadata.Range: { Min: 0, Max: >= 0 } };
    }

    private static string DescribeDefaultValue(BaseExpr defaultValue)
    {
        if (defaultValue is IR.None)
        {
            return "implicit_zero";
        }

        if (defaultValue is TensorConst tensorConst && tensorConst.Value.BytesBuffer.ToArray().All(b => b == 0))
        {
            return "implicit_zero";
        }

        throw new InvalidOperationException("Vector-add masked loads must use implicit-zero defaults.");
    }
}
