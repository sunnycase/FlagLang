// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

#define MULTI_CORE_XPU

// #define DEBUG_PRINT
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Runtime;
using Nncase.TIR;
using Nncase.Utilities;
using Razor.Templating.Core;

namespace Nncase.CodeGen.NTT;

public class DeviceCSourceConvertVisitor : CSourceConvertVisitor
{
    protected readonly StringBuilder _deviceBuilder;
    private readonly Dictionary<MemSpan, TensorViewAccess> _tensorViewAccesses = new(ReferenceEqualityComparer.Instance);

    public DeviceCSourceConvertVisitor()
    {
        _deviceBuilder = new();
    }

    public static void WriteWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.Write($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.Write($"profile_scope profiler(function_name, profile_level::device);\n");
#endif
        IndentScope.Writer.Write($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public static void WriteIndWithProfiler(string functionName, string tagName = "")
    {
        functionName = functionName.TrimEnd(new char[] { ';', '\n' });
        if (tagName == string.Empty)
        {
            int index = functionName.IndexOf('(', StringComparison.Ordinal);
            if (index != -1)
            {
                tagName = functionName.Substring(0, index);
            }
        }

        tagName = tagName == string.Empty ? functionName : tagName;
        IndentScope.Writer.IndWrite("{\n");
#if false // Disable device profiling for now.
        IndentScope.Writer.IndWrite($"constexpr std::string_view function_name = \"{tagName}\";\n");
        IndentScope.Writer.IndWrite($"profile_scope profiler(function_name, profile_level::device);\n");
#endif
        IndentScope.Writer.IndWrite($"{functionName};\n");
        IndentScope.Writer.IndWrite("}\n");
    }

    public string GetHeader()
    {
        return _deviceBuilder.ToString();
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPrimFunction(PrimFunction expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        if (expr.CheckedType is not CallableType { ReturnType: TupleType r } || r != TupleType.Void)
        {
            throw new NotSupportedException("The PrimFunction must return void!");
        }

        var ctype = $"template<{string.Join(", ", Enumerable.Range(0, expr.Parameters.Length).Select(x => $"class T{x}"))}>" + Environment.NewLine +
            $"NTT_DEVICE void {expr.Name}({string.Join(", ", expr.Parameters.AsValueEnumerable().Select(Visit).Select((s, i) => $"T{i} &&{s.Name}").ToArray())})";

        using (var scope = new IndentScope(_deviceBuilder))
        {
            // 1. Function signature
            IndentScope.Writer.IndWrite($"{ctype} {{\n");

            // 2. Function body
            using (_ = new IndentScope())
            {
                Visit(expr.Body);
            }

            // 3. Function closing
            IndentScope.Writer.IndWrite("}\n");
        }

        symbol = new(ctype, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitIfThenElse(IfThenElse expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var cond = Visit(expr.Condition);
        IndentScope.Writer.IndWrite($"if ({cond.Name}) {{\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Then);
        }

        IndentScope.Writer.IndWrite("}\n");
        IndentScope.Writer.IndWrite("else {\n");
        using (_ = new IndentScope())
        {
            Visit(expr.Else);
        }

        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitLet(Let expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var value = Visit(expr.Expression);
        var @var = expr.Var is Var varExpr
            ? new CSymbol(value.Type, varExpr.Name + "_" + varExpr.GlobalVarIndex.ToString())
            : Visit(expr.Var);
        _exprMemo[(BaseExpr)expr.Var] = new(value.Type, @var.Name);

#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"let {@var.Name}\\n\");\n");
#endif
        if (value.Type.StartsWith("array"))
        {
            var ss = value.Type.Split(" ");
            IndentScope.Writer.IndWrite($"{ss[1]} {@var.Name}[{ss[2]}];\n");
        }
        else
        {
            IndentScope.Writer.IndWrite($"{value.Type} {@var.Name} = {value.Name};\n");
        }

        TIR.Buffer? tensorViewBuffer = null;
        TensorViewAccess oldTensorViewAccess = default;
        var hasOldTensorViewAccess = false;
        if (expr.Expression is Call { Target: IR.Buffers.AllocateBufferView } allocateView &&
            allocateView.Arguments[0] is TIR.Buffer buffer)
        {
            tensorViewBuffer = buffer;
            hasOldTensorViewAccess = _tensorViewAccesses.TryGetValue(buffer.MemSpan, out oldTensorViewAccess);
            _tensorViewAccesses[buffer.MemSpan] = new(@var.Name, buffer.Type is DistributedType);
        }

        try
        {
            Visit(expr.Body);
        }
        finally
        {
            if (tensorViewBuffer is not null)
            {
                if (hasOldTensorViewAccess)
                {
                    _tensorViewAccesses[tensorViewBuffer.MemSpan] = oldTensorViewAccess;
                }
                else
                {
                    _tensorViewAccesses.Remove(tensorViewBuffer.MemSpan);
                }
            }
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitPhysicalBuffer(PhysicalBuffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var start = Visit(expr.Start);
        var size = Visit(expr.Size);
        var storage = expr.Storage.WithoutAlignment();
        string name = storage switch
        {
            { Scope: BufferScope.ThreadLocal, PhysicalLocation: PhysicalMemorySpace.LocalAddressable } => $"tar::get_cache_address<{expr.Hierarchy}>()",
            { Usage: BufferUsage.Input or BufferUsage.Output, Scope: BufferScope.Device, PhysicalLocation: PhysicalMemorySpace.GMem } => start.Name,
            _ => throw new NotSupportedException($"Unsupported physical buffer storage for NTT device codegen: {expr.Storage}"),
        };

        var str = $"ntt::span<std::byte, {size.Name}>({name} + {start.Name}, {size.Name})";
        symbol = new(start.Type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitMemSpan(MemSpan expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var buffer = Visit(expr.Buffer);
        var start = Visit(expr.Start);
        var size = Visit(expr.Size);

        var str = $"make_subspan({buffer.Name}, {start.Name}, {size.Name})";
        symbol = new(start.Type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBuffer(TIR.Buffer expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var distributedType = expr.Type as DistributedType;
        var dimensions = distributedType is null ? expr.Dimensions : ((RankedShape)distributedType.TensorType.Shape).Dimensions;
        var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
        var isFixedStrides = expr.Strides.AsValueEnumerable().All(x => x.IsFixed);
        var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
        var strideSymbols = expr.Strides.AsValueEnumerable().Select(Visit).ToArray();

        var dtypeStr = expr.ElemType.ToC();
        var dimensionStr = KernelUtility.DimensionsToC(isFixedDimensions, dimensionSymbols, true);
        var strideStr = KernelUtility.StridesToC(isFixedStrides, strideSymbols, true);
        var type = $"tensor_view<{dtypeStr}, {dimensionStr}, {strideStr}> ";

        symbol = new(type, expr.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitCall(Call expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type = expr.CheckedType switch
        {
            TupleType x when x == TupleType.Void => string.Empty,
            TensorType { IsScalar: true } x => x.DType switch
            {
                ReferenceType => "auto",
                _ => x.DType.ToC(),
            },
            TensorType or DistributedType or DimensionType => "auto",
            _ => throw new NotSupportedException(),
        };

        string str = string.Empty;
        var arguments = expr.Arguments.AsValueEnumerable().Select(Visit).ToArray();
        switch (expr.Target)
        {
            case PrimFunction deviceFunc:
                WriteIndWithProfiler($"{deviceFunc.Name}({string.Join(",", arguments.Select(arg => arg.Name))});\n");
                break;
            case IR.Math.Binary op:
                str = CSourceUtilities.ConvertBinary(op, arguments);
                break;
            case IR.Math.Unary op:
                str = CSourceUtilities.ConvertUnary(op, arguments);
                break;
            case IR.Math.Compare op:
                str = CSourceUtilities.ConvertCompare(op, arguments);
                break;
            case IR.Math.Select op:
                str = CSourceUtilities.ConvertSelect(op, arguments);
                break;
            case IR.Shapes.AsTensor op:
                str = arguments[0].Name;
                break;
            case IR.Tensors.LocalShardDim op:
                str = $"local_shard_dim<0>(make_sharding<{op.Placement.PlacementToC()}>({op.AxisPolicy.SBPToC()}), make_shape({arguments[0].Name}))";
                break;
            case TIR.NTT.SramPtr op:
                str = $"g_cpu_mt->sram_address(bid, tid) + {arguments[0].Name}";
                break;
            case TIR.Load op:
                str = TryGetTensorViewAccess(expr.Arguments[0], out var loadAccess)
                    ? $"{FormatTensorViewAccess(loadAccess)}({arguments[1].Name})"
                    : $"{arguments[0].Name}[{arguments[1].Name}]";
                break;
            case TIR.Store op:
#if DEBUG_PRINT
                IndentScope.Writer.IndWrite($"runtime_util->printf(\"{arguments[0].Name}[%d]\\n\", {arguments[1].Name});\n");
#endif
                var storeTarget = TryGetTensorViewAccess(expr.Arguments[0], out var storeAccess)
                    ? $"{FormatTensorViewAccess(storeAccess)}({arguments[1].Name})"
                    : $"{arguments[0].Name}[{arguments[1].Name}]";
                IndentScope.Writer.IndWrite($"{storeTarget} = {arguments[2].Name};\n");
                break;
            case TIR.NTT.PtrOf op:
                str = op.PtrName + ".data()";
                break;
            case IR.Buffers.Allocate op:
                if (op.Malloc)
                {
                    str = $"({type})runtime_util->malloc({arguments[0].Name})";
                }
                else
                {
                    type = $"array {((PointerType)expr.CheckedDataType).ElemType.ToC()} {arguments[0].Name}";
                    str = $"";
                }

                break;
            case IR.Buffers.BufferSubview op:
                {
                    var arg0 = VisitDimOrShape(expr.Arguments[1], CShapeKind.Shape).Name;
                    var arg1 = VisitDimOrShape(expr.Arguments[2], CShapeKind.Shape).Name;
                    str = expr.Arguments[0].CheckedType switch
                    {
                        DistributedType when expr.CheckedType is TensorType => $"{arguments[0].Name}.local().view({arg0}, {arg1})",
                        DistributedType => throw new NotSupportedException($"Distributed BufferSubview must lower to a local TensorType view, got {expr.CheckedType}."),
                        TensorType => $"{arguments[0].Name}.view({arg0}, {arg1})",
                        var parentType => throw new NotSupportedException($"BufferSubview codegen expects TensorType or DistributedType parent, got {parentType}."),
                    };
                }

                break;
            case IR.Buffers.AllocateBufferView op:
                {
                    var buffer = (TIR.Buffer)expr.Arguments[0];
                    var distributedType = buffer.Type as DistributedType;
                    var dimensions = distributedType is null ? buffer.Dimensions : ((RankedShape)distributedType.TensorType.Shape).Dimensions;
                    var isFixedDimensions = dimensions.AsValueEnumerable().All(x => x.IsFixed);
                    var isFixedStrides = buffer.Strides.AsValueEnumerable().All(x => x.IsFixed);
                    var dimensionSymbols = dimensions.AsValueEnumerable().Select(Visit).ToArray();
                    var strideSymbols = buffer.Strides.AsValueEnumerable().Select(Visit).ToArray();

                    var dtypeStr = buffer.ElemType.ToC();
                    var dimensionStrs = dimensionSymbols.Select(x => x.Name);
                    var strideStrs = strideSymbols.Select(x => x.Name);
                    var spanStr = GetTypedSpan(dtypeStr, Visit(buffer.MemSpan));
                    str = distributedType is null
                        ? $"make_tensor_view({spanStr}, make_shape({StringUtility.Join(", ", dimensionStrs)}), make_strides({StringUtility.Join(", ", strideStrs)}))"
                        : $"make_sharded_tensor_view({spanStr}, make_shape({StringUtility.Join(", ", dimensionStrs)}), {KernelUtility.ShardingToC(distributedType)}, make_strides({StringUtility.Join(", ", strideStrs)}))";
                }

                break;
            case IR.Tensors.Cast op:
                str = ConvertCast(op, arguments[0]);
                break;
            case IR.Tensors.Depend:
                str = arguments[1].Name;
                break;
            case TIR.Memcopy op:
                WriteIndWithProfiler($"tensor_copy_sync({arguments[1].Name}, {arguments[0].Name});\n");
                break;
            case TIR.NTT.Unary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unary.cshtml", new UnaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    UnaryOp = op.UnaryOp,
                }).Result);
                break;
            case TIR.NTT.VectorizedBinary op:
                WriteIndWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Binary.cshtml", new BinaryKernelTemplateModel
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    BinaryOp = op.BinaryOp,
                }).Result);
                break;
            case TIR.NTT.Swish swish:
                if (swish.Beta == 1.0f)
                {
                    WriteIndWithProfiler($"unary<ops::swish>({arguments[0].Name}, {arguments[1].Name});\n");
                }
                else
                {
                    IndentScope.Writer.IndWrite($"\n{{\nauto b= {swish.Beta}; auto tb = make_tensor_view_from_address<float>(&b, fixed_shape_v<>);\n");
                    WriteIndWithProfiler($"binary<ops::swishb>({arguments[0].Name}, tb, {arguments[1].Name});\n}}\n");
                }

                break;
            case TIR.NTT.Matmul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Matmul.cshtml", new TypedKernelTemplateModel<TIR.NTT.Matmul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.PackedMatMul matmul:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/PackedMatMul.cshtml", new TypedKernelTemplateModel<TIR.NTT.PackedMatMul>(matmul)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);

                break;
            case TIR.NTT.Pack vectorize:
                WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Pack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Pack>(vectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Transpose transpose:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Transpose.cshtml", new TypedKernelTemplateModel<TIR.NTT.Transpose>(transpose)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Unpack devectorize:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Unpack.cshtml", new TypedKernelTemplateModel<TIR.NTT.Unpack>(devectorize)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.Reduce reduce:
                IndentScope.Writer.Write(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Reduce.cshtml", new TypedKernelTemplateModel<TIR.NTT.Reduce>(reduce)
                {
                    Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                    Indent = new string(' ', IndentScope.Writer.Indent),
                }).Result);
                break;
            case TIR.NTT.RoPE rope:
                WriteIndWithProfiler($"rope({arguments[0].Name}, {arguments[1].Name}, {arguments[2].Name}, {arguments[3].Name});\n");
                break;
            case TIR.NTT.Cast cast:
                {
                    string postOps = string.Empty;
                    if (expr[TIR.NTT.Cast.PostOps] is Fusion lambda)
                    {
                        postOps = $"<{lambda.Name}>";
                    }

                    IndentScope.Writer.IndWrite($"cast{postOps}({arguments[0].Name}, {arguments[1].Name}, {FixedShapeValue(cast.VectorizeAxes.ToArray())});\n");
                }

                break;
            case TIR.NTT.VectorizedLayerNorm lm:
                {
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/VectorizedLayerNorm.cshtml", new TypedKernelTemplateModel<TIR.NTT.VectorizedLayerNorm>(lm)
                    {
                        Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).Concat(lm.PadedNums.Select(Visit).Select(x => new KernelArgument { Symbol = x })).ToArray(),
                        Indent = new string(' ', IndentScope.Writer.Indent),
                        Args = expr.Arguments[..1].ToArray(),
                    }).Result);
                }

                break;
            case TIR.NTT.Pad pad:
                {
                    var padValueType = expr.Arguments[0].CheckedTensorType.DType is VectorType vt ? vt.ElemType : expr.Arguments[0].CheckedTensorType.DType;
                    WriteWithProfiler($"pad({arguments[0].Name}, {arguments[2].Name}, {arguments[1].Name}, {expr.Arguments[0].CheckedDataType.ToC()} {{ ({padValueType.ToC()}){pad.PadValue} }});\n");
                }

                break;
            case TIR.NTT.Where where:
                WriteWithProfiler($"where({arguments[0].Name}, {arguments[1].Name}, {arguments[2].Name}, {arguments[3].Name});\n");
                break;
            case TIR.NTT.GetPositionIds getPositionIds:
                WriteIndWithProfiler($"get_position_ids({arguments[0].Name}, {arguments[1].Name}, {KernelUtility.ShardingToC(getPositionIds.DistributedType)}, {Visit(getPositionIds.DistributedType.TensorType.Shape).Name});\n");
                break;
            case TIR.NTT.Compare compare:
                {
                    WriteWithProfiler(RazorTemplateEngine.RenderAsync("~/CodeGen/CPU/Templates/Kernels/Compare.cshtml", new CompareKernelTemplateModel
                    {
                        Arguments = arguments.Select(x => new KernelArgument { Symbol = x }).ToArray(),
                        CompareOp = compare.CompareOp,
                    }).Result);
                }

                break;

            default:
                throw new NotSupportedException($"Unsupported call target: {expr.Target}");
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitConst(Const expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        string type;
        string str;
        if (expr is TensorConst { Value: Tensor { ElementType: PrimType ptype, Shape: { IsScalar: true } } scalar })
        {
            str = scalar[Array.Empty<long>()].ToString() switch
            {
                "True" => "1",
                "False" => "0",
                null => string.Empty,
                var x => x,
            };

            type = ptype.ToC();
        }
        else if (expr is TensorConst { Value: Tensor { ElementType: PointerType { ElemType: DataType }, Shape: { IsScalar: true } } pointer })
        {
            str = pointer.ToScalar<ulong>().ToString();
            type = pointer.ElementType.ToC();
        }
        else
        {
            throw new NotSupportedException();
        }

        symbol = new(type, str);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitTupleConst(TupleConst tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Value.Select(x => Visit(Const.FromValue(x)).Name))}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitTuple(IR.Tuple tp)
    {
        if (_exprMemo.TryGetValue(tp, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = $"{string.Join(",", tp.Fields.AsValueEnumerable().Select(x => Visit(x).Name).ToArray())}";
        symbol = new(type, str);
        _exprMemo.Add(tp, symbol);
        return symbol;
    }

    protected override CSymbol VisitFusion(Fusion fusion)
    {
        if (_exprMemo.TryGetValue(fusion, out var symbol))
        {
            return symbol;
        }

        string type = string.Empty;
        string str = fusion.Name;
        symbol = new(type, str);
        _exprMemo.Add(fusion, symbol);
        return symbol;
    }

    /// <inheritdoc/>
    protected override CSymbol VisitSequential(Sequential expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        foreach (var field in expr.Fields)
        {
            Visit(field);
        }

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitFor(For expr)
    {
        if (expr.Mode == LoopMode.Unrolled)
        {
            return VisitUnrolledFor(expr);
        }

        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // 1. For Loop signature
        var loopVar = Visit(expr.LoopVar);
        IndentScope.Writer.IndWrite($"for ({loopVar.Type} {loopVar.Name} = {Visit(expr.Domain.Start).Name}; {loopVar.Name} < {Visit(expr.Domain.Stop).Name}; {loopVar.Name} += {Visit(expr.Domain.Step).Name}) {{\n");
#if DEBUG_PRINT
        IndentScope.Writer.IndWrite($"runtime_util->printf(\"{loopVar.Name} = %d\\n\", {loopVar.Name});\n");
#endif

        using (_ = new IndentScope())
        {
            // 2. For Body
            Visit(expr.Body);
        }

        // 3. For closing
        IndentScope.Writer.IndWrite("}\n");

        symbol = new(string.Empty, string.Empty);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitVar(Var expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var name = IRHelpers.GetIdentityName(expr.Name);
        var index = VisitEntry.Parameters.IndexOf(expr);
        if (index != -1)
        {
            symbol = new CSymbol($"T{index}", name);
        }
        else
        {
            symbol = new(
                expr.CheckedType switch
                {
                    TensorType t => t.DType.ToC(),
                    AnyType => "auto",
                    _ => throw new ArgumentOutOfRangeException(nameof(expr)),
                },
                expr.Name + "_" + expr.GlobalVarIndex.ToString());
        }

        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitAsDim(AsDim expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        var value = Visit(UnwrapDimValue(expr.Dim));
        symbol = new("dim_t", value.Name);
        _exprMemo.Add(expr, symbol);
        return symbol;
    }

    protected override CSymbol VisitBufferRegion(BufferRegion expr)
    {
        if (_exprMemo.TryGetValue(expr, out var symbol))
        {
            return symbol;
        }

        // FIXME: Use extents instead of stop in BufferRegion.
        throw new NotImplementedException();
#if false
        var buffer = Visit(expr.Buffer);
        var begins = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Start).Name))}";
        var extents = $"{StringUtility.Join(", ", expr.Region.AsValueEnumerable().Select(x => Visit(x.Stop).Name))}";
        symbol = new(string.Empty, $"{buffer.Name}.view(make_shape({begins}), make_shape({extents}))");
        _exprMemo.Add(expr, symbol);
        return symbol;
#endif
    }

    private static string ConvertCast(IR.Tensors.Cast op, CSymbol input)
    {
        if (op is { CastMode: CastMode.Reinterpret, NewType: PointerType { ElemType: DataType elemType } })
        {
            return GetTypedSpan(elemType.ToC(), input);
        }

        return $"(({op.NewType.ToC()}){input.Name})";
    }

    private static string GetTypedSpan(string elemType, CSymbol input) =>
        IsTypedSpan(input.Type, elemType)
            ? input.Name
            : $"typed_span_reinterpret<{elemType}>({input.Name})";

    private static bool IsTypedSpan(string type, string elemType)
    {
        var trimmed = type.Trim();
        return trimmed.StartsWith($"ntt::span<{elemType},", StringComparison.Ordinal);
    }

    private static string FormatTensorViewAccess(TensorViewAccess access) =>
        access.IsDistributed ? $"{access.Name}.local()" : access.Name;

    private static string FixedShapeValue(IReadOnlyList<int> dims) =>
        dims.Count == 0 ? "shape_t<>{}" : $"fixed_shape_v<{string.Join(",", dims)}>";

    private static BaseExpr UnwrapDimValue(BaseExpr value)
    {
        while (value is Call call)
        {
            var unwrapped = call.Target switch
            {
                IR.Tensors.Cast => call[IR.Tensors.Cast.Input],
                IR.Shapes.AsTensor => call[IR.Shapes.AsTensor.Input],
                _ => call,
            };
            if (ReferenceEquals(unwrapped, value))
            {
                break;
            }

            value = unwrapped;
        }

        return value;
    }

    private bool TryGetTensorViewAccess(BaseExpr handle, out TensorViewAccess access)
    {
        while (handle is Call { Target: IR.Tensors.Cast { CastMode: CastMode.Reinterpret } } cast)
        {
            handle = cast.Arguments[0];
        }

        if (handle is MemSpan memSpan)
        {
            return _tensorViewAccesses.TryGetValue(memSpan, out access);
        }

        access = default;
        return false;
    }

    private readonly record struct TensorViewAccess(string Name, bool IsDistributed);
}
