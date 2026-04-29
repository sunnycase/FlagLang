// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using DryIoc;
using NetFabric.Hyperlinq;
using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.TIR;
using Nncase.Utilities;
using OrtKISharp;

namespace Nncase.Evaluator.Math;

/// <summary>
/// Evaluator for <see cref="Binary"/>.
/// </summary>
public partial class BinaryEvaluator : IEvaluator<Binary>,
                                       ITypeInferencer<Binary>,
                                       ICostEvaluator<Binary>,
                                       IOpPrinter<Binary>,
                                       IMetricEvaluator<Binary> {
    public static IRType CheckSBP(
        BinaryOp op,
        TensorType tensorType,
        DistributedType a,
        DistributedType b) {
        // assume broadcast shapes are left algin
        var padA = tensorType.Shape.Rank - a.TensorType.Shape.Rank;
        var padB = tensorType.Shape.Rank - b.TensorType.Shape.Rank;
        var ndsbp = new SBP[tensorType.Shape.Rank];
        for (int i = 0; i < ndsbp.Length; i++) {
            var policyA = i < padA ? null : a.AxisPolicies[i - padA];
            var policyB = i < padB ? null : b.AxisPolicies[i - padB];
            switch (policyA, policyB) {
            case (null, _):
                ndsbp[i] = policyB!;
                break;
            case (_, null):
                ndsbp[i] = policyA!;
                break;
            case (SBPSplit sa, SBPSplit sb):
                if (sa.Axes != sb.Axes) {
                    return new InvalidType($"lhs rhs sbp at {i} not equal");
                }

                ndsbp[i] = sa;
                break;
            case (SBPSplit sa, SBPBroadCast):
                // invalid (S, B) if B is not broacast
                if (b.TensorType.Shape[i - padB] is { IsFixed: false } ||
                    (b.TensorType.Shape[i - padB]
                     is { IsFixed: true, FixedValue: var fb } &&
                     fb != 1)) {
                    return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                }

                ndsbp[i] = sa;
                break;
            case (SBPBroadCast, SBPSplit sb):
                // invalid (B, S) if A is not broacast
                if (a.TensorType.Shape[i - padA] is { IsFixed: false } ||
                    (a.TensorType.Shape[i - padA]
                     is { IsFixed: true, FixedValue: var fa } &&
                     fa != 1)) {
                    return new InvalidType($"lhs rhs sbp at {i} not broadcast");
                }

                ndsbp[i] = sb;
                break;
            case (SBPBroadCast, SBPBroadCast):
                ndsbp[i] = SBP.B;
                break;
            default:
                return new InvalidType("not support binary sbp.");
            }
        }

        if (!DistributedUtility.IsDistributable(ndsbp)) {
            return new InvalidType("not support binary sbp.");
        }

        return new DistributedType(tensorType, ndsbp, a.Placement);
    }

    /// <inheritdoc />
    public IValue Visit(IEvaluateContext context, Binary binary) {
        var lhs = context.GetArgumentValueAsTensor(binary, Binary.Lhs);
        var rhs = context.GetArgumentValueAsTensor(binary, Binary.Rhs);

        if (lhs.ElementType is PointerType || rhs.ElementType is PointerType) {
            if (context.GetReturnType() is not TensorType returnType) {
                throw new InvalidOperationException(
                    "Binary evaluator requires tensor return type for " +
                    "pointer ops");
            }

            return Value.FromTensor(
                EvaluatePointerBinary(binary, lhs, rhs, returnType));
        }

        var originDtype = lhs.ElementType;
        IValue result;
        if (lhs.Shape.IsScalar && rhs.Shape.IsScalar) {
            if (lhs.ElementType == DataTypes.Int32 &&
                rhs.ElementType == DataTypes.Int32) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<int>(),
                    rhs.ToScalar<int>())));
            } else if (lhs.ElementType == DataTypes.Int64 &&
                       rhs.ElementType == DataTypes.Int64) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<long>(),
                    rhs.ToScalar<long>())));
            } else if (lhs.ElementType == DataTypes.Float32 &&
                       rhs.ElementType == DataTypes.Float32) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<float>(),
                    rhs.ToScalar<float>())));
            } else if (lhs.ElementType == DataTypes.Boolean &&
                       rhs.ElementType == DataTypes.Boolean) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<bool>(),
                    rhs.ToScalar<bool>())));
            } else if (lhs.ElementType == DataTypes.UInt32 &&
                       rhs.ElementType == DataTypes.UInt32) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<uint>(),
                    rhs.ToScalar<uint>())));
            } else if (lhs.ElementType == DataTypes.UInt64 &&
                       rhs.ElementType == DataTypes.UInt64) {
                result = Value.FromTensor(Tensor.FromScalar(Compute(
                    binary.BinaryOp,
                    lhs.ToScalar<ulong>(),
                    rhs.ToScalar<ulong>())));
            } else {
                result = Value.FromTensor(
                    Ort_compute(binary, lhs, rhs, originDtype));
            }
        } else {
            // for float16/float8/bfloat16 infere
            var expandOrgDtype = LegalizeLowPrecisionFloat(originDtype);
            lhs = LegalizeLowPrecisionFloat(lhs);
            rhs = LegalizeLowPrecisionFloat(rhs);

            result =
                Value.FromTensor(Ort_compute(binary, lhs, rhs, expandOrgDtype)
                                     .CastTo(originDtype));
        }

        return result;
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Binary target) {
        var lhs = context.CheckArgumentType<IRType>(target, Binary.Lhs);
        var rhs = context.CheckArgumentType<IRType>(target, Binary.Rhs);

        return (lhs, rhs) switch {
            (TensorType a, TensorType b) => Visit(target, a, b),
            (DistributedType a, DistributedType b) => Visit(target, a, b),
            (AnyType, _) => AnyType.Default,
            (_, AnyType) => AnyType.Default,
            _ => new InvalidType($"{lhs} {rhs}"),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Binary target) {
        var lhsType = context.GetArgumentType<IRType>(target, Binary.Lhs);
        var rhsType = context.GetArgumentType<IRType>(target, Binary.Rhs);
        var outputType = context.GetReturnType<IRType>();

        return new() {
            [CostFactorNames.MemoryLoad] =
                CostUtility.GetMemoryAccess(lhsType) +
                CostUtility.GetMemoryAccess(rhsType),
            [CostFactorNames.MemoryStore] =
                CostUtility.GetMemoryAccess(outputType),
            [CostFactorNames.CPUCycles] = CostUtility.GetCPUCycles(
                outputType, CostUtility.GetCPUCyclesOfBinary(target.BinaryOp)),
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Binary target) {
        var lhsType = context.GetArgumentType<TensorType>(target, Binary.Lhs);
        var rhsType = context.GetArgumentType<TensorType>(target, Binary.Rhs);
        var outputType = context.GetReturnType<TensorType>();

        return new() {
            [MetricFactorNames.OffChipMemoryTraffic] =
                CostUtility.GetMemoryAccess(lhsType) +
                CostUtility.GetMemoryAccess(rhsType) +
                CostUtility.GetMemoryAccess(outputType),
            [MetricFactorNames.FLOPs] = MetricUtility.GetFLOPs(
                outputType, (int)MetricUtility.GetBinaryFLOPs(target.BinaryOp)),
            [MetricFactorNames.Parallel] = 4,
        };
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Binary target) {
        var lhs = context.GetArgument(target, Binary.Lhs);
        var rhs = context.GetArgument(target, Binary.Rhs);
        if (context.Flags.HasFlag(PrinterFlags.Inline) ||
            context.Flags.HasFlag(PrinterFlags.Script)) {
            return target.BinaryOp switch {
                BinaryOp.Add => $"({lhs} + {rhs})",
                BinaryOp.Sub => $"({lhs} - {rhs})",
                BinaryOp.Mul => $"({lhs} * {rhs})",
                BinaryOp.Div => $"({lhs} / {rhs})",
                BinaryOp.Mod => $"({lhs} % {rhs})",
                BinaryOp.LogicalAnd => $"({lhs} & {rhs})",
                BinaryOp.LogicalOr => $"({lhs} | {rhs})",
                BinaryOp.LogicalXor => $"({lhs} ^ {rhs})",
                BinaryOp.LeftShift => $"({lhs} << {rhs})",
                BinaryOp.RightShift => $"({lhs} >> {rhs})",
                _ => $"{target.BinaryOp}({lhs}, {rhs})",
            };
        }

        return $"{target.BinaryOp}({lhs}, {rhs})";
    }

    private IRType Visit(Binary target, DistributedType a, DistributedType b) {
        if (a.Placement != b.Placement) {
            return new InvalidType("lhs rhs have different placement");
        }

        var rType = Visit(target, a.TensorType, b.TensorType);
        if (rType is not TensorType tensorType) {
            return rType;
        }

        return CheckSBP(target.BinaryOp, tensorType, a, b);
    }

    private int Compute(BinaryOp op, int a, int b) => op switch {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => FloorDivSigned(a, b),
        BinaryOp.CeilDiv => CeilDivSigned(a, b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => PowChecked(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private uint Compute(BinaryOp op, uint a, uint b) => op switch {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => FloorDivUnsigned(a, b),
        BinaryOp.CeilDiv => CeilDivUnsigned(a, b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => PowChecked(a, b),
        BinaryOp.LeftShift => a << (int)b,
        BinaryOp.RightShift => a >> (int)b,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private ulong Compute(BinaryOp op, ulong a, ulong b) => op switch {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => FloorDivUnsigned(a, b),
        BinaryOp.CeilDiv => CeilDivUnsigned(a, b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => PowChecked(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private bool Compute(BinaryOp op, bool a, bool b) => op switch {
        BinaryOp.LogicalAnd => a & b,
        BinaryOp.LogicalOr => a | b,
        BinaryOp.LogicalXor => a ^ b,
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private long Compute(BinaryOp op, long a, long b) => op switch {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => FloorDivSigned(a, b),
        BinaryOp.CeilDiv => CeilDivSigned(a, b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => PowChecked(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private float Compute(BinaryOp op, float a, float b) => op switch {
        BinaryOp.Add => a + b,
        BinaryOp.Sub => a - b,
        BinaryOp.Mul => a * b,
        BinaryOp.Div => a / b,
        BinaryOp.FloorDiv => System.MathF.Floor(a / b),
        BinaryOp.CeilDiv => System.MathF.Ceiling(a / b),
        BinaryOp.Mod => a % b,
        BinaryOp.Min => System.Math.Min(a, b),
        BinaryOp.Max => System.Math.Max(a, b),
        BinaryOp.Pow => System.MathF.Pow(a, b),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    private DataType LegalizeLowPrecisionFloat(DataType dataType) =>
        dataType.Legalize([
            (DataTypes.Float16, DataTypes.Float32),
            (DataTypes.BFloat16, DataTypes.Float32),
            (DataTypes.Float8E4M3, DataTypes.Float32),
            (DataTypes.Float8E5M2, DataTypes.Float32)
        ]);

    private Tensor LegalizeLowPrecisionFloat(Tensor tensor) {
        var legalType = LegalizeLowPrecisionFloat(tensor.ElementType);
        return legalType == tensor.ElementType ? tensor : tensor.CastTo(legalType);
    }

    private T FloorDivSigned<T>(T a, T b)
        where T : unmanaged, IBinaryInteger<T>, ISignedNumber<T>,
                  IEquatable<T> {
        var quotient = a / b;
        var remainder = a % b;
        if (remainder != T.Zero &&
            ((remainder > T.Zero) != (b > T.Zero))) {
            return checked(quotient - T.One);
        }

        return quotient;
    }

    private T CeilDivSigned<T>(T a, T b)
        where T : unmanaged, IBinaryInteger<T>, ISignedNumber<T>,
                  IEquatable<T> {
        var quotient = a / b;
        var remainder = a % b;
        if (remainder != T.Zero &&
            ((remainder > T.Zero) == (b > T.Zero))) {
            return checked(quotient + T.One);
        }

        return quotient;
    }

    private T FloorDivUnsigned<T>(T a, T b)
        where T : unmanaged, IBinaryInteger<T>, IUnsignedNumber<T>,
                  IEquatable<T> =>
        a / b;

    private T CeilDivUnsigned<T>(T a, T b)
        where T : unmanaged, IBinaryInteger<T>, IUnsignedNumber<T>,
                  IEquatable<T> {
        var quotient = a / b;
        return a % b == T.Zero ? quotient : checked(quotient + T.One);
    }

    private T PowChecked<T>(T value, T exponent)
        where T : unmanaged, IBinaryInteger<T>, IEquatable<T> {
        if (exponent < T.Zero) {
            throw new OverflowException(
                "Integer Pow does not support negative exponents.");
        }

        var result = T.One;
        var baseValue = value;
        var remainingExponent = exponent;
        var two = T.One + T.One;
        checked {
            while (remainingExponent > T.Zero) {
                if (remainingExponent % two != T.Zero) {
                    result *= baseValue;
                }

                remainingExponent /= two;
                if (remainingExponent > T.Zero) {
                    baseValue *= baseValue;
                }
            }
        }

        return result;
    }

    private Tensor EvaluatePointerBinary(
        Binary binary,
        Tensor lhs,
        Tensor rhs,
        TensorType resultType) {
        if (resultType.DType is not PointerType pointerType) {
            throw new InvalidOperationException(
                "Pointer binary expects a pointer return type");
        }

        var lhsIsPointer = lhs.ElementType is PointerType;
        var rhsIsPointer = rhs.ElementType is PointerType;
        if (lhsIsPointer == rhsIsPointer) {
            throw new InvalidOperationException(
                "Pointer arithmetic requires exactly one pointer operand.");
        }

        if (binary.BinaryOp is not BinaryOp.Add &&
            !(binary.BinaryOp is BinaryOp.Sub && lhsIsPointer)) {
            throw new InvalidOperationException(
                "Pointer arithmetic only supports pointer + integral, " +
                "integral + pointer, and pointer - integral.");
        }

        var pointer = lhsIsPointer ? lhs : rhs;
        var offset = lhsIsPointer ? rhs : lhs;
        var pointerUInt = ConvertPointerOperandToUInt64Tensor(pointer);
        var offsetInt = ConvertIntegralOperandToInt64Tensor(offset);
        var pointerShape = pointerUInt.Shape.ToValueArray();
        var offsetShape = offsetInt.Shape.ToValueArray();
        var pointerStrides = pointerUInt.Strides.ToArray();
        var offsetStrides = offsetInt.Strides.ToArray();
        var resultShape = resultType.Shape.ToValueArray();
        var resultTensor = resultShape.Length == 0
                               ? new Tensor<ulong>(Array.Empty<long>())
                               : new Tensor<ulong>(resultShape);
        var resultSpan = resultTensor.Buffer.Span;
        var pointerSpan = pointerUInt.Buffer.Span;
        var offsetSpan = offsetInt.Buffer.Span;
        var totalElements = resultShape.Length == 0
                                ? 1
                                : TensorUtilities.GetProduct(resultShape);
        var outIndices = resultShape.Length == 0 ? Array.Empty<long>()
                                                 : new long[resultShape.Length];
        var pointerIndices = pointerShape.Length == 0
                                 ? Array.Empty<long>()
                                 : new long[pointerShape.Length];
        var offsetIndices = offsetShape.Length == 0
                                ? Array.Empty<long>()
                                : new long[offsetShape.Length];
        var elemSize = pointerType.ElemType.SizeInBytes;
        for (long linear = 0; linear < totalElements; linear++) {
            if (outIndices.Length > 0) {
                TensorUtilities.UnravelIndex(linear, resultShape, outIndices);
            }

            var pointerOffset = GetBroadcastOffset(
                outIndices, pointerShape, pointerStrides, pointerIndices);
            var indexOffset = GetBroadcastOffset(
                outIndices,
                offsetShape,
                offsetStrides,
                offsetIndices);
            var byteOffset = checked(offsetSpan[(int)indexOffset] * elemSize);
            if (binary.BinaryOp == BinaryOp.Sub) {
                byteOffset = checked(-byteOffset);
            }

            resultSpan[(int)linear] = AddSignedByteOffset(
                pointerSpan[(int)pointerOffset], byteOffset);
        }

        return resultTensor.CastElementTo(pointerType, CastMode.Reinterpret);
    }

    private Tensor<ulong> ConvertPointerOperandToUInt64Tensor(Tensor operand) {
        Tensor converted = operand.ElementType switch {
            PointerType =>
                operand.CastElementTo(DataTypes.UInt64, CastMode.Reinterpret),
            _ => throw new InvalidOperationException(
                "Pointer arithmetic requires a pointer operand."),
        };

        return (Tensor<ulong>)converted;
    }

    private Tensor<long> ConvertIntegralOperandToInt64Tensor(Tensor operand) {
        Tensor converted = operand.ElementType switch {
            _ when operand.ElementType.IsIntegral() =>
                operand.CastElementTo(DataTypes.Int64),
            _ => throw new InvalidOperationException(
                "Pointer arithmetic requires an integral offset operand."),
        };

        return (Tensor<long>)converted;
    }

    private ulong AddSignedByteOffset(ulong pointer, long byteOffset) {
        checked {
            return byteOffset >= 0 ? pointer + (ulong)byteOffset
                                   : pointer - (ulong)-byteOffset;
        }
    }

    private long GetBroadcastOffset(
        long[] outIndices,
        long[] operandShape,
        long[] operandStrides,
        long[] scratch) {
        if (operandShape.Length == 0) {
            return 0;
        }

        if (outIndices.Length < operandShape.Length) {
            throw new InvalidOperationException("Broadcast rank mismatch");
        }

        var rankDiff = outIndices.Length - operandShape.Length;
        for (int i = 0; i < operandShape.Length; i++) {
            var sourceIndex =
                operandShape[i] == 1 ? 0 : outIndices[i + rankDiff];
            scratch[i] = sourceIndex;
        }

        return TensorUtilities.GetLinearOffset(operandStrides, scratch);
    }

    private Tensor EvaluateIntegralDivision(
        BinaryOp op,
        Tensor lhs,
        Tensor rhs,
        DataType dataType) =>
        dataType switch {
            VectorType vectorType =>
                EvaluateIntegralVectorDivision(op, lhs, rhs, vectorType),
            _ when dataType == DataTypes.Int8 =>
                EvaluateIntegralBinary<sbyte>(
                    lhs, rhs, op, FloorDivSigned, CeilDivSigned),
            _ when dataType == DataTypes.Int16 =>
                EvaluateIntegralBinary<short>(
                    lhs, rhs, op, FloorDivSigned, CeilDivSigned),
            _ when dataType == DataTypes.Int32 =>
                EvaluateIntegralBinary<int>(
                    lhs, rhs, op, FloorDivSigned, CeilDivSigned),
            _ when dataType == DataTypes.Int64 =>
                EvaluateIntegralBinary<long>(
                    lhs, rhs, op, FloorDivSigned, CeilDivSigned),
            _ when dataType == DataTypes.UInt8 =>
                EvaluateIntegralBinary<byte>(
                    lhs, rhs, op, FloorDivUnsigned, CeilDivUnsigned),
            _ when dataType == DataTypes.UInt16 =>
                EvaluateIntegralBinary<ushort>(
                    lhs, rhs, op, FloorDivUnsigned, CeilDivUnsigned),
            _ when dataType == DataTypes.UInt32 =>
                EvaluateIntegralBinary<uint>(
                    lhs, rhs, op, FloorDivUnsigned, CeilDivUnsigned),
            _ when dataType == DataTypes.UInt64 =>
                EvaluateIntegralBinary<ulong>(
                    lhs, rhs, op, FloorDivUnsigned, CeilDivUnsigned),
            _ => throw new NotSupportedException(
                $"Integral {op} does not support data type {dataType}."),
        };

    private Tensor EvaluateIntegralVectorDivision(
        BinaryOp op,
        Tensor lhs,
        Tensor rhs,
        VectorType vectorType) {
        if (!vectorType.ElemType.IsIntegral()) {
            throw new NotSupportedException(
                $"Integral {op} does not support vector element type {vectorType.ElemType}.");
        }

        var expandedLhs = ExpandIntegralVectorOperand(lhs, vectorType);
        var expandedRhs = ExpandIntegralVectorOperand(rhs, vectorType);
        var expandedResult =
            EvaluateIntegralDivision(
                op,
                expandedLhs,
                expandedRhs,
                vectorType.ElemType);
        return RepackIntegralVectorResult(expandedResult, vectorType);
    }

    private Tensor ExpandIntegralVectorOperand(
        Tensor tensor,
        VectorType resultVectorType) {
        if (tensor.ElementType is VectorType operandVectorType) {
            var expandedShape = tensor.Dimensions.ToArray()
                                      .Concat(operandVectorType.Lanes
                                                  .Select(lane => (long)lane))
                                      .ToArray();
            return tensor.CastTo(
                operandVectorType.ElemType,
                CastMode.Reinterpret,
                expandedShape);
        }

        if (tensor.ElementType != resultVectorType.ElemType) {
            throw new InvalidOperationException(
                $"Cannot evaluate vector integral binary with operand type {tensor.ElementType} and vector element type {resultVectorType.ElemType}.");
        }

        var broadcastShape = tensor.Dimensions.ToArray()
                                   .Concat(Enumerable.Repeat(
                                       1L, resultVectorType.Lanes.Count))
                                   .ToArray();
        return tensor.Reshape(broadcastShape);
    }

    private Tensor RepackIntegralVectorResult(
        Tensor expandedResult,
        VectorType vectorType) {
        var expandedShape = expandedResult.Dimensions.ToArray();
        var laneRank = vectorType.Lanes.Count;
        if (expandedShape.Length < laneRank) {
            throw new InvalidOperationException(
                $"Cannot repack scalar shape [{string.Join(", ", expandedShape)}] into vector type {vectorType}.");
        }

        var laneShape = expandedShape[^laneRank..];
        if (!laneShape.SequenceEqual(vectorType.Lanes.Select(l => (long)l))) {
            throw new InvalidOperationException(
                $"Cannot repack scalar lane shape [{string.Join(", ", laneShape)}] into vector type {vectorType}.");
        }

        var tensorShape = expandedShape[..^laneRank];
        return expandedResult.CastTo(
            vectorType,
            CastMode.Reinterpret,
            tensorShape);
    }

    private Tensor EvaluateIntegralBinary<T>(
        Tensor lhsTensor,
        Tensor rhsTensor,
        BinaryOp op,
        Func<T, T, T> floor,
        Func<T, T, T> ceil)
        where T : unmanaged, IBinaryInteger<T>, IEquatable<T> {
        var lhs = (Tensor<T>)lhsTensor;
        var rhs = (Tensor<T>)rhsTensor;
        var resultShape = GetBroadcastShape(lhs, rhs);
        var result = new Tensor<T>(resultShape);
        var resultSpan = result.Buffer.Span;
        var lhsShape = lhs.Dimensions.ToArray();
        var rhsShape = rhs.Dimensions.ToArray();
        var lhsStrides = lhs.Strides.ToArray();
        var rhsStrides = rhs.Strides.ToArray();
        var lhsIndices = lhsShape.Length == 0 ? Array.Empty<long>()
                                              : new long[lhsShape.Length];
        var rhsIndices = rhsShape.Length == 0 ? Array.Empty<long>()
                                              : new long[rhsShape.Length];
        var outIndices = resultShape.Length == 0 ? Array.Empty<long>()
                                                 : new long[resultShape.Length];
        var totalElements = resultShape.Length == 0
                                ? 1
                                : TensorUtilities.GetProduct(resultShape);
        var compute = op switch {
            BinaryOp.FloorDiv => floor,
            BinaryOp.CeilDiv => ceil,
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };

        for (long linear = 0; linear < totalElements; linear++) {
            if (outIndices.Length > 0) {
                TensorUtilities.UnravelIndex(linear, resultShape, outIndices);
            }

            var lhsOffset = GetBroadcastOffset(
                outIndices, lhsShape, lhsStrides, lhsIndices);
            var rhsOffset = GetBroadcastOffset(
                outIndices, rhsShape, rhsStrides, rhsIndices);
            resultSpan[checked((int)linear)] = compute(
                lhs.Buffer.Span[checked((int)lhsOffset)],
                rhs.Buffer.Span[checked((int)rhsOffset)]);
        }

        return result;
    }

    private long[] GetBroadcastShape(Tensor lhs, Tensor rhs) {
        var lhsShape = lhs.Dimensions.ToArray();
        var rhsShape = rhs.Dimensions.ToArray();
        var rank = System.Math.Max(lhsShape.Length, rhsShape.Length);
        var resultShape = new long[rank];
        for (int i = 0; i < rank; i++) {
            var lhsIndex = i - (rank - lhsShape.Length);
            var rhsIndex = i - (rank - rhsShape.Length);
            var lhsDim = lhsIndex < 0 ? 1 : lhsShape[lhsIndex];
            var rhsDim = rhsIndex < 0 ? 1 : rhsShape[rhsIndex];
            if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1) {
                throw new InvalidOperationException(
                    $"Cannot broadcast shapes [{string.Join(", ", lhsShape)}] and [{string.Join(", ", rhsShape)}].");
            }

            resultShape[i] = System.Math.Max(lhsDim, rhsDim);
        }

        return resultShape;
    }

    private OrtDataType GetFloatDivisionType(
        OrtKISharp.Tensor lhs,
        OrtKISharp.Tensor rhs) =>
        lhs.DataType == OrtDataType.Double || rhs.DataType == OrtDataType.Double
            ? OrtDataType.Double
            : OrtDataType.Float;

    private Tensor Ort_compute(
        Binary binary,
        Tensor lhs,
        Tensor rhs,
        DataType dataType) {
        if (binary.BinaryOp is BinaryOp.FloorDiv or BinaryOp.CeilDiv &&
            dataType.IsIntegral()) {
            return EvaluateIntegralDivision(
                binary.BinaryOp, lhs, rhs, dataType);
        }

        var a = lhs.ToOrtTensor();
        var b = rhs.ToOrtTensor();
        if (lhs.ElementType is VectorType vt &&
            rhs.ElementType is not VectorType) {
            b.Reshape(b.Shape.Concat(Enumerable.Repeat(1L, vt.Lanes.Count))
                          .ToArray());
        }

        if (rhs.ElementType is VectorType vt2 &&
            lhs.ElementType is not VectorType) {
            a.Reshape(a.Shape.Concat(Enumerable.Repeat(1L, vt2.Lanes.Count))
                          .ToArray());
        }

        var floatDivisionType = GetFloatDivisionType(a, b);

        static OrtKISharp.Tensor Mod(OrtKISharp.Tensor a, OrtKISharp.Tensor b) {
            var fmod = DataTypes.IsFloat(a.DataType.ToDataType()) &&
                               DataTypes.IsFloat(b.DataType.ToDataType())
                           ? 1L
                           : 0L;
            return OrtKI.Mod(a, b, fmod);
        }

        return (binary.BinaryOp switch {
                   BinaryOp.Add => a + b,
                   BinaryOp.Sub => a - b,
                   BinaryOp.Mul => a * b,
                   BinaryOp.Div => a / b,
                   BinaryOp.FloorDiv => OrtKI.Floor(
                       a.Cast(floatDivisionType) / b.Cast(floatDivisionType))
                       .Cast(a.DataType),
                   BinaryOp.CeilDiv => OrtKI.Ceil(
                       a.Cast(floatDivisionType) / b.Cast(floatDivisionType))
                       .Cast(a.DataType),
                   BinaryOp.Mod => Mod(a, b),
                   BinaryOp.Min => OrtKI.Min(new[] { a, b }),
                   BinaryOp.Max => OrtKI.Max(new[] { a, b }),
                   BinaryOp.Pow => OrtKI.Pow(a, b),
                   BinaryOp.BitwiseAnd => throw new NotSupportedException(
                       "NotSupported BinaryOp BitwiseAnd"),
                   BinaryOp.BitwiseOr => throw new NotSupportedException(
                       "NotSupported BinaryOp BitwiseOr"),
                   BinaryOp.BitwiseXor => throw new NotSupportedException(
                       "NotSupported BinaryOp BitwiseXor"),
                   BinaryOp.LogicalAnd => OrtKI.And(a, b),
                   BinaryOp.LogicalOr => OrtKI.Or(a, b),
                   BinaryOp.LogicalXor => OrtKI.Xor(a, b),
                   BinaryOp.LeftShift => OrtKI.LeftShift(a, b),
                   BinaryOp.RightShift => OrtKI.RightShift(a, b),
                   _ => throw new ArgumentOutOfRangeException(nameof(binary)),
               })
            .ToTensor(dataType);
    }

    private IRType Visit(Binary target, TensorType lhs, TensorType rhs) {
        if (target.BinaryOp is BinaryOp.LeftShift or BinaryOp.RightShift &&
            (lhs.DType != DataTypes.UInt32 || rhs.DType != DataTypes.UInt32)) {
            return new InvalidType("The Binary LeftShift RightShift Only " +
                                   "Accept The UInt32 Datatype.");
        }

        if ((target.BinaryOp is BinaryOp.LogicalAnd or
                 BinaryOp.LogicalOr or BinaryOp.LogicalXor) &&
            (lhs.DType != DataTypes.Boolean ||
             rhs.DType != DataTypes.Boolean)) {
            return new InvalidType(
                "The Binary Logical Only Accept The Boolean Datatype.");
        }

        if (lhs is { DType: PointerType }) {
            if (rhs.DType.IsIntegral() &&
                target.BinaryOp is BinaryOp.Add or BinaryOp.Sub) {
                return TypeInference.BroadcastType(lhs.DType, lhs, rhs);
            }

            return new InvalidType(
                $"The Binary Lhs {CompilerServices.Print(lhs)} != Rhs {CompilerServices.Print(rhs)}");
        }

        if (rhs is { DType: PointerType }) {
            if (lhs.DType.IsIntegral() && target.BinaryOp == BinaryOp.Add) {
                return TypeInference.BroadcastType(rhs.DType, lhs, rhs);
            }

            return new InvalidType(
                $"The Binary Lhs {CompilerServices.Print(lhs)} != Rhs {CompilerServices.Print(rhs)}");
        }

        return TypeInference.BroadcastType(lhs, rhs);
    }
}
