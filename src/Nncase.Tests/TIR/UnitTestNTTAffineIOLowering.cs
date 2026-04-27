// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Buffers;
using Nncase.IR.Distributed;
using Nncase.IR.Logics;
using Nncase.IR.Shapes;
using Nncase.Passes;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestNTTAffineIOLowering : TestClassBase
{
    [Fact]
    public async Task MaskedSymbolicGatherLowersToGuardedLoadAndDefaultStore()
    {
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var output = CreateVectorBuffer("output");
        var defaultValue = CreateVectorBuffer("default_value");
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineGather(source, defaultValue, output, relation, symbols, new RankedShape(4));
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var guard = GetSingleGuard(lowered);

        Assert.IsType<DimCompare>(guard.Condition);
        var thenStore = AssertSingleCall<BufferStore>(guard.Then);
        Assert.IsType<Load>(Assert.IsType<Call>(thenStore[BufferStore.Value]).Target);
        var elseStore = AssertSingleCall<BufferStore>(guard.Else);
        Assert.IsType<BufferLoad>(Assert.IsType<Call>(elseStore[BufferStore.Value]).Target);
    }

    [Fact]
    public async Task MaskedSymbolicGatherEvaluatesFullBlockAndTailBlockSemantics()
    {
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var output = CreateVectorBuffer("output");
        var defaultValue = CreateVectorBuffer("default_value");
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineGather(source, defaultValue, output, relation, symbols, new RankedShape(4));
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var (outerLoop, guard) = GetSingleGuardWithLoop(lowered);
        var thenStore = AssertSingleCall<BufferStore>(guard.Then);
        var loadCall = Assert.IsType<Call>(thenStore[BufferStore.Value]);
        Assert.IsType<Load>(loadCall.Target);
        var loadAddress = Assert.IsAssignableFrom<Dimension>(loadCall[Load.Index]);
        var elseStore = AssertSingleCall<BufferStore>(guard.Else);
        Assert.IsType<BufferLoad>(Assert.IsType<Call>(elseStore[BufferStore.Value]).Target);

        AssertGuardEvaluates(guard, outerLoop, programId: 1, nElements: 8, expected: [true, true, true, true]);
        AssertAddresses(loadAddress, outerLoop, programId: 1, expected: [4, 5, 6, 7]);
        AssertGuardEvaluates(guard, outerLoop, programId: 2, nElements: 10, expected: [true, true, false, false]);
        AssertAddresses(loadAddress, outerLoop, programId: 2, expected: [8, 9, 10, 11]);
    }

    [Fact]
    public async Task MaskedSymbolicScatterLowersToGuardedStoreWithoutElseWrite()
    {
        var source = CreateVectorBuffer("source");
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var guard = GetSingleGuard(lowered);

        Assert.IsType<DimCompare>(guard.Condition);
        AssertSingleCall<Store>(guard.Then);
        Assert.Empty(guard.Else.Fields.ToArray());
    }

    [Fact]
    public async Task MaskedSymbolicScatterEvaluatesFullBlockAndTailBlockSemantics()
    {
        var source = CreateVectorBuffer("source");
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var (outerLoop, guard) = GetSingleGuardWithLoop(lowered);
        var storeCall = AssertSingleCall<Store>(guard.Then);
        var storeAddress = Assert.IsAssignableFrom<Dimension>(storeCall[Store.Index]);
        Assert.Empty(guard.Else.Fields.ToArray());

        AssertGuardEvaluates(guard, outerLoop, programId: 1, nElements: 8, expected: [true, true, true, true]);
        AssertAddresses(storeAddress, outerLoop, programId: 1, expected: [4, 5, 6, 7]);
        AssertGuardEvaluates(guard, outerLoop, programId: 2, nElements: 10, expected: [true, true, false, false]);
        AssertAddresses(storeAddress, outerLoop, programId: 2, expected: [8, 9, 10, 11]);
    }

    [Fact]
    public async Task SymbolPayloadMismatchIsRejected()
    {
        var source = CreateVectorBuffer("source");
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 1);
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => new NTTAffineIOLoweringPass().RunAsync(function, new()));
        Assert.Contains("Symbol payload", ex.Message, StringComparison.Ordinal);
        Assert.Contains("relation", ex.Message, StringComparison.Ordinal);
    }

    private static Nncase.TIR.Buffer CreateVectorBuffer(string name)
    {
        return T.CreateBuffer(new TensorType(DataTypes.Float32, new RankedShape(4)), MemoryLocation.Data, out _, name);
    }

    private static (AffineRelation Relation, RankedShape Symbols) CreateVectorAddRelation(int symbolCount)
    {
        const int blockSize = 4;
        var lane = new DimVar("d0");
        lane.Metadata.Range = new(0, blockSize - 1);
        var programId = new ProgramIdDim(0);
        var nElements = new DimVar("n_elements");
        var relation = new AffineRelation(
            new[] { Nncase.IR.F.Affine.Dim(0) },
            new[] { Nncase.IR.F.Affine.Symbol(0), Nncase.IR.F.Affine.Symbol(1) },
            new AffineExpr[] { (((AffineExpr)Nncase.IR.F.Affine.Symbol(0)) * (AffineConstant)(long)blockSize) + Nncase.IR.F.Affine.Dim(0) },
            Nncase.IR.F.Shapes.LowerThan((programId * blockSize) + lane, nElements));
        var symbols = symbolCount == 1 ? new RankedShape(programId) : new RankedShape(programId, nElements);
        return (relation, symbols);
    }

    private static IfThenElse GetSingleGuard(PrimFunction function)
    {
        return GetSingleGuardWithLoop(function).Guard;
    }

    private static (Nncase.TIR.For Loop, IfThenElse Guard) GetSingleGuardWithLoop(PrimFunction function)
    {
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(function.Body.Fields.ToArray()));
        var guard = Assert.IsType<IfThenElse>(Assert.Single(outerLoop.Body.Fields.ToArray()));
        return (outerLoop, guard);
    }

    private static Call AssertSingleCall<TOp>(Sequential body)
        where TOp : Op
    {
        var call = Assert.IsType<Call>(Assert.Single(body.Fields.ToArray()));
        Assert.IsType<TOp>(call.Target);
        return call;
    }

    private static void AssertGuardEvaluates(IfThenElse guard, Nncase.TIR.For loop, long programId, long nElements, bool[] expected)
    {
        var actual = Enumerable.Range(0, expected.Length)
            .Select(lane => EvaluateLogical(guard.Condition, loop.LoopVar, lane, programId, nElements))
            .ToArray();
        Assert.Equal(expected, actual);
    }

    private static void AssertAddresses(Dimension address, Nncase.TIR.For loop, long programId, long[] expected)
    {
        var actual = Enumerable.Range(0, expected.Length)
            .Select(lane => EvaluateDimension(address, loop.LoopVar, lane, programId, nElements: 0))
            .ToArray();
        Assert.Equal(expected, actual);
    }

    private static bool EvaluateLogical(BaseExpr expr, DimVar loopVar, long lane, long programId, long nElements)
    {
        return expr switch
        {
            LogicalConst logicalConst => logicalConst.Value,
            DimCompare compare => EvaluateCompare(compare.Op, EvaluateDimension(compare.Lhs, loopVar, lane, programId, nElements), EvaluateDimension(compare.Rhs, loopVar, lane, programId, nElements)),
            LogicalAnd logicalAnd => logicalAnd.Operands.ToArray().All(x => EvaluateLogical(x, loopVar, lane, programId, nElements)),
            LogicalOr logicalOr => logicalOr.Operands.ToArray().Any(x => EvaluateLogical(x, loopVar, lane, programId, nElements)),
            _ => throw new NotSupportedException($"Unsupported logical expression {expr.GetType().Name}."),
        };
    }

    private static bool EvaluateCompare(CompareOp op, long lhs, long rhs)
    {
        return op switch
        {
            CompareOp.Equal => lhs == rhs,
            CompareOp.NotEqual => lhs != rhs,
            CompareOp.LowerThan => lhs < rhs,
            CompareOp.LowerOrEqual => lhs <= rhs,
            CompareOp.GreaterThan => lhs > rhs,
            CompareOp.GreaterOrEqual => lhs >= rhs,
            _ => throw new ArgumentOutOfRangeException(nameof(op)),
        };
    }

    private static long EvaluateDimension(Dimension dim, DimVar loopVar, long lane, long programId, long nElements)
    {
        return dim switch
        {
            DimConst constant => constant.Value,
            DimVar dimVar when ReferenceEquals(dimVar, loopVar) || dimVar.Name == loopVar.Name => lane,
            DimVar { Name: "n_elements" } => nElements,
            ProgramIdDim { Axis: 0 } => programId,
            DimSum sum => sum.Bias + sum.Operands.ToArray().Sum(x => EvaluateDimension(x, loopVar, lane, programId, nElements)),
            DimProduct product => product.Scale * product.Operands.ToArray().Aggregate(1L, (acc, x) => acc * EvaluateDimension(x, loopVar, lane, programId, nElements)),
            DimFraction fraction => EvaluateFraction(fraction, loopVar, lane, programId, nElements),
            DimRemainder remainder => EvaluateDimension(remainder.Numerator, loopVar, lane, programId, nElements) % EvaluateDimension(remainder.Denominator, loopVar, lane, programId, nElements),
            _ => throw new NotSupportedException($"Unsupported dimension expression {dim.GetType().Name}: {dim}"),
        };
    }

    private static long EvaluateFraction(DimFraction fraction, DimVar loopVar, long lane, long programId, long nElements)
    {
        var numerator = EvaluateDimension(fraction.Numerator, loopVar, lane, programId, nElements);
        var denominator = EvaluateDimension(fraction.Denominator, loopVar, lane, programId, nElements);
        return fraction.DivMode switch
        {
            DimDivideMode.FloorDiv => numerator / denominator,
            DimDivideMode.CeilDiv => (numerator + denominator - 1) / denominator,
            _ => throw new ArgumentOutOfRangeException(nameof(fraction), $"Unsupported divide mode {fraction.DivMode}."),
        };
    }
}
