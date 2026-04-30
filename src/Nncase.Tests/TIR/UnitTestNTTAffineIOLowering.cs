// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
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
    public async Task MaskedSymbolicGatherWithoutDefaultUsesZeroFallback()
    {
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var output = CreateVectorBuffer("output");
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineGather(source, None.Default, output, relation, symbols, new RankedShape(4));
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var guard = GetSingleGuard(lowered);
        var elseStore = AssertSingleCall<BufferStore>(guard.Else);
        var fallback = Assert.IsType<TensorConst>(elseStore[BufferStore.Value]);

        Assert.Equal(0f, fallback.Value.ToScalar<float>());
        Assert.True(CompilerServices.InferenceType(lowered));
        Assert.NotEmpty(CompilerServices.Print(lowered, PrinterFlags.Script));
    }

    [Fact]
    public async Task MaskedSymbolicGatherWithTensorDefaultMaterializesReadableBuffer()
    {
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var output = CreateVectorBuffer("output");
        var defaultValue = new Var("default_value", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineGather(source, defaultValue, output, relation, symbols, new RankedShape(4));
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var fields = FlattenSequential(lowered.Body).ToArray();

        var setup = Assert.IsType<Call>(fields[0]);
        Assert.IsType<Memcopy>(setup.Target);
        var loop = Assert.IsType<Nncase.TIR.For>(fields[1]);
        var guard = Assert.IsType<IfThenElse>(Assert.Single(loop.Body.Fields.ToArray()));
        var elseStore = AssertSingleCall<BufferStore>(guard.Else);
        Assert.IsType<BufferLoad>(Assert.IsType<Call>(elseStore[BufferStore.Value]).Target);
        Assert.True(CompilerServices.InferenceType(lowered));
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
    public async Task DistributedTensorLoadAfterAffineGatherFusesIntoLocalShardGather()
    {
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var temp = CreateVectorBuffer("temp");
        var output = CreateDistributedVectorBuffer("output", globalSize: 4, threadShards: 2);
        var (relation, symbols) = CreateIdentityRelation();
        var gather = Nncase.TIR.F.NTT.AffineGather(source, None.Default, temp, relation, symbols, new RankedShape(4));
        var tensorLoad = Nncase.TIR.F.NTT.TensorLoad(output, temp, output.DistributedType!.AxisPolicies, output.DistributedType.Placement);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(gather, tensorLoad));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(lowered.Body.Fields.ToArray()));
        var storeCall = AssertSingleCall<BufferStore>(outerLoop.Body);
        var loadCall = Assert.IsType<Call>(storeCall[BufferStore.Value]);
        Assert.IsType<Load>(loadCall.Target);
        var loadAddress = Assert.IsAssignableFrom<Dimension>(loadCall[Load.Index]);

        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 0));
        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 1));
        AssertAddresses(loadAddress, outerLoop, programId: 0, expected: [0, 1], threadId: 0);
        AssertAddresses(loadAddress, outerLoop, programId: 0, expected: [2, 3], threadId: 1);
        Assert.True(CompilerServices.InferenceType(lowered));
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
        Assert.True(CompilerServices.InferenceType(lowered));
    }

    [Fact]
    public async Task ScatterWithTensorSourceParameterMaterializesReadableBuffer()
    {
        var source = new Var("source", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 2);
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var fields = FlattenSequential(lowered.Body).ToArray();

        var setup = Assert.IsType<Call>(fields[0]);
        Assert.IsType<Memcopy>(setup.Target);
        Assert.IsType<Nncase.TIR.For>(fields[1]);
        Assert.True(CompilerServices.InferenceType(lowered));
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
    public async Task DistributedScatterIteratesLocalShardAndUsesGlobalAddress()
    {
        var source = CreateDistributedVectorBuffer("source", globalSize: 4, threadShards: 2);
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateIdentityRelation();
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(lowered.Body.Fields.ToArray()));
        var storeCall = AssertSingleCall<Store>(outerLoop.Body);
        var storeAddress = Assert.IsAssignableFrom<Dimension>(storeCall[Store.Index]);

        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 0));
        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 1));
        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [0, 1], threadId: 0);
        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [2, 3], threadId: 1);
        AssertLocalBufferLoadIndex(storeCall, outerLoop);
        Assert.True(CompilerServices.InferenceType(lowered));
    }

    [Fact]
    public async Task ExplicitSbpScatterUsesOwnerLocalMapWithoutLegacyAxisPolicies()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4));
        var placement = new Placement([2], "t");
        var layout = DistributionLayout.FromAxisPolicies(tensorType, [SBP.S(0)], placement);
        var distributedType = DistributedType.FromLayouts(tensorType, placement, layout);
        var source = T.CreateBuffer(tensorType, MemoryLocation.Data, out _, "source", distributedType);
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateIdentityRelation();
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(lowered.Body.Fields.ToArray()));
        var storeCall = AssertSingleCall<Store>(outerLoop.Body);
        var storeAddress = Assert.IsAssignableFrom<Dimension>(storeCall[Store.Index]);

        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 0));
        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [0, 1], threadId: 0);
        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [2, 3], threadId: 1);
        Assert.True(CompilerServices.InferenceType(lowered));
    }

    [Fact]
    public async Task ExplicitStorageLayoutLogicalToPhysicalControlsBufferLoadIndex()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4));
        var placement = new Placement([2], "t");
        var layout = DistributionLayout.FromAxisPolicies(tensorType, [SBP.S(0)], placement);
        var storageLayout = new StorageLayout(
            "ReverseLocal",
            layout.LocalShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                ["l0"],
                [new IndexMapBinding("p0", IndexExpr.Add(IndexExpr.Const(1), IndexExpr.Mul(IndexExpr.Const(-1), IndexExpr.Var("l0"))))],
                ["0<=l0<2"],
                ["0<=p0<2"],
                Inverse: "LogicalToPhysical"));
        var distributedType = DistributedType.FromLayouts(tensorType, placement, layout, storageLayout);
        var source = T.CreateBuffer(tensorType, MemoryLocation.Data, out _, "source", distributedType);
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateIdentityRelation();
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(lowered.Body.Fields.ToArray()));
        var storeCall = AssertSingleCall<Store>(outerLoop.Body);
        var storeAddress = Assert.IsAssignableFrom<Dimension>(storeCall[Store.Index]);
        var loadIndex = GetSingleBufferLoadIndex(storeCall);

        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [0, 1], threadId: 0);
        AssertDimensionValues(loadIndex, outerLoop, expected: [1, 0], threadId: 0);
        Assert.True(CompilerServices.InferenceType(lowered));
    }

    [Fact]
    public async Task ExplicitTritonBlockedScatterUsesLayoutDomainMap()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(8));
        var layout = DistributionLayout.TritonBlocked(
            tensorType.Shape,
            new TritonBlockedLayout(
                SizePerThread: 2,
                ThreadsPerWarp: 2,
                WarpsPerCTA: 1,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0],
                ThreadElementOrder: TritonThreadElementOrder.Strided));
        var distributedType = DistributedType.FromLayouts(tensorType, new Placement([2], "t"), layout);
        var source = T.CreateBuffer(tensorType, MemoryLocation.Data, out _, "source", distributedType);
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateIdentityRelation();
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(function, new()));
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(lowered.Body.Fields.ToArray()));
        var storeCall = AssertSingleCall<Store>(outerLoop.Body);
        var storeAddress = Assert.IsAssignableFrom<Dimension>(storeCall[Store.Index]);

        Assert.Equal(2, EvaluateDimension(outerLoop.Domain.Stop, outerLoop.LoopVar, lane: 0, programId: 0, nElements: 0, threadId: 0));
        AssertAddresses(storeAddress, outerLoop, programId: 0, expected: [1, 3], threadId: 1);
        AssertAddresses(storeAddress, outerLoop, programId: 1, expected: [4, 6], threadId: 0);
        Assert.True(CompilerServices.InferenceType(lowered));
    }

    [Fact]
    public async Task UnsupportedExplicitDistributionLayoutFailsFast()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(4));
        var baseLayout = DistributionLayout.TritonBlocked(
            tensorType.Shape,
            new TritonBlockedLayout(
                SizePerThread: 2,
                ThreadsPerWarp: 2,
                WarpsPerCTA: 1,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0]));
        var unsupported = baseLayout with { Kind = "xor_swizzle_owner" };
        var distributedType = DistributedType.FromLayouts(tensorType, new Placement([2], "t"), unsupported);
        var source = T.CreateBuffer(tensorType, MemoryLocation.Data, out _, "source", distributedType);
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateIdentityRelation();
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTAffineIOLoweringPass().RunAsync(function, new()));
        Assert.Contains("cannot evaluate explicit distribution layout xor_swizzle_owner", ex.Message, StringComparison.Ordinal);
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

    private static Nncase.TIR.Buffer CreateDistributedVectorBuffer(string name, int globalSize, int threadShards)
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(globalSize));
        var placement = new Placement(new[] { threadShards }, "t");
        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.S(0) }, placement);
        return T.CreateBuffer(tensorType, MemoryLocation.Data, out _, name, distributedType);
    }

    private static (AffineRelation Relation, RankedShape Symbols) CreateIdentityRelation()
    {
        var lane = Nncase.IR.F.Affine.Dim(0);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        return (relation, new RankedShape(Array.Empty<Dimension>()));
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

    private static IEnumerable<Expr> FlattenSequential(Sequential body)
    {
        foreach (var field in body.Fields.ToArray())
        {
            if (field is Sequential nested)
            {
                foreach (var nestedField in FlattenSequential(nested))
                {
                    yield return nestedField;
                }
            }
            else
            {
                yield return field;
            }
        }
    }

    private static void AssertGuardEvaluates(IfThenElse guard, Nncase.TIR.For loop, long programId, long nElements, bool[] expected, long threadId = 0)
    {
        var actual = Enumerable.Range(0, expected.Length)
            .Select(lane => EvaluateLogical(guard.Condition, loop.LoopVar, lane, programId, nElements, threadId))
            .ToArray();
        Assert.Equal(expected, actual);
    }

    private static void AssertAddresses(Dimension address, Nncase.TIR.For loop, long programId, long[] expected, long threadId = 0)
    {
        var actual = Enumerable.Range(0, expected.Length)
            .Select(lane => EvaluateDimension(address, loop.LoopVar, lane, programId, nElements: 0, threadId))
            .ToArray();
        Assert.Equal(expected, actual);
    }

    private static void AssertLocalBufferLoadIndex(Call storeCall, Nncase.TIR.For loop)
    {
        var asTensor = GetSingleBufferLoadAsTensor(storeCall);
        Assert.Same(loop.LoopVar, asTensor.Arguments[0]);
    }

    private static Dimension GetSingleBufferLoadIndex(Call storeCall)
    {
        var asTensor = GetSingleBufferLoadAsTensor(storeCall);
        return Assert.IsAssignableFrom<Dimension>(asTensor.Arguments[0]);
    }

    private static Call GetSingleBufferLoadAsTensor(Call storeCall)
    {
        var loadCall = Assert.IsType<Call>(storeCall[Store.Value]);
        Assert.IsType<BufferLoad>(loadCall.Target);
        var indices = Assert.IsType<Nncase.IR.Tuple>(loadCall[BufferLoad.Indices]);
        var cast = Assert.IsType<Call>(Assert.Single(indices.Fields.ToArray()));
        Assert.IsType<Nncase.IR.Tensors.Cast>(cast.Target);
        var asTensor = Assert.IsType<Call>(cast.Arguments[0]);
        Assert.IsType<Nncase.IR.Shapes.AsTensor>(asTensor.Target);
        return asTensor;
    }

    private static void AssertDimensionValues(Dimension dimension, Nncase.TIR.For loop, long[] expected, long threadId)
    {
        var actual = Enumerable.Range(0, expected.Length)
            .Select(lane => EvaluateDimension(dimension, loop.LoopVar, lane, programId: 0, nElements: 0, threadId))
            .ToArray();
        Assert.Equal(expected, actual);
    }

    private static bool EvaluateLogical(BaseExpr expr, DimVar loopVar, long lane, long programId, long nElements, long threadId = 0)
    {
        return expr switch
        {
            LogicalConst logicalConst => logicalConst.Value,
            DimCompare compare => EvaluateCompare(compare.Op, EvaluateDimension(compare.Lhs, loopVar, lane, programId, nElements, threadId), EvaluateDimension(compare.Rhs, loopVar, lane, programId, nElements, threadId)),
            LogicalAnd logicalAnd => logicalAnd.Operands.ToArray().All(x => EvaluateLogical(x, loopVar, lane, programId, nElements, threadId)),
            LogicalOr logicalOr => logicalOr.Operands.ToArray().Any(x => EvaluateLogical(x, loopVar, lane, programId, nElements, threadId)),
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

    private static long EvaluateDimension(Dimension dim, DimVar loopVar, long lane, long programId, long nElements, long threadId = 0)
    {
        return dim switch
        {
            DimConst constant => constant.Value,
            DimVar dimVar when ReferenceEquals(dimVar, loopVar) || dimVar.Name == loopVar.Name => lane,
            DimVar { Name: "n_elements" } => nElements,
            ThreadIdDim => threadId,
            ProgramIdDim { Axis: 0 } => programId,
            DimSum sum => sum.Bias + sum.Operands.ToArray().Sum(x => EvaluateDimension(x, loopVar, lane, programId, nElements, threadId)),
            DimProduct product => product.Scale * product.Operands.ToArray().Aggregate(1L, (acc, x) => acc * EvaluateDimension(x, loopVar, lane, programId, nElements, threadId)),
            DimFraction fraction => EvaluateFraction(fraction, loopVar, lane, programId, nElements, threadId),
            DimRemainder remainder => EvaluateDimension(remainder.Numerator, loopVar, lane, programId, nElements, threadId) % EvaluateDimension(remainder.Denominator, loopVar, lane, programId, nElements, threadId),
            DimMin min => min.Operands.ToArray().Min(x => EvaluateDimension(x, loopVar, lane, programId, nElements, threadId)),
            DimMax max => max.Operands.ToArray().Max(x => EvaluateDimension(x, loopVar, lane, programId, nElements, threadId)),
            _ => throw new NotSupportedException($"Unsupported dimension expression {dim.GetType().Name}: {dim}"),
        };
    }

    private static long EvaluateFraction(DimFraction fraction, DimVar loopVar, long lane, long programId, long nElements, long threadId)
    {
        var numerator = EvaluateDimension(fraction.Numerator, loopVar, lane, programId, nElements, threadId);
        var denominator = EvaluateDimension(fraction.Denominator, loopVar, lane, programId, nElements, threadId);
        return fraction.DivMode switch
        {
            DimDivideMode.FloorDiv => numerator / denominator,
            DimDivideMode.CeilDiv => (numerator + denominator - 1) / denominator,
            _ => throw new ArgumentOutOfRangeException(nameof(fraction), $"Unsupported divide mode {fraction.DivMode}."),
        };
    }
}
