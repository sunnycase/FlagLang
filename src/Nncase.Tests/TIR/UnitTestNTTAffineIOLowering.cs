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
    public async Task SymbolPayloadMismatchIsRejected()
    {
        var source = CreateVectorBuffer("source");
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols) = CreateVectorAddRelation(symbolCount: 1);
        var call = Nncase.TIR.F.NTT.AffineScatter(source, dest, relation, symbols);
        var function = new PrimFunction("main", CUDATarget.Kind, T.Sequential(call));

        await Assert.ThrowsAsync<InvalidOperationException>(() => new NTTAffineIOLoweringPass().RunAsync(function, new()));
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
        var outerLoop = Assert.IsType<Nncase.TIR.For>(Assert.Single(function.Body.Fields.ToArray()));
        return Assert.IsType<IfThenElse>(Assert.Single(outerLoop.Body.Fields.ToArray()));
    }

    private static Call AssertSingleCall<TOp>(Sequential body)
        where TOp : Op
    {
        var call = Assert.IsType<Call>(Assert.Single(body.Fields.ToArray()));
        Assert.IsType<TOp>(call.Target);
        return call;
    }
}
