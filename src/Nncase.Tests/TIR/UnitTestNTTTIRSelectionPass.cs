// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Buffers;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Tiling;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestNTTTIRSelectionPass : TestClassBase
{
    public UnitTestNTTTIRSelectionPass()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions();
    }

    [Fact]
    public async Task BroadcastSelectsNttExpandKernel()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, Shape.Scalar));
        var body = IR.F.Tensors.Broadcast(input, new RankedShape(4));
        var function = new Function("main", new IRBlock(body, input));
        Assert.True(CompilerServices.InferenceType(function));

        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        var fields = lowered.Body.Fields.ToArray();
        var expand = Assert.IsType<Call>(fields[0]);

        Assert.IsType<Nncase.TIR.NTT.Expand>(expand.Target);
        Assert.IsType<Return>(fields[^1]);
    }

    [Fact]
    public async Task SequentialRegisterDirectAffineLowersWithoutAddressableIntermediates()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var sum = IR.F.Math.Binary(BinaryOp.Add, left, right);
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var body = new Sequential(new Expr[] { left, right, sum, scatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = lowered.Body.Fields.ToArray();
        var allExprs = ExprCollector.Collect(lowered.Body).ToArray();
        var calls = allExprs.OfType<Call>().ToArray();

        Assert.Contains(fields, expr => expr is Nncase.TIR.For { Mode: LoopMode.Serial });
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.VectorizedBinary);
        Assert.DoesNotContain(allExprs, expr => expr is Nncase.TIR.Buffer);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Load);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Store);
        var ret = Assert.IsType<Return>(fields[^1]);
        Assert.Empty(ret.Values.ToArray());
    }

    [Fact]
    public async Task RegisterDirectAffineLowersAllIndependentScattersInBlockOrder()
    {
        const int blockSize = 256;
        var lhs0 = new Var("lhs0", TensorType.Pointer(DataTypes.Float32));
        var rhs0 = new Var("rhs0", TensorType.Pointer(DataTypes.Float32));
        var dest0 = new Var("dest0", TensorType.Pointer(DataTypes.Float32));
        var lhs1 = new Var("lhs1", TensorType.Pointer(DataTypes.Float32));
        var rhs1 = new Var("rhs1", TensorType.Pointer(DataTypes.Float32));
        var dest1 = new Var("dest1", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left0 = Nncase.IR.F.Affine.Gather(lhs0, relation, symbols, shape, None.Default);
        var right0 = Nncase.IR.F.Affine.Gather(rhs0, relation, symbols, shape, None.Default);
        var scatter0 = Nncase.IR.F.Affine.Scatter(left0 + right0, dest0, relation, symbols);
        var left1 = Nncase.IR.F.Affine.Gather(lhs1, relation, symbols, shape, None.Default);
        var right1 = Nncase.IR.F.Affine.Gather(rhs1, relation, symbols, shape, None.Default);
        var scatter1 = Nncase.IR.F.Affine.Scatter(left1 + right1, dest1, relation, symbols);
        var body = new Sequential(new Expr[] { left0, right0, scatter0, left1, right1, scatter1 });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs0, rhs0, dest0, lhs1, rhs1, dest1));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = lowered.Body.Fields.ToArray();
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();

        Assert.Equal(2, fields.OfType<Nncase.TIR.For>().Count());
        Assert.Equal(2, calls.Count(call => call.Target is Nncase.TIR.Store));
        Assert.IsType<Return>(fields[^1]);
    }

    [Fact]
    public async Task RegisterDirectAffineRejectsUnrelatedSideEffectField()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var scatter = Nncase.IR.F.Affine.Scatter(left + right, dest, relation, symbols);
        var unrelatedStore = T.Store(dest, Dimension.Zero, Tensor.FromScalar(0f));
        var body = new Sequential(new Expr[] { unrelatedStore, left, right, scatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        Assert.Contains("cannot preserve unrelated field", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RegisterDirectAffineRejectsMaskedGatherDefaultValue()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var defaultValue = Const.FromTensor(Tensor.Zeros(DataTypes.Float32, new long[] { blockSize }));
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, defaultValue);
        var scatter = Nncase.IR.F.Affine.Scatter(left, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        Assert.Contains("default is None", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task RegisterDirectAffineLowersUnaryCastWhereWithoutAddressableIntermediates()
    {
        const int blockSize = 256;
        var input = new Var("input", TensorType.Pointer(DataTypes.Float32));
        var cond = new Var("cond", TensorType.Pointer(DataTypes.Boolean));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var whereDest = new Var("where_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var inputTile = Nncase.IR.F.Affine.Gather(input, relation, symbols, shape, None.Default);
        var castSource = IR.F.Tensors.Cast(IR.F.Math.Unary(UnaryOp.Neg, inputTile), DataTypes.Float32);
        var condTile = Nncase.IR.F.Affine.Gather(cond, relation, symbols, shape, None.Default);
        var rhsTile = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var whereScatter = Nncase.IR.F.Affine.Scatter(IR.F.Tensors.Where(condTile, castSource, rhsTile), whereDest, relation, symbols);
        var body = new Sequential(new Expr[] { inputTile, castSource, condTile, rhsTile, whereScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, input, cond, rhs, whereDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));

        AssertRegisterDirectAffineLowering(lowered);
    }

    [Fact]
    public async Task DistributedRegisterDirectAffineUsesTritonBlockedStridedThreadOwnership()
    {
        const int blockSize = 1024;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var domain = Nncase.IR.F.Affine.Dim(0);
        domain.Metadata.Range = new(0, blockSize - 1);
        var programId = new ProgramIdDim(0);
        var programIdSymbol = Nncase.IR.F.Affine.Symbol(0);
        var relation = new AffineRelation(
            [domain],
            [programIdSymbol],
            [(((AffineExpr)programIdSymbol) * (AffineConstant)(long)blockSize) + domain]);
        var symbols = new RankedShape(programId);
        var shape = new RankedShape(blockSize);
        var ndsbp = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var placement = new Placement([128], "t");
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var sum = IR.F.Math.Binary(BinaryOp.Add, left, right);
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var printed = CompilerServices.Print(lowered, PrinterFlags.Script);

        AssertRegisterDirectAffineLowering(lowered);
        Assert.Contains("T.Serial(out var d0, (0, 8, 1)", printed, StringComparison.Ordinal);
        Assert.Contains("128 * d0", printed, StringComparison.Ordinal);
        Assert.Contains("tid", printed, StringComparison.Ordinal);
        Assert.DoesNotContain("8 * tid", printed, StringComparison.Ordinal);
    }

    [Fact]
    public async Task BlockLocalSmemSharedGatherLowersWithSyncAndSharedBuffer()
    {
        const int blockSize = 128;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, shape, None.Default);
        var firstScatter = Nncase.IR.F.Affine.Scatter(tile, firstDest, relation, symbols);
        var secondScatter = Nncase.IR.F.Affine.Scatter(tile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { tile, firstScatter, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, source, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal(BufferScope.BlockLocal, decision.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, decision.Storage.PhysicalLocation);

        var selected = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = selected.Body.Fields.ToArray();
        Assert.IsType<Nncase.TIR.NTT.AffineGather>(Assert.IsType<Call>(fields[0]).Target);
        Assert.IsType<Nncase.TIR.NTT.SynchronizeThreads>(Assert.IsType<Call>(fields[1]).Target);
        Assert.IsType<Nncase.TIR.NTT.AffineScatter>(Assert.IsType<Call>(fields[2]).Target);
        Assert.IsType<Nncase.TIR.NTT.AffineScatter>(Assert.IsType<Call>(fields[3]).Target);
        Assert.IsType<Return>(fields[4]);
        var smemBuffer = Assert.Single(ExprCollector.Collect(selected.Body).OfType<Nncase.TIR.Buffer>());
        Assert.Equal(BufferScope.BlockLocal, smemBuffer.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, smemBuffer.Storage.PhysicalLocation);

        var lowered = Assert.IsType<PrimFunction>(await new NTTAffineIOLoweringPass().RunAsync(selected, new()));
        var calls = ExprCollector.Collect(lowered.Body).OfType<Call>().ToArray();
        Assert.Contains(calls, call => call.Target is Nncase.TIR.NTT.SynchronizeThreads);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.Contains(calls, call => call.Target is BufferStore);
        Assert.Contains(calls, call => call.Target is BufferLoad);
        Assert.True(CompilerServices.InferenceType(lowered), CompilerServices.Print(lowered, PrinterFlags.Script));

        var module = new IRModule(lowered);
        var passManager = CompileSession.CreatePassManager("smem-bufferize");
        passManager.Add<BufferizePass>();
        module = await passManager.RunAsync(module);
        var scheduled = Assert.IsType<PrimFunction>(module.Entry);
        Assert.True(scheduled.SchedResult.IsScheduled);
        Assert.True(scheduled.SchedResult.BlockLocalDataPoolSize >= (ulong)decision.ByteSize);
    }

    [Fact]
    public async Task DirectAffineSelectionRequiresTileDecision()
    {
        const int blockSize = 256;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(left, lhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.Contains("requires a tile decision", ex.Message, StringComparison.Ordinal);
    }

    private static void AssertRegisterDirectAffineLowering(PrimFunction lowered)
    {
        var fields = lowered.Body.Fields.ToArray();
        var allExprs = ExprCollector.Collect(lowered.Body).ToArray();
        var calls = allExprs.OfType<Call>().ToArray();

        Assert.Contains(fields, expr => expr is Nncase.TIR.For { Mode: LoopMode.Serial });
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineGather);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.AffineScatter);
        Assert.DoesNotContain(calls, call => call.Target is Nncase.TIR.NTT.VectorizedBinary);
        Assert.DoesNotContain(allExprs, expr => expr is Nncase.TIR.Buffer);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Load);
        Assert.Contains(calls, call => call.Target is Nncase.TIR.Store);
        var ret = Assert.IsType<Return>(fields[^1]);
        Assert.Empty(ret.Values.ToArray());
    }
}
