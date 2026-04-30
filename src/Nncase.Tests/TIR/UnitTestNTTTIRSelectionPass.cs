// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestNTTTIRSelectionPass : TestClassBase
{
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
    public async Task SequentialScatterDoesNotCloneVoidResultIntoStatementFields()
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

        var tiled = Assert.IsType<Function>(await new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new()));
        var lowered = Assert.IsType<PrimFunction>(await new NTTTIRSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(tiled, new()));
        var fields = lowered.Body.Fields.ToArray();

        Assert.Contains(fields, expr => expr is Call { Target: Nncase.TIR.NTT.AffineScatter });
        var ret = Assert.IsType<Return>(fields[^1]);
        Assert.Empty(ret.Values.ToArray());
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
}
