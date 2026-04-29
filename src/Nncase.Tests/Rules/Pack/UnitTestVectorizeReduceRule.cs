// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.NN;
using Nncase.Passes;
using Nncase.Passes.Rules.NTT;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestVectorizeReduceRule : TestClassBase
{
    [Fact]
    public void TestVectorizeRuleReturnsNullWhenMatchHasNoCandidate()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new long[] { 1, 4, 8, 8 }));
        var weights = new Var(new TensorType(DataTypes.Float32, new long[] { 4, 4, 3, 3 }));
        var bias = new Var(new TensorType(DataTypes.Float32, new long[] { 4 }));
        var conv = IR.F.NN.Conv2D(
            input,
            weights,
            bias,
            new long[] { 1, 1 },
            new long[,] { { 0, 0 }, { 0, 0 } },
            new long[] { 1, 1 },
            PadMode.Constant,
            1);
        var rule = new VectorizeConv2D(1, 32);

        Assert.True(CompilerServices.InferenceType(conv), CompilerServices.Print(conv));
        Assert.True(CompilerServices.TryMatch(conv, rule.Pattern, out var result));
        Assert.Empty(rule.GetReplaceCandidates(result!, new RunPassContext()));
        Assert.Null(rule.GetReplace(result!, new RunPassContext()));
    }

    [Fact]
    public void TestVectorizeReduceRejectsMeanOnVectorizedReduceAxis()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new long[] { 1, 3 }));
        var candidates = VectorizeReduce.AddCandidate(
            new Nncase.IR.Math.Reduce(ReduceOp.Mean),
            input,
            [-1],
            0f,
            true,
            [1],
            [4]);

        Assert.Empty(candidates);
    }

    [Fact]
    public void TestVectorizeReduceUsesNeutralPadForVectorizedMaxAxis()
    {
        var input = new Var(new TensorType(DataTypes.Float32, new long[] { 1, 3 }));
        var candidates = VectorizeReduce.AddCandidate(
            new Nncase.IR.Math.Reduce(ReduceOp.Max),
            input,
            [-1],
            0f,
            true,
            [1],
            [4]);

        var candidate = Assert.Single(candidates);
        var reduceCall = Assert.IsType<Call>(candidate);
        Assert.IsType<Nncase.IR.NTT.VectorizedReduce>(reduceCall.Target);
        var packCall = Assert.IsType<Call>(reduceCall.Arguments[0]);
        Assert.IsType<Nncase.IR.Tensors.Pack>(packCall.Target);
        var padCall = Assert.IsType<Call>(packCall.Arguments[0]);
        Assert.IsType<Pad>(padCall.Target);
        var padValue = Assert.IsType<TensorConst>(padCall[Pad.Value]);
        Assert.Equal(float.MinValue, padValue.Value.ToScalar<float>());
    }
}
