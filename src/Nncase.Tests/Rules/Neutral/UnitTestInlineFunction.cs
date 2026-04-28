// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Passes.Rules.Neutral;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.Rules.NeutralTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestInlineFunction : TestClassBase
{
    [Fact]
    public void InlineFunctionAcceptsCallArguments()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new[] { 1 }));
        var helper = new Function("identity", new IRBlock(input, input));
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, new[] { 1 }));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, new[] { 1 }));
        var argument = Binary(BinaryOp.Add, lhs, rhs);
        var pre = new Call(helper, argument);
        Assert.True(CompilerServices.InferenceType(pre), CompilerServices.Print(pre));

        var post = CompilerServices.Rewrite(pre, new[] { new InlineFunction(20) }, new());

        var inlined = Assert.IsType<Call>(post);
        var binary = Assert.IsType<Binary>(inlined.Target);
        Assert.Equal(BinaryOp.Add, binary.BinaryOp);
        Assert.Same(lhs, inlined.Arguments[0]);
        Assert.Same(rhs, inlined.Arguments[1]);
        Assert.True(CompilerServices.InferenceType(post), CompilerServices.Print(post));
    }
}
