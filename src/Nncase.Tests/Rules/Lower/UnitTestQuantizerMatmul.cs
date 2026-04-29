// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Rules.Lower;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;

namespace Nncase.Tests.Rules.LowerTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestQuantizerMatmul : TransformTestBase
{
    public static TheoryData<DataType> Float8QuantTypes => new()
    {
        DataTypes.Float8E4M3,
        DataTypes.Float8E5M2,
    };

    [Fact]
    public void QuantMatmulE5M2AcceptsDynamicRhs()
    {
        var inputA = new Var("inputA", new TensorType(DataTypes.Float32, new long[] { 2, 3 }));
        var inputB = new Var("inputB", new TensorType(DataTypes.Float32, new long[] { 3, 4 }));
        var markedA = RangeOfMarker(inputA, Tensor.FromScalar(0.5f), DataTypes.Float8E5M2);
        var markedB = RangeOfMarker(inputB, Tensor.FromScalar(0.25f), DataTypes.Float8E5M2);
        var matmul = MatMul(markedA, markedB);

        Assert.True(CompilerServices.InferenceType(matmul), CompilerServices.Print(matmul));

        var rule = new QuantizerMatmul();
        Assert.True(CompilerServices.TryMatchRoot(matmul, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = Assert.IsAssignableFrom<Expr>(rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() }));

        Assert.True(CompilerServices.InferenceType(rewritten), CompilerServices.Print(rewritten));

        var casts = ExprCollector.Collect(rewritten)
            .OfType<Call>()
            .Where(call => call.Target is IR.Tensors.Cast cast && cast.NewType == DataTypes.Float8E5M2)
            .ToArray();
        Assert.Equal(2, casts.Length);
        Assert.Contains(casts, call => ExprCollector.Collect(call).Contains(inputB));
    }

    [Theory]
    [MemberData(nameof(Float8QuantTypes))]
    public void QuantMatmulFloat8PreservesConstLhsOperandOrder(DataType quantType)
    {
        var inputA = Const.FromTensor(Tensor.From(new float[] { 1, 2, 3, 4, 5, 6 }, new long[] { 2, 3 }));
        var inputB = new Var("inputB", new TensorType(DataTypes.Float32, new long[] { 3, 4 }));
        var markedA = RangeOfMarker(inputA, Tensor.FromScalar(0.5f), quantType);
        var markedB = RangeOfMarker(inputB, Tensor.FromScalar(0.25f), quantType);
        var matmul = MatMul(markedA, markedB);

        Assert.True(CompilerServices.InferenceType(matmul), CompilerServices.Print(matmul));

        var rule = new QuantizerMatmul();
        Assert.True(CompilerServices.TryMatchRoot(matmul, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = Assert.IsAssignableFrom<Expr>(rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() }));

        Assert.True(CompilerServices.InferenceType(rewritten), CompilerServices.Print(rewritten));
        Assert.Equal(new long[] { 2, 4 }, rewritten.CheckedShape.ToValueArray());
    }

    [Fact]
    public void QuantMatmulInt8RejectsDynamicRhs()
    {
        var inputA = new Var("inputA", new TensorType(DataTypes.Float32, new long[] { 2, 3 }));
        var inputB = new Var("inputB", new TensorType(DataTypes.Float32, new long[] { 3, 4 }));
        var markedA = RangeOfMarker(inputA, Tensor.FromScalar(0.5f), DataTypes.Int8);
        var markedB = RangeOfMarker(inputB, Tensor.FromScalar(0.25f), DataTypes.Int8);
        var matmul = MatMul(markedA, markedB);

        Assert.True(CompilerServices.InferenceType(matmul), CompilerServices.Print(matmul));

        var rule = new QuantizerMatmul();
        Assert.True(CompilerServices.TryMatchRoot(matmul, rule.Pattern, new MatchOptions(), out var match));
        Assert.Null(rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() }));
    }
}
