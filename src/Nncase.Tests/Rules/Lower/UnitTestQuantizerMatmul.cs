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
}
