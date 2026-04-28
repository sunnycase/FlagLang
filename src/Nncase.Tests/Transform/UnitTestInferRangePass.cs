// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Threading.Tasks;
using Nncase;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestInferRangePass : TestClassBase
{
    [Fact]
    public async Task EmptyTupleHasFullRange()
    {
        var tuple = new IR.Tuple();
        var function = new Function("main", new IRBlock(tuple));

        var result = (Function)await new InferRangePass().RunAsync(function, new RunPassContext());

        Assert.Equal(ValueRange<double>.Full, tuple.Metadata.Range);
        Assert.Equal(ValueRange<double>.Full, result.Body.Body.Metadata.Range);
    }

    [Fact]
    public async Task AbsRangeCrossingZeroIncludesZero()
    {
        var input = new Var("input", TensorType.Scalar(DataTypes.Float32));
        input.Metadata.Range = new ValueRange<double>(-5, 3);
        var abs = IR.F.Math.Abs(input);
        var function = new Function("main", new IRBlock(abs, input));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        await new InferRangePass().RunAsync(function, new RunPassContext());

        Assert.Equal(new ValueRange<double>(0, 5), abs.Metadata.Range);
    }

    [Fact]
    public async Task IntegralModRangeUsesConservativeResidueBound()
    {
        var lhs = new Var("lhs", TensorType.Scalar(DataTypes.Int32));
        var rhs = new Var("rhs", TensorType.Scalar(DataTypes.Int32));
        lhs.Metadata.Range = new ValueRange<double>(0, 10);
        rhs.Metadata.Range = new ValueRange<double>(3, 3);
        var mod = IR.F.Math.Mod(lhs, rhs);
        var function = new Function("main", new IRBlock(mod, lhs, rhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        await new InferRangePass().RunAsync(function, new RunPassContext());

        Assert.Equal(new ValueRange<double>(0, 2), mod.Metadata.Range);
    }

    [Fact]
    public async Task SignedIntegralModRangePreservesDividendSignPossibility()
    {
        var lhs = new Var("lhs", TensorType.Scalar(DataTypes.Int32));
        var rhs = new Var("rhs", TensorType.Scalar(DataTypes.Int32));
        lhs.Metadata.Range = new ValueRange<double>(-10, 10);
        rhs.Metadata.Range = new ValueRange<double>(3, 3);
        var mod = IR.F.Math.Mod(lhs, rhs);
        var function = new Function("main", new IRBlock(mod, lhs, rhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        await new InferRangePass().RunAsync(function, new RunPassContext());

        Assert.Equal(new ValueRange<double>(-2, 2), mod.Metadata.Range);
    }
}
