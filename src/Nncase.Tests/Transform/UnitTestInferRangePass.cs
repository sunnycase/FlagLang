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
}
