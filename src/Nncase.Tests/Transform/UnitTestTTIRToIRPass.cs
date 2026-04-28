// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Threading.Tasks;
using Nncase;
using Nncase.IR;
using Nncase.Passes;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTTIRToIRPass : TestClassBase
{
    [Fact]
    public async Task HelperReturnValueSurvivesTTIRToIR()
    {
        var input = new Var("param_0", new TensorType(DataTypes.Float32, new[] { 1 }));
        var body = new Sequential(new Expr[] { new Return(new Expr[] { input }) }, new IVar[] { input });
        var primFunction = new PrimFunction("identity", CUDATarget.Kind, body);
        Assert.True(CompilerServices.InferenceType(primFunction), CompilerServices.Print(primFunction));

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        Assert.Same(input, converted.Body.Body);
        Assert.True(CompilerServices.InferenceType(converted), CompilerServices.Print(converted));
        var callableType = Assert.IsType<CallableType>(converted.CheckedType);
        Assert.Equal(input.CheckedType, callableType.ReturnType);
    }
}
