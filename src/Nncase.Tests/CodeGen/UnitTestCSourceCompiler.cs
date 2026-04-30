// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Reflection;
using Nncase.CodeGen.NTT;
using Xunit;

namespace Nncase.Tests.CodeGen;

public sealed class UnitTestCSourceCompiler
{
    [Theory]
    [InlineData("sm_90a", "90a")]
    [InlineData("compute_100a", "100a")]
    [InlineData("90", "90")]
    [InlineData("", "80")]
    public void NormalizeCudaArchitecturePreservesSuffix(string input, string expected)
    {
        Assert.Equal(expected, CSourceCompiler.NormalizeCudaArchitecture(input));
    }

    [Fact]
    public void CudaCmakeArgumentsPreserveArchitectureSuffix()
    {
        var compiler = new CSourceCompiler(true, "nvcc", "sm_90a");
        var argumentsSpecific = typeof(CSourceCompiler).GetMethod("ArgumentsSpecific", BindingFlags.Instance | BindingFlags.NonPublic)!;

        var arguments = Assert.IsType<string>(argumentsSpecific.Invoke(compiler, new object[] { "src", "out" }));

        Assert.Contains("-DCMAKE_CUDA_ARCHITECTURES=90a", arguments, StringComparison.Ordinal);
        Assert.DoesNotContain("-DCMAKE_CUDA_ARCHITECTURES=90 ", arguments, StringComparison.Ordinal);
    }
}
