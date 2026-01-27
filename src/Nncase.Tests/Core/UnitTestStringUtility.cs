// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Immutable;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Utilities;
using Xunit;
using Xunit.Abstractions;

namespace Nncase.Tests.CoreTest;

public static class TestExtensions
{
    public static ArrayExtensions.SpanWhereEnumerable<TIR.Buffer, FunctionWrapper<TIR.Buffer, bool>> InputOf(this ReadOnlySpan<TIR.Buffer> arr) => arr.AsValueEnumerable().Where(b => b.MemSpan.Buffer.Location == MemoryLocation.Input);

    public static ArrayExtensions.SpanWhereEnumerable<TIR.Buffer, FunctionWrapper<TIR.Buffer, bool>> OutputOf(this ReadOnlySpan<TIR.Buffer> arr) => arr.AsValueEnumerable().Where(b => b.MemSpan.Buffer.Location == MemoryLocation.Output);
}

public sealed class UnitTestStringUtility
{
    private readonly TIR.PrimFunction _entry = CreateEntry();

    [Fact]
    public void TestJoin()
    {
        /* var result = StringUtility.Join(",", _entry.Parameters.InputOf().Select(b => b));
        Assert.Equal("Nncase.TIR.Buffer", result);
        var result1 = StringUtility.Join(",", _entry.Parameters.OutputOf().Select(b => b));
        Assert.Equal("Nncase.TIR.Buffer", result1); */
    }

    private static TIR.PrimFunction CreateEntry()
    {
        var input = new Var("testInput", new TensorType(DataTypes.Float32, new[] { 1, 16, 64, 400 }));
        var output = new Var("testOutput", new TensorType(DataTypes.Float32, new[] { 1, 16, 64, 400 }));
        var body = new Sequential(Array.Empty<Expr>(), new IVar[] { input, output });
        return new TIR.PrimFunction("test_module", BaseFunction.CPUModuleKind, body);
    }
}
