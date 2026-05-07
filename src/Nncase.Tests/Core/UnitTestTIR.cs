// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.Evaluator;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.Targets;
using Nncase.TIR;
using Nncase.TIR.Builders;
using OrtKISharp;
using Xunit;
using Buffer = Nncase.TIR.Buffer;
using Function = Nncase.IR.Function;
using Range = Nncase.TIR.Range;

namespace Nncase.Tests.CoreTest;

public sealed class UnitTestTIR
{
    [Fact]
    public void TestBufferStore()
    {
        Expr value = 42;
        TIR.T.CreateBuffer(new TensorType(DataTypes.Float32, new[] { 1, 16, 64, 400 }), BufferStorage.GlobalInput(), out var testInput);
        _ = new Expr[] { 0, 1 };
        _ = T.Store(testInput, 0, value);
    }

    [Fact]
    public void DistributedBufferUsesCanonicalIrType()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var distributedType = new DistributedType(tensorType, new SBP[] { SBP.S(0) }, new Placement(new[] { 128 }, "t"));
        var buffer = TIR.T.CreateBuffer(distributedType, BufferStorage.ThreadLocalTemp(), out _, "buffer");
        var subview = IR.F.Buffer.BufferSubview(buffer, new RankedShape(0), new RankedShape(8));

        Assert.Equal(distributedType, buffer.Type);
        Assert.Equal(tensorType, buffer.TensorType);
        Assert.True(CompilerServices.InferenceType(subview));

        var subviewType = Assert.IsType<TensorType>(subview.CheckedType);
        Assert.Equal(new RankedShape(8), subviewType.Shape);
    }

    [Fact]
    public void TestIterVar()
    {
        var dom = new Range(-1, 1, 1);
        var mode = IterationMode.Opaque;
        var value = new DimVar("test");
        var iterVar = new IterVar(dom, mode, value);
        Assert.Equal(dom, iterVar.Dom);
        Assert.Equal(mode, iterVar.Mode);
        Assert.Equal(value, iterVar.Value);
    }

    [Fact]
    public void TestSizeVar()
    {
        var name = "test";
        var actual = T.SizeVar(name);
        var expected = new DimVar(name);
        Assert.Equal(expected.ToString(), actual.ToString());
    }

    [Fact]
    public void TestSerial()
    {
        var domain = new Range(-1, 1, 1);
        var actual = T.Serial(out _, domain);
        var expect = T.ForLoop(out _, domain, LoopMode.Serial, "v");
        Assert.Equal(expect.ToString(), actual.ToString());
    }

    [Fact]
    public void TestSequential()
    {
        var expect1 = new SequentialBuilder<Sequential>(body => body);
        var actual1 = T.Sequential();
        Assert.Equal(expect1.ToString(), actual1.ToString());

        var expect2 = TIR.Sequential.Flatten(Array.Empty<object>());
        var actual2 = T.Sequential(Array.Empty<object>());
        Assert.Equal(expect2, actual2);
    }

    [Fact]
    public void TestForSegment()
    {
        var count = 2 / 2;
        var expect = T.Serial(out _, (0L, count));
        var actual = T.ForSegment(out _, 1L, 2L, 3L);
        Assert.Equal(expect.ToString(), actual.ToString());
    }

    [Fact]
    public void TestGrid()
    {
        var grid1 = T.Grid(out _, LoopMode.Serial, new Range(-1, 1, 1));
        var grid2 = T.Grid(out _, LoopMode.Serial, new Range(1, 1, 1));
        Assert.Equal(grid1.GetType(), grid2.GetType());
    }

    [Fact]
    public void TestQwen3MoEParameterMetadataMatchesFunctionalCall()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 1 });
        var args = Enumerable.Range(0, 12).Select(i => (Expr)new Var($"arg{i}", type)).ToArray();
        var call = Assert.IsType<Call>(Nncase.TIR.F.NTT.Qwen3MoE(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            args[10],
            args[11],
            0,
            1,
            1,
            1,
            1,
            1,
            1));
        var op = Assert.IsType<Nncase.TIR.NTT.Qwen3MoE>(call.Target);
        var parameters = op.Parameters.ToArray();

        Assert.Equal(call.Arguments.Length, parameters.Length);
        Assert.Same(Nncase.TIR.NTT.Qwen3MoE.Output, parameters[11]);
        Assert.Equal("output", parameters[11].Name);
        call.ParametersForeach((_, _) => { });
    }

    [Fact]
    public void TestQwen3MoEIsHostOnlyForNTTModuleCompilers()
    {
        var type = new TensorType(DataTypes.Float32, new[] { 1 });
        var args = Enumerable.Range(0, 11).Select(i => (Expr)new Var($"arg{i}", type)).ToArray();
        var call = Assert.IsType<Call>(IR.F.NN.Qwen3MoE(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            args[7],
            args[8],
            args[9],
            args[10],
            0,
            1,
            1,
            1,
            1,
            1,
            1));
        CompilerServices.InferenceType(call);
        var options = new CompileOptions();

        Assert.True(new CPUModuleCompiler().IsSupportedCall(call, options));
        Assert.False(new CUDAModuleCompiler().IsSupportedCall(call, options));
    }

    [Fact]
    public void TestEmit()
    {
        int result;
        T.Emit(out result, () => 5);
        Assert.Equal(5, result);
    }

    [Fact]
    public void TestBufferRegion()
    {
        var buffer = T.CreateBuffer(new(DataTypes.Float32, new[] { 1, 16, 64, 400 }), BufferStorage.GlobalInput(), out _);
        var region = new Range[] { new Range(1, 2, 2), new Range(-1, 3, 2) };
        var bufferRegion = new BufferRegion(buffer, region);

        var newRegion = bufferRegion[new Range(0, 1, 2), new Range(-3, 3, 2)];
        Assert.Equal(buffer, newRegion.Buffer);
        Assert.Equal(new Range(0, 1, 2), newRegion.Region[0]);
        Assert.Equal(new Range(-3, 3, 2), newRegion.Region[1]);
    }

    [Fact]
    public void TestNop()
    {
        var nop = new Nop();
        Assert.False(nop.CanFoldConstCall);
    }

    [Fact]
    public void TestPrimFunction()
    {
        var parameters = new IVar[]
        {
            TIR.T.CreateBufferVar(new(DataTypes.Float32, new[] { 1, 16, 64, 400 }), out var _),
            TIR.T.CreateBufferVar(new(DataTypes.Float32, new[] { 1, 16, 64, 400 }), out var _),
        };

        var primFunc = new PrimFunction("test_module", BaseFunction.CPUModuleKind, new Sequential(new Expr[] { 1 }, parameters));

        var primFuncParameters = primFunc.Parameters;
        var primFuncParameterTypes = primFunc.ParameterTypes;
        var expect = primFuncParameters.AsValueEnumerable().Select(x => x.CheckedType).ToArray();
        Assert.Equal(expect, primFuncParameterTypes);

        var newModuleKind = "new_module";
        var newParams = new[]
        {
            TIR.T.CreateBufferVar(new(DataTypes.Float32, new[] { 1, 16, 64, 400 }), out var _),
            TIR.T.CreateBufferVar(new(DataTypes.Float32, new[] { 1, 16, 64, 400 }), out var _),
        };
        var newBody = new Sequential(new Expr[] { 3 }, newParams);

        var newPrimFunc = primFunc.With(moduleKind: newModuleKind, body: newBody);

        Assert.NotSame(primFunc, newPrimFunc);
        Assert.Equal(newModuleKind, newPrimFunc.ModuleKind);
        Assert.Equal(newBody, newPrimFunc.Body);
        Assert.Equal(newParams, newPrimFunc.Parameters.ToArray());
        Assert.Equal(primFunc.Name, newPrimFunc.Name); // should not change the name

        Assert.NotNull(new PrimFunction("test_module", BaseFunction.CPUModuleKind, new Sequential(new Expr[] { 1 })));
    }

    [Fact]
    public void TestTIRExtensions()
    {
        var list = new List<Expr>();
        list.Add(1);
        list.Add(2);
        list.Add(3);

        var seq = list.ToSequential();

        Assert.Equal(3, seq.Count);
        Assert.Equal(1, seq[0]);
        Assert.Equal(2, seq[1]);
        Assert.Equal(3, seq[2]);
    }

    [Fact]
    public void TestRange()
    {
        var expectedStart = long.MinValue;
        var expectedStop = long.MaxValue;
        var expectedStep = 1L;

        var range = Range.All;

        Assert.Equal(expectedStart, range.Start);
        Assert.Equal(expectedStop, range.Stop);
        Assert.Equal(expectedStep, range.Step);

        var range0 = new Range(0, 1, 1);
        Assert.Equal(0L, range0.Start);

        var range1 = range0 + 1L;

        Assert.Equal(1, range1.Start.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(2, range1.Stop.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(1, range1.Step.Evaluate().AsTensor().ToScalar<int>());

        var range2 = range0 - 1L;
        Assert.Equal(-1, range2.Start.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(0, range2.Stop.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(1, range2.Step.Evaluate().AsTensor().ToScalar<int>());

        var range3 = range0 * 2L;
        Assert.Equal(0, range3.Start.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(2, range3.Stop.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(1, range3.Step.Evaluate().AsTensor().ToScalar<int>());

        var range4 = range0 / 2L;
        Assert.Equal(0, range4.Start.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(0, range4.Stop.Evaluate().AsTensor().ToScalar<int>());
        Assert.Equal(1, range4.Step.Evaluate().AsTensor().ToScalar<int>());
    }
}
