// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Threading.Tasks;
using Nncase;
using Nncase.IR;
using Nncase.IR.F;
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

    [Fact]
    public async Task StoreSequenceBecomesTupleDagRoot()
    {
        var (ptr, value, mask, store) = CreateStore("ordered");
        var secondValue = new Var("ordered_second_value", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var secondStore = IR.F.Triton.Store(ptr, secondValue, mask);
        var body = new Sequential(new Expr[] { store, secondStore }, new IVar[] { ptr, value, secondValue, mask });
        var primFunction = new PrimFunction("ordered_store", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        var convertedBody = Assert.IsType<IR.Tuple>(converted.Body.Body);
        Assert.Equal(new Expr[] { store, secondStore }, convertedBody.Fields.ToArray());
        Assert.DoesNotContain("Sequential", CompilerServices.Print(converted), StringComparison.Ordinal);
    }

    [Fact]
    public async Task TerminalVoidReturnUsesMemoryEffectRoot()
    {
        var (ptr, value, mask, store) = CreateStore("terminal_void");
        var body = new Sequential(new Expr[] { store, new Return(Array.Empty<Expr>()) }, new IVar[] { ptr, value, mask });
        var primFunction = new PrimFunction("terminal_void_store", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        Assert.Same(store, converted.Body.Body);
        Assert.DoesNotContain("Sequential", CompilerServices.Print(converted), StringComparison.Ordinal);
    }

    [Fact]
    public async Task PureNoReturnBodyUsesTerminalValue()
    {
        var lhs = new Var("lhs", TensorType.Scalar(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Scalar(DataTypes.Float32));
        var unused = IR.F.Math.Binary(BinaryOp.Sub, lhs, rhs);
        var sum = IR.F.Math.Binary(BinaryOp.Add, lhs, rhs);
        var body = new Sequential(new Expr[] { unused, sum }, new IVar[] { lhs, rhs });
        var primFunction = new PrimFunction("pure_no_return", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        Assert.Same(sum, converted.Body.Body);
        Assert.DoesNotContain("Sequential", CompilerServices.Print(converted), StringComparison.Ordinal);
    }

    [Fact]
    public async Task PureHelperReturnCanUseTerminalValue()
    {
        var lhs = new Var("lhs", TensorType.Scalar(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Scalar(DataTypes.Float32));
        var sum = IR.F.Math.Binary(BinaryOp.Add, lhs, rhs);
        var body = new Sequential(new Expr[] { sum, new Return(new Expr[] { sum }) }, new IVar[] { lhs, rhs });
        var primFunction = new PrimFunction("pure_helper_return", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        Assert.Same(sum, converted.Body.Body);
        Assert.True(CompilerServices.InferenceType(converted), CompilerServices.Print(converted));
    }

    [Fact]
    public async Task GuardedStoreBecomesDagRoot()
    {
        var (ptr, value, mask, store) = CreateStore("guarded");
        var cond = new Var("guarded_cond", TensorType.Scalar(DataTypes.Boolean));
        var guardedStore = new If(cond, new IRBlock(store), new IRBlock(new IR.Tuple()));
        var body = new Sequential(new Expr[] { guardedStore }, new IVar[] { ptr, value, mask, cond });
        var primFunction = new PrimFunction("guarded_store", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        Assert.Same(guardedStore, converted.Body.Body);
        Assert.DoesNotContain("Sequential", CompilerServices.Print(converted), StringComparison.Ordinal);
    }

    [Fact]
    public async Task StoreMixedWithHelperReturnUsesSsaDependency()
    {
        var (ptr, value, mask, store) = CreateStore("mixed");
        var secondValue = new Var("mixed_second_value", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var firstStore = Assert.IsType<Call>(store);
        var secondStore = IR.F.Triton.Store((Expr)firstStore[Nncase.IR.Triton.Store.Ptr], secondValue, mask);
        var body = new Sequential(new Expr[] { store, secondStore, new Return(new Expr[] { value }) }, new IVar[] { ptr, value, secondValue, mask });
        var primFunction = new PrimFunction("mixed_store_return", CUDATarget.Kind, body);

        var converted = Assert.IsType<Function>(await new TTIRToIRPass().RunAsync(primFunction, new RunPassContext()));

        var dependency = Assert.IsType<Call>(converted.Body.Body);
        Assert.IsType<Nncase.IR.Tensors.Depend>(dependency.Target);
        var memoryDependencies = Assert.IsType<IR.Tuple>(dependency[Nncase.IR.Tensors.Depend.Dependencies]);
        Assert.Equal(new Expr[] { store, secondStore }, memoryDependencies.Fields.ToArray());
        Assert.Same(value, dependency[Nncase.IR.Tensors.Depend.Value]);
        Assert.DoesNotContain("Sequential", CompilerServices.Print(converted), StringComparison.Ordinal);
        Assert.True(CompilerServices.InferenceType(converted), CompilerServices.Print(converted));
        var callableType = Assert.IsType<CallableType>(converted.CheckedType);
        Assert.Equal(value.CheckedType, callableType.ReturnType);
    }

    private static (Var Ptr, Var Value, Var Mask, Expr Store) CreateStore(string name)
    {
        var ptr = new Var($"{name}_ptr", TensorType.Pointer(DataTypes.Float32));
        var value = new Var($"{name}_value", new TensorType(DataTypes.Float32, new RankedShape(4)));
        var offsets = Tensors.Range((Const)0, (Const)4, (Const)1);
        var ptrExpr = IR.F.Math.Binary(BinaryOp.Add, ptr, offsets);
        var mask = new Var($"{name}_mask", new TensorType(DataTypes.Boolean, new RankedShape(4)));
        var store = IR.F.Triton.Store(ptrExpr, value, mask);
        return (ptr, value, mask, store);
    }
}
