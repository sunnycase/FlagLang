// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Logics;
using Nncase.IR.Math;
using Nncase.Passes;
using Nncase.Passes.Rules.Triton;
using Nncase.PatternMatch;
using Nncase.Tests.TestFixture;
using Xunit;
using static Nncase.IR.F.Math;
using static Nncase.IR.F.Tensors;
using static Nncase.IR.F.Triton;
using AffineGather = Nncase.IR.Affine.Gather;
using AffineScatter = Nncase.IR.Affine.Scatter;

namespace Nncase.Tests.TransformTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTensorizeIO : TransformTestBase
{
    private static readonly TensorType PointerF32 = TensorType.Pointer(DataTypes.Float32);

    private static readonly TensorType ScalarI32 = TensorType.Scalar(DataTypes.Int32);

    private static readonly TensorType VectorF32 = new TensorType(DataTypes.Float32, new RankedShape(4));

    [Fact]
    public void LoadToAffineGatherRewritesLinearPointer()
    {
        var ptrBase = new Var("ptr_base", PointerF32);
        var lane = new Var("lane", ScalarI32);
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var addrDelta = Binary(BinaryOp.Add, offsets, lane);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, addrDelta);
        var mask = Compare(CompareOp.LowerThan, addrDelta, (Const)64);
        var defaultValue = Const.FromTensor(Tensor.From<float>(new float[4], new long[] { 4 }));
        var load = Load(ptrExpr, mask, defaultValue);
        var function = new Function("main", new IRBlock(load, ptrBase, lane));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));
        var loadShape = ((TensorType)load.CheckedType).Shape;

        var rule = new LoadToAffineGather();
        Assert.True(CompilerServices.TryMatchRoot(load, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() });
        var gatherCall = Assert.IsType<Call>(rewritten);
        var gatherOp = Assert.IsType<AffineGather>(gatherCall.Target);

        Assert.Same(ptrBase, gatherCall[AffineGather.Source]);
        Assert.Same(defaultValue, gatherCall[AffineGather.DefaultValue]);
        Assert.Equal(loadShape, gatherOp.Shape);

        var relation = gatherOp.Relation;
        Assert.Equal(2, relation.Domains.Length);
        Assert.Equal(0, relation.Symbols.Length);
        Assert.Equal(1, relation.Results.Length);
        Assert.NotSame(LogicalExpr.True, relation.Constraint);
    }

    [Fact]
    public void StoreToAffineScatterRewritesLinearPointer()
    {
        var ptrBase = new Var("ptr_base", PointerF32);
        var lane = new Var("lane", ScalarI32);
        var value = new Var("value", VectorF32);
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var addrDelta = Binary(BinaryOp.Add, offsets, lane);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, addrDelta);
        var mask = Compare(CompareOp.LowerThan, addrDelta, (Const)64);
        var store = Store(ptrExpr, value, mask);
        var function = new Function("main", new IRBlock(store, ptrBase, lane, value));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var rule = new StoreToAffineScatter();
        Assert.True(CompilerServices.TryMatchRoot(store, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() });
        var scatterCall = Assert.IsType<Call>(rewritten);
        var scatterOp = Assert.IsType<AffineScatter>(scatterCall.Target);

        Assert.Same(value, scatterCall[AffineScatter.Source]);
        Assert.Same(ptrBase, scatterCall[AffineScatter.Dest]);

        var relation = scatterOp.Relation;
        Assert.Equal(2, relation.Domains.Length);
        Assert.Equal(0, relation.Symbols.Length);
        Assert.Equal(1, relation.Results.Length);
        Assert.NotSame(LogicalExpr.True, relation.Constraint);
    }
}
