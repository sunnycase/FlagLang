// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
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
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, offsets);
        var mask = Compare(CompareOp.LowerThan, offsets, (Const)64);
        var defaultValue = Const.FromTensor(Tensor.From<float>(new float[4], new long[] { 4 }));
        var load = Load(ptrExpr, mask, defaultValue);
        var function = new Function("main", new IRBlock(load, ptrBase));
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
        Assert.Equal(1, relation.Domains.Length);
        Assert.Equal(0, relation.Symbols.Length);
        Assert.Equal(1, relation.Results.Length);
        Assert.Equal(new ValueRange<double>(0, 3), relation.Domains[0].Metadata.Range);
        Assert.NotSame(LogicalExpr.True, relation.Constraint);
    }

    [Fact]
    public void StoreToAffineScatterRewritesLinearPointer()
    {
        var ptrBase = new Var("ptr_base", PointerF32);
        var value = new Var("value", VectorF32);
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, offsets);
        var mask = Compare(CompareOp.LowerThan, offsets, (Const)64);
        var store = Store(ptrExpr, value, mask);
        var function = new Function("main", new IRBlock(store, ptrBase, value));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var rule = new StoreToAffineScatter();
        Assert.True(CompilerServices.TryMatchRoot(store, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() });
        var scatterCall = Assert.IsType<Call>(rewritten);
        var scatterOp = Assert.IsType<AffineScatter>(scatterCall.Target);

        Assert.Same(value, scatterCall[AffineScatter.Source]);
        Assert.Same(ptrBase, scatterCall[AffineScatter.Dest]);

        var relation = scatterOp.Relation;
        Assert.Equal(1, relation.Domains.Length);
        Assert.Equal(0, relation.Symbols.Length);
        Assert.Equal(1, relation.Results.Length);
        Assert.NotSame(LogicalExpr.True, relation.Constraint);
    }

    [Fact]
    public void LoadToAffineGatherModelsVectorAddSymbolsAndLaneDomain()
    {
        const int blockSize = 4;
        var ptrBase = new Var("ptr_base", PointerF32);
        var nElements = new Var("n_elements", ScalarI32);
        var programId = Cast(Nncase.IR.F.Shapes.AsTensor(Nncase.IR.F.Distributed.ProgramId(0)), DataTypes.Int32);
        var blockStart = Binary(BinaryOp.Mul, programId, (Const)blockSize);
        var offsets = Range((Const)0, (Const)blockSize, (Const)1);
        var addrDelta = Binary(BinaryOp.Add, blockStart, offsets);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, addrDelta);
        var mask = Compare(CompareOp.LowerThan, addrDelta, nElements);
        var defaultValue = Const.FromTensor(Tensor.From<float>(new float[blockSize], new long[] { blockSize }));
        var load = Load(ptrExpr, mask, defaultValue);
        var function = new Function("main", new IRBlock(load, ptrBase, nElements));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var rule = new LoadToAffineGather();
        Assert.True(CompilerServices.TryMatchRoot(load, rule.Pattern, new MatchOptions(), out var match));
        var rewritten = rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() });
        var gatherCall = Assert.IsType<Call>(rewritten);
        var gatherOp = Assert.IsType<AffineGather>(gatherCall.Target);

        Assert.Same(ptrBase, gatherCall[AffineGather.Source]);
        Assert.Equal(1, gatherOp.Relation.Domains.Length);
        Assert.Equal(2, gatherOp.Relation.Symbols.Length);
        Assert.Equal(1, gatherOp.Relation.Results.Length);
        Assert.IsType<ProgramIdDim>(gatherOp.Symbols[0]);
        var symbolDim = Assert.IsType<DimVar>(gatherOp.Symbols[1]);
        Assert.Equal("n_elements", symbolDim.Name);
        Assert.Equal(new ValueRange<double>(0, blockSize - 1), gatherOp.Relation.Domains[0].Metadata.Range);
        Assert.Equal("(d0)[s0, s1] -> (((s0 * 4) + d0))", gatherOp.Relation.With(constraint: LogicalExpr.True).ToString());
        Assert.Contains("< n_elements", gatherOp.Relation.Constraint.ToString(), System.StringComparison.Ordinal);
    }

    [Fact]
    public void ReadDimRejectsMultiplePointerBases()
    {
        var ptrBaseA = new Var("ptr_base_a", PointerF32);
        var ptrBaseB = new Var("ptr_base_b", PointerF32);
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var ptrExpr = Binary(BinaryOp.Add, ptrBaseA, Binary(BinaryOp.Add, ptrBaseB, offsets));

        var generator = new TritonAffineUtility.ReadDimGenerator();
        Assert.Null(generator.GeneratePtrBaseAndReadDim(ptrExpr));
    }

    [Fact]
    public void LoadToAffineGatherRejectsNonAffineMask()
    {
        var ptrBase = new Var("ptr_base", PointerF32);
        var offsets = Range((Const)0, (Const)4, (Const)1);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, offsets);
        var nonlinearMaskValue = Binary(BinaryOp.Mul, offsets, offsets);
        var mask = Compare(CompareOp.LowerThan, nonlinearMaskValue, (Const)64);
        var defaultValue = Const.FromTensor(Tensor.From<float>(new float[4], new long[] { 4 }));
        var load = Load(ptrExpr, mask, defaultValue);
        var function = new Function("main", new IRBlock(load, ptrBase));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var rule = new LoadToAffineGather();
        Assert.True(CompilerServices.TryMatchRoot(load, rule.Pattern, new MatchOptions(), out var match));
        Assert.Null(rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() }));
    }

    [Fact]
    public void LoadToAffineGatherRejectsUnboundedDynamicLaneDomain()
    {
        var ptrBase = new Var("ptr_base", PointerF32);
        var blockSize = new Var("block_size", ScalarI32);
        var offsets = Range((Const)0, blockSize, (Const)1);
        var ptrExpr = Binary(BinaryOp.Add, ptrBase, offsets);
        var mask = Compare(CompareOp.LowerThan, offsets, (Const)64);
        var defaultValue = Const.FromTensor(Tensor.From<float>(new float[4], new long[] { 4 }));
        var load = Load(ptrExpr, mask, defaultValue);
        var function = new Function("main", new IRBlock(load, ptrBase, blockSize));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var rule = new LoadToAffineGather();
        Assert.True(CompilerServices.TryMatchRoot(load, rule.Pattern, new MatchOptions(), out var match));
        Assert.Null(rule.GetReplace(match!, new RunPassContext { Driver = new DataflowPass() }));
    }
}
