// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Linq;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.Passes;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Xunit;

namespace Nncase.Tests.TIRTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestNTTAffineSelectionPass : TestClassBase
{
    public UnitTestNTTAffineSelectionPass()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions();
    }

    [Fact]
    public async Task GatherSelectsAffineGridWithAxisScopedInputRegionAndTiledIndex()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 8)));
        var index = new Var("index", new TensorType(DataTypes.Int64, new RankedShape(2)));
        var gather = IR.F.Tensors.Gather(input, 1, index);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(gather, input, index));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        var grid = Assert.IsType<Grid>(selected.Body.Body);
        var inputMap = grid.AccessMaps[0];
        var indexMap = grid.AccessMaps[1];
        var bodyCall = Assert.Single(grid.Body.Fields.ToArray().OfType<Call>());

        Assert.IsType<Nncase.TIR.NTT.Gather>(bodyCall.Target);
        AssertDomainRange(inputMap.Results[0], 0);
        AssertFullRange(inputMap.Results[1], 0, 8);
        AssertDomainRange(indexMap.Results[0], 1);
    }

    [Fact]
    public async Task ScatterNDSelectsAffineGridWithFullRandomUpdateRegions()
    {
        var input = new Var("input", new TensorType(DataTypes.Float32, new RankedShape(4, 8)));
        var indices = new Var("indices", new TensorType(DataTypes.Int64, new RankedShape(3, 1)));
        var updates = new Var("updates", new TensorType(DataTypes.Float32, new RankedShape(3, 8)));
        var scatter = IR.F.Tensors.ScatterND(input, indices, updates);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, input, indices, updates));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        var grid = Assert.IsType<Grid>(selected.Body.Body);
        var inputMap = grid.AccessMaps[0];
        var indicesMap = grid.AccessMaps[1];
        var updatesMap = grid.AccessMaps[2];
        var outputMap = grid.AccessMaps[3];
        var bodyCall = Assert.Single(grid.Body.Fields.ToArray().OfType<Call>());

        Assert.IsType<Nncase.TIR.NTT.ScatterND>(bodyCall.Target);
        AssertDomainRange(inputMap.Results[0], 0);
        AssertDomainRange(inputMap.Results[1], 1);
        AssertFullRange(indicesMap.Results[0], 0, 3);
        AssertFullRange(indicesMap.Results[1], 0, 1);
        AssertFullRange(updatesMap.Results[0], 0, 3);
        AssertFullRange(updatesMap.Results[1], 0, 8);
        AssertDomainRange(outputMap.Results[0], 0);
        AssertDomainRange(outputMap.Results[1], 1);
    }

    [Fact]
    public async Task DirectAffineIOChainSelectsGatherBinaryScatterAsGridDAG()
    {
        const int blockSize = 128;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            System.Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(System.Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var sum = left + right;
        var scatter = IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var selected = Assert.IsType<Function>(await new NTTAffineSelectionPass(CompileOptions, CUDATarget.Kind).RunAsync(function, new()));
        Assert.True(CompilerServices.InferenceType(selected), CompilerServices.Print(selected));
        var selectedScatter = Assert.IsType<Grid>(selected.Body.Body);
        Assert.False(selectedScatter.HasOutput);
        Assert.IsType<Nncase.TIR.NTT.AffineScatter>(Assert.Single(selectedScatter.Body.Fields.ToArray().OfType<Call>()).Target);

        var selectedSum = Assert.IsType<Grid>(selectedScatter.Reads[0]);
        Assert.True(selectedSum.HasOutput);
        Assert.IsType<Nncase.TIR.NTT.VectorizedBinary>(Assert.Single(selectedSum.Body.Fields.ToArray().OfType<Call>()).Target);

        var selectedLeft = Assert.IsType<Grid>(selectedSum.Reads[0]);
        var selectedRight = Assert.IsType<Grid>(selectedSum.Reads[1]);
        Assert.True(selectedLeft.HasOutput);
        Assert.True(selectedRight.HasOutput);
        Assert.All(new[] { selectedLeft, selectedRight }, grid => Assert.IsType<Nncase.TIR.NTT.AffineGather>(Assert.Single(grid.Body.Fields.ToArray().OfType<Call>()).Target));
    }

    private static void AssertDomainRange(AffineRange range, int position)
    {
        var offset = Assert.IsType<AffineDim>(range.Offset);
        var extent = Assert.IsType<AffineExtent>(range.Extent);
        Assert.Equal(position, offset.Position);
        Assert.Equal(position, extent.Position);
    }

    private static void AssertFullRange(AffineRange range, long offsetValue, long extentValue)
    {
        var offset = Assert.IsType<AffineConstant>(range.Offset);
        var extent = Assert.IsType<AffineConstant>(range.Extent);
        Assert.Equal(offsetValue, offset.Value);
        Assert.Equal(extentValue, extent.Value);
    }
}
