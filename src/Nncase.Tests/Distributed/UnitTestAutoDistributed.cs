// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.Passes.Distributed;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Utilities;
using Xunit;

namespace Nncase.Tests.DistributedTest;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestDistribAutoDistributed : TestClassBase
{
    public UnitTestDistribAutoDistributed()
    {
        DefaultTargetName = CPUTarget.Kind;
        CompileOptions.TargetOptions = new NTTTargetOptions();
#if DEBUG
        CompileOptions.DumpFlags = DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.EGraphCost | DumpFlags.CodeGen | DumpFlags.Compile;
#endif
    }

    [Fact]
    public void TestDistributeBinary()
    {
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [32, 1]));
        var rhs = new Var("rhs", new TensorType(DataTypes.Float32, [16]));
        var main = new Function("main", new IRBlock(lhs + rhs, lhs, rhs));
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);
        pass.RunAsync(main, new()).Wait();
    }

    [Fact]
    public void TestDistributeDynamicBinaryWithRhsVector()
    {
        var dimX = new DimVar("dimX") { Metadata = { Range = (1, 256) } };
        var lhs = new Var("lhs", new TensorType(DataTypes.Float32, [dimX, 1]));
        var rhs = new Var("rhs", new TensorType(new VectorType(DataTypes.Float32, [8]), [16]));
        var main = new Function("main", new IRBlock(lhs + rhs, lhs, rhs));
        var pass = new AutoDistributedPass(false, CPUTarget.Kind, CompileOptions);
        pass.RunAsync(main, new()).Wait();
    }

    [Fact]
    public async Task TestDistributeAffineScatterTerminator()
    {
        var targetOptions = (NTTTargetOptions)CompileOptions.TargetOptions;
        targetOptions.HierarchyNames = "t";
        targetOptions.Hierarchies = [[8]];

        const int blockSize = 32;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var rhs = new Var("rhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default);
        var sum = left + right;
        var scatter = Nncase.IR.F.Affine.Scatter(sum, dest, relation, symbols);
        var main = new Function("main", new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(main), CompilerServices.Print(main));

        var pass = new AutoDistributedPass(false, CUDATarget.Kind, CompileOptions);
        var result = Assert.IsType<Function>(await pass.RunAsync(main, new()));
        var call = Assert.IsType<Call>(result.Body.Body);
        Assert.IsType<Nncase.IR.Affine.Scatter>(call.Target);
        var sourceType = Assert.IsType<DistributedType>(call.Arguments[0].CheckedType);
        Assert.Contains(sourceType.AxisPolicies, sbp => sbp is SBPSplit split && split.Axes.Contains(0));
        var printed = CompilerServices.Print(result);
        Assert.Contains("Dist:", printed, StringComparison.Ordinal);
        Assert.DoesNotContain("Boxing", printed, StringComparison.Ordinal);
        Assert.DoesNotContain("Sequential", printed, StringComparison.Ordinal);
    }
}
