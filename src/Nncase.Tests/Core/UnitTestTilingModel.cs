// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Shapes;
using Nncase.Passes.Transforms;
using Nncase.Targets;
using Nncase.Tests.TestFixture;
using Nncase.Tiling;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.Core;

[AutoSetupTestMethod(InitSession = true)]
public sealed class UnitTestTilingModel : TestClassBase
{
    public UnitTestTilingModel()
    {
        CompileOptions.TargetOptions = new NTTTargetOptions();
    }

    [Fact]
    public void BufferStorageDescribesOrthogonalStorage()
    {
        var data = BufferStorage.ThreadLocalTemp();
        Assert.Equal(BufferUsage.Temp, data.Usage);
        Assert.Equal(BufferScope.ThreadLocal, data.Scope);
        Assert.Equal(PhysicalMemorySpace.LocalAddressable, data.PhysicalLocation);

        var smem = BufferStorage.BlockLocalSMem();
        Assert.Equal(BufferUsage.Temp, smem.Usage);
        Assert.Equal(BufferScope.BlockLocal, smem.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, smem.PhysicalLocation);

        var buffer = new PhysicalBuffer(8, (Dimension)64, smem);
        Assert.Equal(smem with { Alignment = 8 }, buffer.Storage);

        T.CreateBuffer(
            new TensorType(DataTypes.Float32, new RankedShape(16)),
            smem,
            out var tirBuffer);
        Assert.Equal(smem with { Alignment = 4 }, tirBuffer.Storage);
    }

    [Fact]
    public void RegisterStorageCannotMaterializeAsPhysicalBuffer()
    {
        var register = new BufferStorage(BufferUsage.Temp, BufferScope.ThreadLocal, PhysicalMemorySpace.Register);
        Assert.False(register.IsAddressable);
        Assert.Throws<InvalidOperationException>(() => new PhysicalBuffer(4, (Dimension)4, register));
    }

    [Fact]
    public void DistributedTypeExposesSbpOwnershipLayout()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var distributedType = new DistributedType(tensorType, [SBP.S(0)], new Placement([128], "t"));

        var layout = distributedType.DistributionLayout;
        var ownerLocalOutputs = layout.GlobalToOwnerLocal.Outputs.Select(x => x.ToString()).ToArray();
        var globalOutputs = layout.OwnerLocalToGlobal.Outputs.Select(x => x.ToString()).ToArray();

        LayoutVerifier.Verify(layout);
        Assert.Equal("SBP", layout.Kind);
        Assert.Equal(new long[] { 8 }, layout.LocalShape.ToValueArray());
        Assert.Contains("owner0=floor(g0/8)%128", ownerLocalOutputs);
        Assert.Contains("l0=g0%8", ownerLocalOutputs);
        Assert.Contains("g0=owner0*8+l0", globalOutputs);

        var storageLayout = distributedType.StorageLayout;
        LayoutVerifier.Verify(layout, storageLayout);
        Assert.Equal("Identity", storageLayout.Kind);
        Assert.Equal(layout.LocalShape, storageLayout.LogicalShape);
    }

    [Fact]
    public void TritonBlockedLayoutCarriesBlockedOwnershipParameters()
    {
        var blocked = new TritonBlockedLayout(
            SizePerThread: 2,
            ThreadsPerWarp: 32,
            WarpsPerCTA: 4,
            Order: [0],
            CTAsPerCGA: [1],
            CTASplitNum: [1],
            CTAOrder: [0]);

        var layout = DistributionLayout.TritonBlocked(new RankedShape(1024), blocked);
        var ownerLocalOutputs = layout.GlobalToOwnerLocal.Outputs.Select(x => x.ToString()).ToArray();
        var globalOutputs = layout.OwnerLocalToGlobal.Outputs.Select(x => x.ToString()).ToArray();

        LayoutVerifier.Verify(layout);
        Assert.Equal("TritonBlocked", layout.Kind);
        Assert.Equal(new long[] { 2 }, layout.LocalShape.ToValueArray());
        Assert.Contains("sizePerThread=2", layout.Attributes!.Value.ToArray());
        Assert.Contains("threadsPerWarp=32", layout.Attributes!.Value.ToArray());
        Assert.Contains("lane=floor(g0%64/2)", ownerLocalOutputs);
        Assert.Contains("elem=g0%2", ownerLocalOutputs);
        Assert.Contains("g0=cta*256+warp*64+lane*2+elem", globalOutputs);
    }

    [Fact]
    public void TritonBlockedLayoutCanDescribeStridedPerThreadOwnership()
    {
        var blocked = new TritonBlockedLayout(
            SizePerThread: 2,
            ThreadsPerWarp: 32,
            WarpsPerCTA: 4,
            Order: [0],
            CTAsPerCGA: [1],
            CTASplitNum: [1],
            CTAOrder: [0],
            ThreadElementOrder: TritonThreadElementOrder.Strided);

        var layout = DistributionLayout.TritonBlocked(new RankedShape(1024), blocked);
        var ownerLocalOutputs = layout.GlobalToOwnerLocal.Outputs.Select(x => x.ToString()).ToArray();
        var globalOutputs = layout.OwnerLocalToGlobal.Outputs.Select(x => x.ToString()).ToArray();

        Assert.Contains("threadElementOrder=Strided", layout.Attributes!.Value.ToArray());
        Assert.Contains("lane=(g0%128)%32", ownerLocalOutputs);
        Assert.Contains("elem=floor(g0%256/128)", ownerLocalOutputs);
        Assert.Contains("g0=cta*256+warp*32+lane+elem*128", globalOutputs);
    }

    [Fact]
    public void DistributedTypeCanCarryExplicitTritonBlockedLayout()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var distributionLayout = DistributionLayout.TritonBlocked(
            tensorType.Shape,
            new TritonBlockedLayout(
                SizePerThread: 2,
                ThreadsPerWarp: 32,
                WarpsPerCTA: 4,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0],
                ThreadElementOrder: TritonThreadElementOrder.Strided));

        var distributedType = DistributedType.FromLayouts(
            tensorType,
            new Placement([1, 4, 32], "cwl"),
            distributionLayout);

        Assert.Equal("TritonBlocked", distributedType.DistributionLayout.Kind);
        Assert.Equal(distributionLayout, distributedType.DistributionLayout);
        Assert.Equal(distributionLayout.LocalShape, distributedType.StorageLayout.LogicalShape);
    }

    [Fact]
    public async Task DirectAffineTilingPassUsesStridedTritonBlockedRegisterLayoutForCudaThreadSplit()
    {
        const int blockSize = 1024;
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
        var placement = new Placement([128], "t");
        var ndsbp = new IRArray<SBP>(new SBP[] { SBP.S(0) });
        var left = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var right = Nncase.IR.F.Affine.Gather(rhs, relation, symbols, shape, None.Default, ndsbp, placement);
        var scatter = Nncase.IR.F.Affine.Scatter(left + right, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(left, out var leftDecision));
        Assert.Equal(new long[] { 8 }, leftDecision.TileShape.ToValueArray());
        Assert.Equal("TritonBlocked", leftDecision.DistributionLayout!.Kind);
        Assert.Contains("threadElementOrder=Strided", leftDecision.DistributionLayout.Attributes!.Value.ToArray());
        Assert.Contains("g0=cta*1024+warp*32+lane+elem*128", leftDecision.DistributionLayout.OwnerLocalToGlobal.Outputs.Select(x => x.ToString()).ToArray());
        Assert.Equal(32, leftDecision.ByteSize);
    }

    [Fact]
    public void DistributionLayoutVerifierRequiresInverseMap()
    {
        var invalid = new DistributionLayout(
            "Invalid",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                ["g0"],
                [new IndexMapBinding("l0", IndexExpr.Var("g0"))],
                ["0<=g0<8"],
                ["0<=l0<8"]),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["l0"],
                [new IndexMapBinding("g0", IndexExpr.Var("l0"))],
                ["0<=l0<8"],
                ["0<=g0<8"],
                Inverse: "GlobalToOwnerLocal"),
            new RankedShape(8));

        Assert.Throws<InvalidOperationException>(() => LayoutVerifier.Verify(invalid));
    }

    [Fact]
    public void StorageLayoutVerifierRejectsDomainMismatchWithoutViewMap()
    {
        var distributedType = new DistributedType(new TensorType(DataTypes.Float32, new RankedShape(1024)), [SBP.S(0)], new Placement([128], "t"));
        var invalidStorage = StorageLayout.Identity(new RankedShape(16));

        Assert.Throws<InvalidOperationException>(() => LayoutVerifier.Verify(distributedType.DistributionLayout, invalidStorage));
    }

    [Fact]
    public void StorageLayoutVerifierRejectsUnsupportedNamedPrimitive()
    {
        var distributedType = new DistributedType(new TensorType(DataTypes.Float32, new RankedShape(1024)), [SBP.S(0)], new Placement([128], "t"));
        var invalidStorage = new StorageLayout(
            "Swizzle",
            distributedType.DistributionLayout.LocalShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                ["l0"],
                [new IndexMapBinding("p0", new IndexNamedPrimitive("xor_swizzle", [IndexExpr.Var("l0")]))],
                ["0<=l0<8"],
                ["0<=p0<8"],
                Inverse: "LogicalToPhysical"));

        Assert.Throws<NotSupportedException>(() => LayoutVerifier.Verify(distributedType.DistributionLayout, invalidStorage));
    }

    [Fact]
    public void StorageLayoutVerifierRejectsUnsupportedViewMapPrimitive()
    {
        var tensorShape = new RankedShape(1024);
        var distributedType = new DistributedType(new TensorType(DataTypes.Float32, tensorShape), [SBP.S(0)], new Placement([128], "t"));
        var invalidViewMap = new IndexMapDescriptor(
            "OwnerLocalToGlobal",
            ["owner0", "l0"],
            [new IndexMapBinding("g0", new IndexNamedPrimitive("xor_swizzle", [IndexExpr.Var("owner0"), IndexExpr.Var("l0")]))],
            ["0<=owner0<128", "0<=l0<8"],
            ["0<=g0<1024"],
            Inverse: "GlobalToOwnerLocal");
        var invalidStorage = StorageLayout.SharedBlock(tensorShape, distributedType.DistributionLayout) with { ViewMap = invalidViewMap };

        Assert.Throws<NotSupportedException>(() => LayoutVerifier.Verify(distributedType.DistributionLayout, invalidStorage));
    }

    [Fact]
    public void StorageLayoutVerifierRequiresViewMapInverse()
    {
        var tensorShape = new RankedShape(1024);
        var distributedType = new DistributedType(new TensorType(DataTypes.Float32, tensorShape), [SBP.S(0)], new Placement([128], "t"));
        var invalidViewMap = distributedType.DistributionLayout.OwnerLocalToGlobal with { Inverse = "WrongInverse" };
        var invalidStorage = StorageLayout.SharedBlock(tensorShape, distributedType.DistributionLayout) with { ViewMap = invalidViewMap };

        Assert.Throws<InvalidOperationException>(() => LayoutVerifier.Verify(distributedType.DistributionLayout, invalidStorage));
    }

    [Fact]
    public void TensorUtilitiesVerifiesExplicitStorageLayoutBeforeSizing()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var placement = new Placement([128], "t");
        var layout = DistributionLayout.FromAxisPolicies(tensorType, [SBP.S(0)], placement);
        var invalidStorage = StorageLayout.Identity(new RankedShape(16));
        var distributedType = new DistributedType(
            tensorType,
            [SBP.S(0)],
            placement,
            ExplicitDistributionLayout: layout,
            ExplicitStorageLayout: invalidStorage);

        var ex = Assert.Throws<InvalidOperationException>(() => TensorUtilities.GetTensorMaxSizeAndStrides(distributedType));
        Assert.Contains(nameof(TensorUtilities.GetTensorMaxSizeAndStrides), ex.Message, StringComparison.Ordinal);
        Assert.Contains("StorageLayout", ex.Message, StringComparison.Ordinal);
        Assert.Contains("DistributionLayout", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TensorUtilitiesRejectsUnrankedExplicitStorageBeforeContiguousSizing()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var placement = new Placement([128], "t");
        var layout = DistributionLayout.FromAxisPolicies(tensorType, [SBP.S(0)], placement);
        var unrankedStorage = new StorageLayout(
            "UnrankedStorage",
            Shape.Unranked,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                ["l0"],
                [new IndexMapBinding("p0", IndexExpr.Var("l0"))],
                ["0<=l0<1"],
                ["0<=p0<1"],
                Inverse: "LogicalToPhysical"));
        var distributedType = new DistributedType(
            tensorType,
            [SBP.S(0)],
            placement,
            ExplicitDistributionLayout: layout,
            ExplicitStorageLayout: unrankedStorage);

        var ex = Assert.Throws<NotSupportedException>(() => TensorUtilities.GetTensorSizeAndContiguousStrides(distributedType));
        Assert.Contains(nameof(TensorUtilities.GetTensorSizeAndContiguousStrides), ex.Message, StringComparison.Ordinal);
        Assert.Contains("UnrankedStorage", ex.Message, StringComparison.Ordinal);
        Assert.Contains("DistributionLayout", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitDistributionLayoutVerifierRejectsOwnerDomainOutsidePlacement()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var placement = new Placement([128], "t");
        var invalidLayout = new DistributionLayout(
            "InvalidOwnerBounds",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                ["g0"],
                [new IndexMapBinding("owner0", IndexExpr.Var("g0")), new IndexMapBinding("l0", IndexExpr.Var("g0"))],
                ["0<=g0<1024"],
                ["0<=owner0<256", "0<=l0<8"],
                Inverse: "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["owner0", "l0"],
                [new IndexMapBinding("g0", IndexExpr.Add(IndexExpr.Mul(IndexExpr.Var("owner0"), IndexExpr.Const(8)), IndexExpr.Var("l0")))],
                ["0<=owner0<256", "0<=l0<8"],
                ["0<=g0<1024"],
                Inverse: "GlobalToOwnerLocal"),
            new RankedShape(8));

        Assert.Throws<InvalidOperationException>(() => DistributedType.FromLayouts(tensorType, placement, invalidLayout));
    }

    [Fact]
    public void ExplicitDistributionOwnerBoundsAllowSymbolicLocalDomains()
    {
        var localExtent = new DimVar("n");
        localExtent.Metadata.Range = new(1, 1024);
        var layout = new DistributionLayout(
            "DynamicLocalOwnerBounds",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                ["g0"],
                [new IndexMapBinding("owner0", IndexExpr.FloorDiv(IndexExpr.Var("g0"), IndexExpr.Var("n"))), new IndexMapBinding("l0", IndexExpr.Mod(IndexExpr.Var("g0"), IndexExpr.Var("n")))],
                ["0<=g0<N"],
                ["0<=owner0<2", "0<=l0<n"],
                Inverse: "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["owner0", "l0"],
                [new IndexMapBinding("g0", IndexExpr.Add(IndexExpr.Mul(IndexExpr.Var("owner0"), IndexExpr.Var("n")), IndexExpr.Var("l0")))],
                ["0<=owner0<2", "0<=l0<n"],
                ["0<=g0<N"],
                Inverse: "GlobalToOwnerLocal"),
            new RankedShape(localExtent));
        var verifyOwnerBounds = typeof(LayoutVerifier).GetMethod("VerifyOwnerBounds", BindingFlags.NonPublic | BindingFlags.Static)!;

        var ex = Record.Exception(() => verifyOwnerBounds.Invoke(null, new object[] { layout, new Placement([2], "t"), "unit test" }));

        Assert.Null(ex);
    }

    [Fact]
    public void ExplicitDistributionLayoutVerifierRejectsWrongInverseFormula()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(16));
        var placement = new Placement([2], "t");
        var invalidLayout = CreateExplicitSplitLayout(
            tensorType.Shape,
            ownerOutput: IndexExpr.FloorDiv(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            localOutput: IndexExpr.Mod(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            globalOutput: IndexExpr.Var("l0"));

        var ex = Assert.Throws<InvalidOperationException>(() => DistributedType.FromLayouts(tensorType, placement, invalidLayout));
        Assert.Contains("inverse composition", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitDistributionLayoutVerifierRejectsReverseOwnerLocalAliasing()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(8));
        var placement = new Placement([2], "t");
        var invalidLayout = CreateExplicitSplitLayout(
            tensorType.Shape,
            ownerOutput: IndexExpr.Const(0),
            localOutput: IndexExpr.Var("g0"),
            globalOutput: IndexExpr.Var("l0"),
            localExtent: 8);

        var ex = Assert.Throws<InvalidOperationException>(() => DistributedType.FromLayouts(tensorType, placement, invalidLayout));
        Assert.Contains("reverse composition", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ExplicitDistributionLayoutVerifierRejectsOwnerExpressionOutsidePlacement()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(256));
        var placement = new Placement([2], "t");
        var invalidLayout = CreateExplicitSplitLayout(
            tensorType.Shape,
            ownerOutput: IndexExpr.Var("g0"),
            localOutput: IndexExpr.Mod(IndexExpr.Var("g0"), IndexExpr.Const(128)),
            globalOutput: IndexExpr.Add(IndexExpr.Mul(IndexExpr.Var("owner0"), IndexExpr.Const(128)), IndexExpr.Var("l0")),
            localExtent: 128);

        var ex = Assert.Throws<InvalidOperationException>(() => DistributedType.FromLayouts(tensorType, placement, invalidLayout));
        Assert.Contains("ExplicitSplit", ex.Message, StringComparison.Ordinal);
        Assert.True(
            ex.Message.Contains("outside declared domain", StringComparison.Ordinal) ||
            ex.Message.Contains("inverse composition", StringComparison.Ordinal),
            ex.Message);
    }

    [Fact]
    public async Task DirectAffineTilingPassRejectsPublicConstructorInvalidExplicitLayout()
    {
        const int blockSize = 16;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(blockSize));
        var invalidLayout = CreateExplicitSplitLayout(
            tensorType.Shape,
            ownerOutput: IndexExpr.FloorDiv(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            localOutput: IndexExpr.Mod(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            globalOutput: IndexExpr.Var("l0"));
        var invalidType = new DistributedType(
            tensorType,
            [SBP.S(0)],
            new Placement([2], "t"),
            ExplicitDistributionLayout: invalidLayout,
            ExplicitStorageLayout: StorageLayout.Identity(invalidLayout.LocalShape));
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, tensorType.Shape, None.Default);
        tile.CheckedType = invalidType;
        var scatter = Nncase.IR.F.Affine.Scatter(tile, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, source, dest));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("direct affine tiling", ex.Message, StringComparison.Ordinal);
        Assert.Contains("inverse composition", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NttShardingCodegenRejectsExplicitLayoutWithIncompatibleAxisPolicies()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(16));
        var placement = new Placement([2], "t");
        var explicitSplit = CreateExplicitSplitLayout(
            tensorType.Shape,
            ownerOutput: IndexExpr.FloorDiv(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            localOutput: IndexExpr.Mod(IndexExpr.Var("g0"), IndexExpr.Const(8)),
            globalOutput: IndexExpr.Add(IndexExpr.Mul(IndexExpr.Var("owner0"), IndexExpr.Const(8)), IndexExpr.Var("l0")));
        var distributedType = new DistributedType(
            tensorType,
            [SBP.B],
            placement,
            ExplicitDistributionLayout: explicitSplit,
            ExplicitStorageLayout: StorageLayout.Identity(explicitSplit.LocalShape));

        var ex = Assert.Throws<NotSupportedException>(() => KernelUtility.ShardingToC(distributedType));
        Assert.Contains("NTT C++ sharding codegen", ex.Message, StringComparison.Ordinal);
        Assert.Contains("ExplicitSplit", ex.Message, StringComparison.Ordinal);
        Assert.Contains("AxisPolicies=(B)", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NttShardingCodegenRejectsExplicitTritonBlockedLayout()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(1024));
        var layout = DistributionLayout.TritonBlocked(
            tensorType.Shape,
            new TritonBlockedLayout(
                SizePerThread: 2,
                ThreadsPerWarp: 32,
                WarpsPerCTA: 4,
                Order: [0],
                CTAsPerCGA: [1],
                CTASplitNum: [1],
                CTAOrder: [0],
                ThreadElementOrder: TritonThreadElementOrder.Strided));
        var distributedType = DistributedType.FromLayouts(
            tensorType,
            new Placement([1, 4, 32], "cwl"),
            layout);

        var ex = Assert.Throws<NotSupportedException>(() => KernelUtility.ShardingToC(distributedType));
        Assert.Contains("TritonBlocked", ex.Message, StringComparison.Ordinal);
        Assert.Contains("legacy AxisPolicies", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineTilingPassAnnotatesGatherBinaryScatterDecisions()
    {
        const int blockSize = 256;
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
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(left, out var leftDecision));
        Assert.True(TileDecisionMetadata.TryGet(right, out var rightDecision));
        Assert.True(TileDecisionMetadata.TryGet(sum, out var sumDecision));
        Assert.True(TileDecisionMetadata.TryGet(scatter, out var scatterDecision));
        Assert.Equal(PhysicalMemorySpace.Register, leftDecision.Storage.PhysicalLocation);
        Assert.Equal(BufferScope.ThreadLocal, sumDecision.Storage.Scope);
        Assert.Equal(blockSize * DataTypes.Float32.SizeInBytes, scatterDecision.ByteSize);
        Assert.Equal(4096, scatterDecision.Capacity.BudgetBytes);
        Assert.Equal("target-options:RegisterTileBudgetBytes", scatterDecision.Capacity.Source);
        Assert.Equal(2, leftDecision.Lifetime.End);
        Assert.Equal(3, sumDecision.Lifetime.End);
        Assert.Contains("Affine.Gather", leftDecision.ToDumpString(), StringComparison.Ordinal);
        Assert.Equal(leftDecision.Storage, rightDecision.Storage);
    }

    [Fact]
    public async Task DirectAffineTilingPassIgnoresCudaElementwiseWithoutTiledInput()
    {
        var tensorType = new TensorType(DataTypes.Float32, new RankedShape(16));
        var lhs = new Var("lhs", tensorType);
        var rhs = new Var("rhs", tensorType);
        var sum = lhs + rhs;
        var function = new Function("main", CUDATarget.Kind, new IRBlock(sum, lhs, rhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.False(TileDecisionMetadata.TryGet(sum, out _));
    }

    [Fact]
    public async Task DirectAffineTilingPassUsesBlockLocalSmemForSharedGather()
    {
        const int blockSize = 128;
        var source = new Var("source", TensorType.Pointer(DataTypes.Float32));
        var firstDest = new Var("first_dest", TensorType.Pointer(DataTypes.Float32));
        var secondDest = new Var("second_dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var tile = Nncase.IR.F.Affine.Gather(source, relation, symbols, shape, None.Default);
        var firstScatter = Nncase.IR.F.Affine.Scatter(tile, firstDest, relation, symbols);
        var secondScatter = Nncase.IR.F.Affine.Scatter(tile, secondDest, relation, symbols);
        var body = new Sequential(new Expr[] { tile, firstScatter, secondScatter });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, source, firstDest, secondDest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal(BufferUsage.Temp, decision.Storage.Usage);
        Assert.Equal(BufferScope.BlockLocal, decision.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, decision.Storage.PhysicalLocation);
        Assert.True(decision.RequiresSynchronization);
        Assert.Equal(blockSize * DataTypes.Float32.SizeInBytes, decision.ByteSize);
        Assert.Equal(49152, decision.Capacity.BudgetBytes);
        Assert.Equal("target-options:SharedMemoryTileBudgetBytes", decision.Capacity.Source);
        Assert.Equal(2, decision.Lifetime.End);
        Assert.True(TileDecisionMetadata.TryGet(firstScatter, out var firstScatterDecision));
        Assert.True(TileDecisionMetadata.TryGet(secondScatter, out var secondScatterDecision));
        Assert.Equal(PhysicalMemorySpace.GMem, firstScatterDecision.Storage.PhysicalLocation);
        Assert.Equal(BufferUsage.Output, secondScatterDecision.Storage.Usage);
    }

    [Fact]
    public async Task DirectAffineTilingPassReusesSmemSlotForNonOverlappingSharedTiles()
    {
        const int blockSize = 128;
        var firstSource = new Var("first_source", TensorType.Pointer(DataTypes.Float32));
        var secondSource = new Var("second_source", TensorType.Pointer(DataTypes.Float32));
        var firstDest0 = new Var("first_dest_0", TensorType.Pointer(DataTypes.Float32));
        var firstDest1 = new Var("first_dest_1", TensorType.Pointer(DataTypes.Float32));
        var secondDest0 = new Var("second_dest_0", TensorType.Pointer(DataTypes.Float32));
        var secondDest1 = new Var("second_dest_1", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols, shape) = CreateRankOneIdentity(blockSize);
        var firstTile = Nncase.IR.F.Affine.Gather(firstSource, relation, symbols, shape, None.Default);
        var firstScatter0 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest0, relation, symbols);
        var firstScatter1 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest1, relation, symbols);
        var secondTile = Nncase.IR.F.Affine.Gather(secondSource, relation, symbols, shape, None.Default);
        var secondScatter0 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest0, relation, symbols);
        var secondScatter1 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest1, relation, symbols);
        var body = new Sequential(new Expr[] { firstTile, firstScatter0, firstScatter1, secondTile, secondScatter0, secondScatter1 });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, firstSource, secondSource, firstDest0, firstDest1, secondDest0, secondDest1));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(firstTile, out var firstDecision));
        Assert.True(TileDecisionMetadata.TryGet(secondTile, out var secondDecision));
        Assert.Equal(new TileLifetime(0, 2), firstDecision.Lifetime);
        Assert.Equal(new TileLifetime(3, 5), secondDecision.Lifetime);
        Assert.Equal("smem-slot0@0+512[0,2]", firstDecision.Telemetry.AllocationSlot);
        Assert.Equal("smem-slot0@0+512[3,5]", secondDecision.Telemetry.AllocationSlot);
    }

    [Fact]
    public async Task DirectAffineTilingPassRejectsOverlappingSmemLiveBytesOverBudget()
    {
        ((NTTTargetOptions)CompileOptions.TargetOptions).SharedMemoryTileBudgetBytes = 768;
        const int blockSize = 128;
        var firstSource = new Var("first_source", TensorType.Pointer(DataTypes.Float32));
        var secondSource = new Var("second_source", TensorType.Pointer(DataTypes.Float32));
        var firstDest0 = new Var("first_dest_0", TensorType.Pointer(DataTypes.Float32));
        var firstDest1 = new Var("first_dest_1", TensorType.Pointer(DataTypes.Float32));
        var secondDest0 = new Var("second_dest_0", TensorType.Pointer(DataTypes.Float32));
        var secondDest1 = new Var("second_dest_1", TensorType.Pointer(DataTypes.Float32));
        var (relation, symbols, shape) = CreateRankOneIdentity(blockSize);
        var firstTile = Nncase.IR.F.Affine.Gather(firstSource, relation, symbols, shape, None.Default);
        var secondTile = Nncase.IR.F.Affine.Gather(secondSource, relation, symbols, shape, None.Default);
        var firstScatter0 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest0, relation, symbols);
        var secondScatter0 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest0, relation, symbols);
        var firstScatter1 = Nncase.IR.F.Affine.Scatter(firstTile, firstDest1, relation, symbols);
        var secondScatter1 = Nncase.IR.F.Affine.Scatter(secondTile, secondDest1, relation, symbols);
        var body = new Sequential(new Expr[] { firstTile, secondTile, firstScatter0, secondScatter0, firstScatter1, secondScatter1 });
        var function = new Function("main", CUDATarget.Kind, new IRBlock(body, firstSource, secondSource, firstDest0, firstDest1, secondDest0, secondDest1));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("aggregate live", ex.Message, StringComparison.Ordinal);
        Assert.Contains("BlockLocal/SMem", ex.Message, StringComparison.Ordinal);
        Assert.Contains("1024 bytes", ex.Message, StringComparison.Ordinal);
        Assert.Contains("budget 768 bytes", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineTilingPassRejectsUnsupportedCudaDirectAffineFallback()
    {
        const int blockSize = 128;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var tile = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var reshaped = IR.F.Tensors.Reshape(tile, new[] { (Dimension)blockSize });
        var scatter = Nncase.IR.F.Affine.Scatter(reshaped, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("Unsupported direct-affine DAGs must fail fast", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineTilingPassRejectsDynamicTileShape()
    {
        var extent = new DimVar("n");
        extent.Metadata.Range = new(1, 1024);
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, 1023);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(extent);
        var tile = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var scatter = Nncase.IR.F.Affine.Scatter(tile, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("fixed tile extent", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineTilingPassNormalizesBroadcastPolicyForCudaThreadTile()
    {
        const int blockSize = 1024;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var dest = new Var("dest", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var placement = new Placement([128], "t");
        var tile = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default, new IRArray<SBP>(new SBP[] { SBP.B }), placement);
        var scatter = Nncase.IR.F.Affine.Scatter(tile, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        _ = await new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal("TritonBlocked", decision.DistributionLayout!.Kind);
        Assert.Equal(new long[] { 8 }, decision.TileShape.ToValueArray());
    }

    [Fact]
    public async Task DirectAffineTilingPassRejectsRegisterTileOverBudget()
    {
        const int blockSize = 2048;
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
        var scatter = Nncase.IR.F.Affine.Scatter(left + right, dest, relation, symbols);
        var function = new Function("main", CUDATarget.Kind, new IRBlock(scatter, lhs, rhs, dest));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("exceeding", ex.Message, StringComparison.Ordinal);
        Assert.Contains("budget", ex.Message, StringComparison.Ordinal);
        Assert.Contains("Register", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task DirectAffineTilingPassHandlesStaleCpuModuleKindInCudaPipeline()
    {
        const int blockSize = 128;
        var lhs = new Var("lhs", TensorType.Pointer(DataTypes.Float32));
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        var symbols = new RankedShape(Array.Empty<Dimension>());
        var shape = new RankedShape(blockSize);
        var tile = Nncase.IR.F.Affine.Gather(lhs, relation, symbols, shape, None.Default);
        var function = new Function("main", CPUTarget.Kind, new IRBlock(tile, lhs));
        Assert.True(CompilerServices.InferenceType(function), CompilerServices.Print(function));

        var ex = await Assert.ThrowsAsync<NotSupportedException>(
            () => new DirectAffineTilingPass(CUDATarget.Kind, CompileOptions).RunAsync(function, new()));
        Assert.Contains("Unsupported direct-affine DAGs must fail fast", ex.Message, StringComparison.Ordinal);
    }

    private static DistributionLayout CreateExplicitSplitLayout(
        Shape tensorShape,
        IndexExpr ownerOutput,
        IndexExpr localOutput,
        IndexExpr globalOutput,
        int localExtent = 8)
    {
        var globalExtent = tensorShape[0].FixedValue;
        return new DistributionLayout(
            "ExplicitSplit",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                ["g0"],
                [new IndexMapBinding("owner0", ownerOutput), new IndexMapBinding("l0", localOutput)],
                [$"0<=g0<{globalExtent}"],
                ["0<=owner0 && owner0<2", $"0<=l0<{localExtent}"],
                Inverse: "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["owner0", "l0"],
                [new IndexMapBinding("g0", globalOutput)],
                ["0<=owner0 && owner0<2", $"0<=l0<{localExtent}"],
                [$"0<=g0<{globalExtent}"],
                Inverse: "GlobalToOwnerLocal"),
            new RankedShape(localExtent));
    }

    private static (AffineRelation Relation, RankedShape Symbols, RankedShape Shape) CreateRankOneIdentity(int blockSize)
    {
        var lane = Nncase.IR.F.Affine.Dim(0);
        lane.Metadata.Range = new(0, blockSize - 1);
        var relation = new AffineRelation(
            new[] { lane },
            Array.Empty<AffineSymbol>(),
            new AffineExpr[] { lane });
        return (relation, new RankedShape(Array.Empty<Dimension>()), new RankedShape(blockSize));
    }
}
