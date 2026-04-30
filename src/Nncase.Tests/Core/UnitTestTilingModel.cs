// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using System.Threading.Tasks;
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
    [Fact]
    public void LegacyMemoryLocationMapsToOrthogonalStorage()
    {
        var data = BufferStorage.FromLegacy(MemoryLocation.Data);
        Assert.Equal(BufferUsage.Temp, data.Usage);
        Assert.Equal(BufferScope.ThreadLocal, data.Scope);
        Assert.Equal(PhysicalMemorySpace.LocalAddressable, data.PhysicalLocation);

        var smem = BufferStorage.FromLegacy(MemoryLocation.BlockLocalData);
        Assert.Equal(BufferUsage.Temp, smem.Usage);
        Assert.Equal(BufferScope.BlockLocal, smem.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, smem.PhysicalLocation);

        var buffer = new PhysicalBuffer(8, (Dimension)64, smem);
        Assert.Equal(MemoryLocation.BlockLocalData, buffer.Location);
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

        _ = await new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new());

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

        _ = await new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(left, out var leftDecision));
        Assert.True(TileDecisionMetadata.TryGet(right, out var rightDecision));
        Assert.True(TileDecisionMetadata.TryGet(sum, out var sumDecision));
        Assert.True(TileDecisionMetadata.TryGet(scatter, out var scatterDecision));
        Assert.Equal(PhysicalMemorySpace.Register, leftDecision.Storage.PhysicalLocation);
        Assert.Equal(BufferScope.ThreadLocal, sumDecision.Storage.Scope);
        Assert.Equal(blockSize * DataTypes.Float32.SizeInBytes, scatterDecision.ByteSize);
        Assert.Equal(4096, scatterDecision.Capacity.BudgetBytes);
        Assert.Equal("cuda-default-register-tile-budget", scatterDecision.Capacity.Source);
        Assert.Equal(2, leftDecision.Lifetime.End);
        Assert.Equal(3, sumDecision.Lifetime.End);
        Assert.Contains("Affine.Gather", leftDecision.ToDumpString(), StringComparison.Ordinal);
        Assert.Equal(leftDecision.Storage, rightDecision.Storage);
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

        _ = await new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal(BufferUsage.Temp, decision.Storage.Usage);
        Assert.Equal(BufferScope.BlockLocal, decision.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.SMem, decision.Storage.PhysicalLocation);
        Assert.True(decision.RequiresSynchronization);
        Assert.Equal(blockSize * DataTypes.Float32.SizeInBytes, decision.ByteSize);
        Assert.Equal(49152, decision.Capacity.BudgetBytes);
        Assert.Equal("cuda-default-smem-tile-budget", decision.Capacity.Source);
        Assert.Equal(2, decision.Lifetime.End);
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
            () => new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new()));
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

        _ = await new DirectAffineTilingPass(CUDATarget.Kind).RunAsync(function, new());

        Assert.True(TileDecisionMetadata.TryGet(tile, out var decision));
        Assert.Equal(BufferScope.ThreadLocal, decision.Storage.Scope);
        Assert.Equal(PhysicalMemorySpace.LocalAddressable, decision.Storage.PhysicalLocation);
    }
}
