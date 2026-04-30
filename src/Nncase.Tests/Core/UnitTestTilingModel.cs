// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;
using Xunit;

namespace Nncase.Tests.Core;

public sealed class UnitTestTilingModel
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
        Assert.Equal("SBP", layout.Kind);
        Assert.Equal(new long[] { 8 }, layout.LocalShape.ToValueArray());
        Assert.Contains("owner0=floor(g0/8)%128", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("l0=g0%8", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("g0=owner0*8+l0", layout.OwnerLocalToGlobal.Outputs.ToArray());

        var storageLayout = distributedType.StorageLayout;
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
        Assert.Equal("TritonBlocked", layout.Kind);
        Assert.Equal(new long[] { 256 }, layout.LocalShape.ToValueArray());
        Assert.Contains("sizePerThread=2", layout.Attributes!.Value.ToArray());
        Assert.Contains("threadsPerWarp=32", layout.Attributes!.Value.ToArray());
        Assert.Contains("lane=floor((g0%64)/2)", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("elem=g0%2", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("g0=cta*256+warp*64+lane*2+elem", layout.OwnerLocalToGlobal.Outputs.ToArray());
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
        Assert.Contains("threadElementOrder=Strided", layout.Attributes!.Value.ToArray());
        Assert.Contains("lane=(g0%128)%32", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("elem=floor((g0%256)/128)", layout.GlobalToOwnerLocal.Outputs.ToArray());
        Assert.Contains("g0=cta*256+warp*32+lane+elem*128", layout.OwnerLocalToGlobal.Outputs.ToArray());
    }
}
