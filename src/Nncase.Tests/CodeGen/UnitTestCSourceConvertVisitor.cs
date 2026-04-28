// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CodeGen.NTT;
using Nncase.IR;
using Nncase.IR.Distributed;
using Xunit;

namespace Nncase.Tests.CodeGen;

public sealed class UnitTestCSourceConvertVisitor
{
    [Fact]
    public void ProgramIdDimZeroMapsToBlockProgramId()
    {
        var visitor = new DimensionVisitor();

        var programId = visitor.Render(new ProgramIdDim(0));
        var threadId = visitor.Render(ThreadIdDim.Default);

        Assert.Contains("topology::block", programId, StringComparison.Ordinal);
        Assert.DoesNotContain("topology::thread", programId, StringComparison.Ordinal);
        Assert.Contains("topology::thread", threadId, StringComparison.Ordinal);
    }

    [Fact]
    public void ProgramIdDimRejectsUnsupportedAxis()
    {
        var visitor = new DimensionVisitor();

        Assert.Throws<NotSupportedException>(() => visitor.Render(new ProgramIdDim(1)));
    }

    private sealed class DimensionVisitor : CSourceConvertVisitor
    {
        public string Render(BaseExpr expr) => Visit(expr).Name;
    }
}
