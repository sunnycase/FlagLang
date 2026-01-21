// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Affine;

namespace Nncase.Schedule;

public sealed record GridSchedule(AffineMap DomainMap, GridSchedule.Loop[] Loops, GridSchedule.Place[] Places, AffineMap[] BodyBufferViews)
{
    public sealed record Loop(AffineDim Domain, long Stop, long Stride, string Name);

    public sealed record TemporalBuffer(int Buffer, AffineMap Subview, TemporalBuffer? Parent);

    public sealed record Place(TemporalBuffer[] TemporalBuffers);
}
