// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Affine;

[PatternFunctionalGenerator]
public sealed partial class Gather : Op
{
    public static readonly ParameterInfo Source = new(typeof(Gather), 0, "source", IsPointer(), ParameterKind.Input);

    public static readonly ParameterInfo DefaultValue = new(typeof(Gather), 1, "defaultValue");

    public AffineRelation Relation { get; }

    public RankedShape Symbols { get; }

    public Shape Shape { get; }

    [Browsable(false)]
    public IRArray<SBP> NdSBP { get; }

    [Browsable(false)]
    public Placement Placement { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    public override string DisplayProperty()
    {
        var distribution = Placement.Rank == 0 ? string.Empty : $", Dist: ({string.Join(',', NdSBP)}), {Placement}";
        return $"{Relation}, Symbols: {Symbols}, Shape: {Shape}{distribution}";
    }
}
