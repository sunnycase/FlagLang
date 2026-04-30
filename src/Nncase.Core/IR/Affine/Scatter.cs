// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Affine;

[PatternFunctionalGenerator]
public sealed partial class Scatter : Op
{
    public static readonly ParameterInfo Source = new(typeof(Scatter), 0, "source", ParameterKind.Input);

    public static readonly ParameterInfo Dest = new(typeof(Scatter), 1, "dest", IsPointer());

    public AffineRelation Relation { get; }

    public RankedShape Symbols { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    public override string DisplayProperty() => $"{Relation}, Symbols: {Symbols}";
}
