// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

/// <summary>
/// Affine scatter expression that writes into a raw pointer using an affine relation.
/// </summary>
public sealed partial class AffineScatter : NTTKernelOp
{
    public static readonly ParameterInfo Source = new(typeof(AffineScatter), 0, "source", ParameterKind.Input);

    public static readonly ParameterInfo Dest = new(typeof(AffineScatter), 1, "dest", IsPointer(), ParameterKind.Input);

    public AffineRelation Relation { get; }

    public RankedShape Symbols { get; }

    public override string DisplayProperty() => $"{Relation}, Symbols: {Symbols}";
}
