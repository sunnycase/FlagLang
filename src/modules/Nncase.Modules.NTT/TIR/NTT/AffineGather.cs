// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.IR.Affine;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.TIR.NTT;

/// <summary>
/// Affine gather expression that reads from a raw pointer using an affine relation.
/// </summary>
public sealed partial class AffineGather : NTTKernelOp
{
    public static readonly ParameterInfo Source = new(typeof(AffineGather), 0, "source", IsPointer(), ParameterKind.Input);

    public static readonly ParameterInfo DefaultValue = new(typeof(AffineGather), 1, "defaultValue", ParameterKind.Input);

    public static readonly ParameterInfo Output = new(typeof(AffineGather), 2, "output");

    public AffineRelation Relation { get; }

    public RankedShape Symbols { get; }

    public Shape Shape { get; }

    public override string DisplayProperty() => $"{Relation}, Symbols: {Symbols}, Shape: {Shape}";
}
