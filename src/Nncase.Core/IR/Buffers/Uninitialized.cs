// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.ComponentModel;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// Gets input.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Uninitialized : Op
{
    /// <summary>
    /// the shape.
    /// </summary>
    public static readonly ParameterInfo Shape = new(typeof(Uninitialized), 0, "shape", IsShapeType());

    public DataType DType { get; }

    public TIR.BufferStorage Storage { get; }

    [Browsable(false)]
    public IRArray<SBP> NdSBP { get; }

    [Browsable(false)]
    public Placement Placement { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;

    /// <inheritdoc/>
    public override string DisplayProperty() => $"{DType.GetCSharpName()}, Storage: {Storage}, {NdSBP}, {Placement}";

    public TIR.BufferStorage GetStorage() => Storage;
}
