// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Triton;

/// <summary>
/// Store expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Store : Op
{
    /// <summary>
    /// Get the pointer parameter.
    /// </summary>
    public static readonly ParameterInfo Ptr = new(typeof(Store), 0, "ptr", IsPointer());

    /// <summary>
    /// Get the value parameter.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Store), 1, "value", IsTensor());

    /// <summary>
    /// Get the mask.
    /// </summary>
    public static readonly ParameterInfo Mask = new(typeof(Store), 2, "mask", IsBool() | IsMaskVector());

    public CacheModifier CacheModifier { get; }

    public EvictionPolicy EvictionPolicy { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
