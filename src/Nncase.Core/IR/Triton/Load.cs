// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Triton;

public enum CacheModifier : uint
{
    /// <summary>
    /// No cache modifier.
    /// </summary>
    None = 1,
    CA = 2,
    CG = 3,
    WB = 4,
    CS = 5,
    WT = 6,
    CV = 7,
}

public enum EvictionPolicy : uint
{
    Normal = 1,
    EvictFirst = 2,
    EvictLast = 3,
}

/// <summary>
/// Load expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Load : Op
{
    /// <summary>
    /// Get the pointer parameter.
    /// </summary>
    public static readonly ParameterInfo Ptr = new(typeof(Load), 0, "ptr", IsPointer());

    /// <summary>
    /// Get the mask.
    /// </summary>
    public static readonly ParameterInfo Mask = new(typeof(Load), 1, "mask", IsBool() | IsMaskVector());

    /// <summary>
    /// Get the other.
    /// </summary>
    public static readonly ParameterInfo Other = new(typeof(Load), 2, "other", IsTensor());

    public CacheModifier CacheModifier { get; }

    public EvictionPolicy EvictionPolicy { get; }

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
