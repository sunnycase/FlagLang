// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR.Buffers;

/// <summary>
/// BufferGather expression.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class BufferGather : Op
{
    /// <summary>
    /// Get the input parameter.
    /// </summary>
    public static readonly ParameterInfo Input = new(typeof(BufferGather), 0, "input", IsPointerScalar());

    /// <summary>
    /// Get the indices.
    /// </summary>
    public static readonly ParameterInfo Indices = new(typeof(BufferGather), 1, "indices", IsTuple());

    public static readonly ParameterInfo Mask = new(typeof(BufferGather), 2, "mask", IsPointerScalar());

    public static readonly ParameterInfo DefaultValue = new(typeof(BufferGather), 3, "defaultValue", IsPointerScalar());

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
