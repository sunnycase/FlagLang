// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.PatternMatch;

namespace Nncase.IR.Tensors;

/// <summary>
/// Anchors SSA dependencies while preserving the value type.
/// </summary>
[PatternFunctionalGenerator]
public sealed partial class Depend : Op
{
    /// <summary>
    /// Gets the dependency expression that must stay reachable.
    /// </summary>
    public static readonly ParameterInfo Dependencies = new(typeof(Depend), 0, "dependencies", ParameterKind.Input);

    /// <summary>
    /// Gets the value expression.
    /// </summary>
    public static readonly ParameterInfo Value = new(typeof(Depend), 1, "value", ParameterKind.Input);

    /// <inheritdoc/>
    public override bool CanFoldConstCall => false;
}
