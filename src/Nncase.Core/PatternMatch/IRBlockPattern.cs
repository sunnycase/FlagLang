// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.PatternMatch;

/// <summary>
/// Pattern for <see cref="IRBlock"/>.
/// </summary>
/// <param name="Body">Body pattern.</param>
/// <param name="Parameters">Parameters pattern.</param>
/// <param name="Name">Name.</param>
public sealed record IRBlockPattern(Pattern Body, VArgsPattern Parameters, string? Name) : Pattern<IRBlock>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlockPattern"/> class.
    /// </summary>
    /// <param name="block"><see cref="IRBlock"/> expression.</param>
    /// <param name="name">name.</param>
    public IRBlockPattern(IRBlock block, string? name)
        : this(block.Body, new VArgsPattern(block.Parameters.AsValueEnumerable().Select(x => (Pattern)x).ToArray(), null), name)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlockPattern"/> class.
    /// </summary>
    /// <param name="body">Body pattern.</param>
    /// <param name="parameters">Parameter patterns.</param>
    /// <param name="name">name.</param>
    public IRBlockPattern(Pattern body, Pattern[] parameters, string? name)
        : this(body, new VArgsPattern(parameters, null), name)
    {
    }
}

public static partial class Utility
{
    /// <summary>
    /// Create the function pattern.
    /// </summary>
    /// <param name="name">name.</param>
    /// <param name="body">body.</param>
    /// <param name="parameters">params.</param>
    /// <returns>IRBlockPattern .</returns>
    public static IRBlockPattern IsIRBlock(string? name, Pattern body, VArgsPattern parameters) => new IRBlockPattern(body, parameters, name);

    public static IRBlockPattern IsIRBlock(Pattern body, VArgsPattern parameters) => IsIRBlock(null, body, parameters);

    public static IRBlockPattern IsIRBlock(string? name, Pattern body, params Pattern[] parameters) => IsIRBlock(name, body, IsVArgs(parameters));

    public static IRBlockPattern IsIRBlock(Pattern body, params Pattern[] parameters) => IsIRBlock(null, body, IsVArgs(parameters));
}
