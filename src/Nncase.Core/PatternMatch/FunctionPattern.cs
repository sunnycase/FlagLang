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
/// Pattern for <see cref="Function"/>.
/// </summary>
/// <param name="Body">Body pattern.</param>
/// <param name="Name">Name.</param>
public sealed record FunctionPattern(Pattern Body, string? Name) : Pattern<Function>(Name)
{
    /// <summary>
    /// Initializes a new instance of the <see cref="FunctionPattern"/> class.
    /// </summary>
    /// <param name="function"><see cref="Function"/> expression.</param>
    /// <param name="name">name.</param>
    public FunctionPattern(Function function, string? name)
        : this(function.Body, name)
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
    /// <returns>FunctionPattern .</returns>
    public static FunctionPattern IsFunction(string? name, Pattern body) => new FunctionPattern(body, name);

    public static FunctionPattern IsFunction(Pattern body) => IsFunction(null, body);
}
