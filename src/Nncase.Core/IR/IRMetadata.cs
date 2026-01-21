// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase.Diagnostics;

namespace Nncase.IR;

/// <summary>
/// Expression's metadata.
/// </summary>
public class IRMetadata
{
    /// <summary>
    /// Gets or sets outputs names.
    /// </summary>
    public IReadOnlyList<string>? OutputNames { get; set; }

    public ValueRange<double>? Range { get; set; }

    public Location? Location { get; set; }

    public Dictionary<string, object> Attributes { get; } = new();
}
