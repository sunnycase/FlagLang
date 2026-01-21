// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Triton;

namespace Nncase.IR.F;

/// <summary>
/// Random functional helper.
/// </summary>
public static class Triton
{
    public static Call Load(Expr ptr, Expr? mask = null, Expr? other = null, CacheModifier cacheModifier = CacheModifier.None, EvictionPolicy evictionPolicy = EvictionPolicy.Normal) =>
        new Call(new Load(cacheModifier, evictionPolicy), ptr, mask ?? None.Default, other ?? None.Default);

    public static Call Store(Expr ptr, Expr value, Expr? mask = null, CacheModifier cacheModifier = CacheModifier.None, EvictionPolicy evictionPolicy = EvictionPolicy.Normal) =>
        new Call(new Store(cacheModifier, evictionPolicy), ptr, value, mask ?? None.Default);
}
