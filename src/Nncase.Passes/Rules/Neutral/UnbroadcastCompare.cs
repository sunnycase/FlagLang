// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.PatternMatch;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.F.Tensors;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Neutral;

/// <summary>
/// Unbroadcast <see cref="IR.Math.Compare"/>.
/// </summary>
[RuleGenerator]
public sealed partial class UnbroadcastCompareLhs : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCompare(
        "compare",
        "compareCall",
        x => true,
        IsBroadcast("broadcast", "broadcastCall", IsWildcard("lhsInner")),
        IsWildcard("rhs"));

    private Expr? GetReplace(Compare compare, Call compareCall, Expr lhsInner, Expr rhs)
    {
        if (rhs.CheckedShape == compareCall.CheckedShape)
        {
            return compareCall.WithArguments([(Compare.Lhs, lhsInner)]);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class UnbroadcastCompareRhs : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsCompare(
        "compare",
        "compareCall",
        x => true,
        IsWildcard("lhs"),
        IsBroadcast("broadcast", "broadcastCall", IsWildcard("rhsInner")));

    private Expr? GetReplace(Compare compare, Call compareCall, Expr lhs, Expr rhsInner)
    {
        if (lhs.CheckedShape == compareCall.CheckedShape)
        {
            return compareCall.WithArguments([(Compare.Rhs, rhsInner)]);
        }

        return null;
    }
}
