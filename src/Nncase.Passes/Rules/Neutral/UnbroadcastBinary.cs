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
/// Unbroadcast <see cref="IR.Math.Binary"/>.
/// </summary>
[RuleGenerator]
public sealed partial class UnbroadcastBinaryLhs : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary(
        "binary",
        "binaryCall",
        x => true,
        IsBroadcast("broadcast", "broadcastCall", IsWildcard("lhsInner")),
        IsWildcard("rhs"));

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr lhsInner, Expr rhs)
    {
        if (rhs.CheckedShape == binaryCall.CheckedShape)
        {
            return binaryCall.WithArguments([(Binary.Lhs, lhsInner)]);
        }

        return null;
    }
}

[RuleGenerator]
public sealed partial class UnbroadcastBinaryRhs : IRewriteRule
{
    /// <inheritdoc/>
    public IPattern Pattern { get; } = IsBinary(
        "binary",
        "binaryCall",
        x => true,
        IsWildcard("lhs"),
        IsBroadcast("broadcast", "broadcastCall", IsWildcard("rhsInner")));

    private Expr? GetReplace(Binary binary, Call binaryCall, Expr lhs, Expr rhsInner)
    {
        if (lhs.CheckedShape == binaryCall.CheckedShape)
        {
            return binaryCall.WithArguments([(Binary.Rhs, rhsInner)]);
        }

        return null;
    }
}
