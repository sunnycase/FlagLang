// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.IR.Logics;

public sealed class DimCompare : LogicalExpr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimCompare"/> class.
    /// </summary>
    public DimCompare(CompareOp op, Dimension lhs, Dimension rhs)
        : base(new BaseExpr[] { lhs, rhs })
    {
        Op = op;
    }

    public Dimension Lhs => (Dimension)Operands[0];

    public Dimension Rhs => (Dimension)Operands[1];

    public CompareOp Op { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimCompare(this, context);

    public DimCompare With(CompareOp? op = null, Dimension? lhs = null, Dimension? rhs = null) =>
        new DimCompare(op ?? Op, lhs ?? Lhs, rhs ?? Rhs);

    public override string ToString() => $"({Lhs} {OpToString(Op)} {Rhs})";

    private static string OpToString(CompareOp op) => op switch
    {
        CompareOp.Equal => "==",
        CompareOp.NotEqual => "!=",
        CompareOp.LowerThan => "<",
        CompareOp.LowerOrEqual => "<=",
        CompareOp.GreaterThan => ">",
        CompareOp.GreaterOrEqual => ">=",
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };
}
