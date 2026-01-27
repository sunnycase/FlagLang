// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using Nncase.Utilities;

namespace Nncase.IR.Logics;

public sealed class LogicalAnd : LogicalExpr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalAnd"/> class.
    /// </summary>
    public LogicalAnd(LogicalExpr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<LogicalExpr> Operands => SpanUtility.UnsafeCast<BaseExpr, LogicalExpr>(base.Operands);

    public int Count => Operands.Length;

    public LogicalExpr this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitLogicalAnd(this, context);

    public LogicalAnd With(LogicalExpr[]? operands = null) =>
        new LogicalAnd(operands ?? Operands.ToArray());

    public override string ToString()
    {
        return $"({StringUtility.Join(" && ", Operands)})";
    }
}

public sealed class LogicalOr : LogicalExpr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalOr"/> class.
    /// </summary>
    public LogicalOr(LogicalExpr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<LogicalExpr> Operands => SpanUtility.UnsafeCast<BaseExpr, LogicalExpr>(base.Operands);

    public int Count => Operands.Length;

    public LogicalExpr this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitLogicalOr(this, context);

    public LogicalOr With(LogicalExpr[]? operands = null) =>
        new LogicalOr(operands ?? Operands.ToArray());

    public override string ToString()
    {
        return $"({StringUtility.Join(" || ", Operands)})";
    }
}
