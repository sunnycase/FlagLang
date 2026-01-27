// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.IR.Logics;

public sealed class LogicalConst : LogicalExpr, IEquatable<LogicalConst?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalConst"/> class.
    /// </summary>
    /// <param name="value">Value.</param>
    public LogicalConst(bool value)
        : base(Array.Empty<Expr>())
    {
        Value = value;
        Metadata.Range = new ValueRange<double>(value ? 1 : 0, value ? 1 : 0);
    }

    /// <summary>
    /// Gets a value indicating whether the logical constant is true or false.
    /// </summary>
    public bool Value { get; }

    public static implicit operator LogicalConst(bool value) => new LogicalConst(value);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitLogicalConst(this, context);

    public LogicalConst With(bool? value = null) => new LogicalConst(value ?? Value);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as LogicalConst);

    /// <inheritdoc/>
    public bool Equals(LogicalConst? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Value == other.Value;
    }

    public override string ToString() => Value.ToString();

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value);
}
