// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.TIR;

/// <summary>
/// if(xxx) then { zzz } else { yyy }.
/// </summary>
public sealed class IfThenElse : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="IfThenElse"/> class.
    /// </summary>
    public IfThenElse(BaseExpr condition, Sequential then, Sequential @else)
        : base([condition, then, @else])
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IfThenElse"/> class.
    /// </summary>
    public IfThenElse(BaseExpr condition, Sequential then)
        : this(condition, then, new())
    {
    }

    public BaseExpr Condition => (BaseExpr)Operands[0];

    public Sequential Then => (Sequential)Operands[1];

    public Sequential Else => (Sequential)Operands[2];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIfThenElse(this, context);

    public IfThenElse With(BaseExpr? condition = null, Sequential? then = null, Sequential? @else = null)
        => new IfThenElse(condition ?? Condition, then ?? Then, @else ?? Else);
}
