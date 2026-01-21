// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Collections;
using Nncase.IR;
using Nncase.TIR.Builders;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// The container of Exprs.
/// Represent a sequence of Expr.
/// </summary>
public sealed class Sequential : Expr
{
    public static readonly Sequential Empty = new Sequential(ReadOnlySpan<Expr>.Empty);

    private readonly int _parameterCount;

    public Sequential(ReadOnlySpan<Expr> fields = default, ReadOnlySpan<IVar> parameters = default)
        : base(SpanUtility.Concat(SpanUtility.UnsafeCast<Expr, BaseExpr>(fields), SpanUtility.UnsafeCast<IVar, BaseExpr>(parameters)))
    {
        _parameterCount = parameters.Length;
    }

    public ReadOnlySpan<IVar> Parameters => SpanUtility.UnsafeCast<BaseExpr, IVar>(Operands[^_parameterCount..]);

    public ReadOnlySpan<Expr> Fields => SpanUtility.UnsafeCast<BaseExpr, Expr>(Operands[..^_parameterCount]);

    public int Count => Fields.Length;

    public bool HasTerminator => Fields.Length > 0 && Fields[^1] is TIR.Return;

    /// <summary>
    /// get the fields.
    /// </summary>
    public new Expr this[int index] => Fields[index];

    public static Sequential Flatten(ReadOnlySpan<object> exprOrBuilders, ReadOnlySpan<IVar> parameters = default)
    {
        var ret = new List<Expr>();
        foreach (var item in exprOrBuilders)
        {
            Flatten(ret, item);
        }

        return new Sequential(CollectionsMarshal.AsSpan(ret), parameters);
    }

    public static Sequential Flatten(ReadOnlySpan<Expr> exprs) => Flatten(SpanUtility.UnsafeCast<Expr, object>(exprs));

    public static Sequential Flatten(Expr[] exprs) => Flatten(exprs.AsSpan());

    public static Sequential Flatten(object[] exprs) => Flatten(exprs.AsSpan());

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitSequential(this, context);

    public Sequential With(Expr[]? fields = null, IVar[]? parameters = null) => new Sequential(fields ?? Fields, parameters ?? Parameters);

    public void InsertAt(int index, Expr expr)
    {
        InsertOperandAt(index, expr);
    }

    private static void Flatten(List<Expr> exprs, object exprOrBuilder)
    {
        switch (exprOrBuilder)
        {
            case Sequential sub:
                exprs.AddRange(Flatten(sub.Fields).Fields);
                break;
            case Expr expr:
                if (expr is not Call { Target: Nop })
                {
                    exprs.Add(expr);
                }

                break;
            case IExprBuilder<Expr> builder:
                Flatten(exprs, builder.Build());
                break;
            default:
                throw new ArgumentException("Invalid exprOrBuilder type: " + exprOrBuilder.ToString());
        }
    }
}
