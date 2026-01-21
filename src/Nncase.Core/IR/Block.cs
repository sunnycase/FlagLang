// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.ComponentModel;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;

namespace Nncase.IR;

public sealed class IRBlock : Callable
{
    private static int _globalBlockIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlock"/> class.
    /// build block.
    /// </summary>
    public IRBlock(string name, BaseExpr body, ReadOnlySpan<IVar> parameters)
        : base(name, ArrayUtility.Concat(body, SpanUtility.UnsafeCast<IVar, BaseExpr>(parameters)))
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlock"/> class.
    /// build block.
    /// </summary>
    public IRBlock(BaseExpr body, ReadOnlySpan<IVar> parameters)
        : this($"block_{_globalBlockIndex++}", body, parameters)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlock"/> class.
    /// build function.
    /// </summary>
    public IRBlock(string name, BaseExpr body, params IVar[] parameters)
        : this(name, body, parameters.AsReadOnlySpan())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRBlock"/> class.
    /// build function.
    /// </summary>
    public IRBlock(BaseExpr body, params IVar[] parameters)
        : this(body, parameters.AsReadOnlySpan())
    {
    }

    public BaseExpr Body => Operands[0];

    public ReadOnlySpan<IVar> Parameters => SpanUtility.UnsafeCast<BaseExpr, IVar>(Operands[1..]);

    public Dictionary<IVar, Dimension[]>? VarMap { get; }

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public IEnumerable<IRType> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((Expr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIRBlock(this, context);

    public IRBlock With(string? name = null)
    {
        return new IRBlock(name ?? Name, Body, Parameters);
    }

    public IRBlock With(string? name = null, BaseExpr? body = null, IVar[]? parameters = null)
        => new IRBlock(name ?? Name, body ?? Body, parameters ?? Parameters);
}
