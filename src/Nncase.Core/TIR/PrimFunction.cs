// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.TIR;

/// <summary>
/// PrimFunction expression.
/// </summary>
public sealed class PrimFunction : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="body">Body.</param>
    public PrimFunction(string name, string moduleKind, Sequential body)
        : base(name, moduleKind, new BaseExpr[] { body })
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="PrimFunction"/> class.
    /// </summary>
    /// <param name="moduleKind">module kind.</param>
    /// <param name="body">Body.</param>
    public PrimFunction(string moduleKind, Sequential body)
        : this($"primfunc_{_globalFuncIndex++}", moduleKind, body)
    {
    }

    /// <summary>
    /// Gets body.
    /// </summary>
    public Sequential Body => (Sequential)Operands[0];

    public ReadOnlySpan<IVar> Parameters => Body.Parameters;

    public override IEnumerable<IRType> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((Expr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitPrimFunction(this, context);

    public override BaseFunction With(string? name = null, string? moduleKind = null)
    {
        return new PrimFunction(name ?? Name, moduleKind ?? ModuleKind, Body);
    }

    public PrimFunction With(string? name = null, string? moduleKind = null, Sequential? body = null, Schedule.SchedFunctionResult? sched = null)
        => new PrimFunction(name ?? Name, moduleKind ?? ModuleKind, body ?? Body)
        {
            // note maybe add SchedResult into ctor.
            SchedResult = sched ?? SchedResult,
        };
}
