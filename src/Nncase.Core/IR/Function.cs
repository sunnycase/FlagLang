// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Function expression.
/// </summary>
public sealed class Function : BaseFunction
{
    private static int _globalFuncIndex;

    /// <summary>
    /// used for save expr in VarMap.
    /// </summary>
    private readonly ExprPinner? _pinner;

    public Function(string name, string moduleKind, IRBlock body, Dictionary<IVar, Dimension[]>? varMap = null)
        : base(name, moduleKind, new BaseExpr[] { body })
    {
        VarMap = varMap ?? new();
        var dynamicDims = VarMap.Values.SelectMany(x => x).ToArray();
        _pinner = new ExprPinner(dynamicDims);
    }

    public Function(string name, IRBlock body, Dictionary<IVar, Dimension[]>? varMap = null)
        : this(name, CPUModuleKind, body, varMap)
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="Function"/> class.
    /// build function.
    /// </summary>
    public Function(IRBlock body)
        : this($"func_{_globalFuncIndex++}", body)
    {
    }

    public IRBlock Body => (IRBlock)Operands[0];

    public ReadOnlySpan<IVar> Parameters => Body.Parameters;

    public Dictionary<IVar, Dimension[]>? VarMap { get; }

    /// <summary>
    /// Gets get all parameter checked types.
    /// </summary>
    public override IEnumerable<IRType> ParameterTypes => Parameters.AsValueEnumerable().Select(x => ((Expr)x).CheckedType).ToArray();

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitFunction(this, context);

    public override BaseFunction With(string? name = null, string? moduleKind = null)
    {
        return new Function(name ?? Name, moduleKind ?? ModuleKind, Body, VarMap);
    }

    public Function With(string? name = null, string? moduleKind = null, IRBlock? body = null)
        => new Function(name ?? Name, moduleKind ?? ModuleKind, body ?? Body, VarMap);
}
