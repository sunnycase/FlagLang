// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Module.
/// </summary>
public sealed class IRModule : BaseExpr
{
    private BaseFunction? _entry;

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// the default IrModule ctor.
    /// </summary>
    public IRModule()
        : base(Array.Empty<BaseExpr>())
    {
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// </summary>
    /// <param name="main">main func.</param>
    public IRModule(BaseFunction main)
        : base(new BaseExpr[] { main })
    {
        Entry = main;
    }

    /// <summary>
    /// Initializes a new instance of the <see cref="IRModule"/> class.
    /// </summary>
    /// <param name="functions">functions.</param>
    /// <param name="entry">entry function.</param>
    public IRModule(BaseFunction[] functions, BaseFunction? entry = null)
        : base(functions.Cast<BaseExpr>().ToArray())
    {
        Entry = entry;
    }

    /// <summary>
    /// Gets functions.
    /// </summary>
    public ReadOnlySpan<BaseFunction> Functions => SpanUtility.UnsafeCast<BaseExpr, BaseFunction>(Operands);

    /// <summary>
    /// Gets or sets entry function.
    /// </summary>
    public BaseFunction? Entry
    {
        get => _entry;
        set
        {
            if (!SpanUtility.ReferenceContains(Functions, value))
            {
                throw new ArgumentException("Entry function must be in the module functions list.");
            }

            if (_entry is not null)
            {
                _entry.IsEntry = false;
            }

            _entry = value;
            if (value is not null)
            {
                value.IsEntry = true;
            }
        }
    }

    /// <summary>
    /// Add function.
    /// </summary>
    /// <param name="function">Callable to add.</param>
    public void Add(BaseFunction function)
    {
        CompilerServices.InferenceType(function);
        AddOperand(function);
    }

    /// <summary>
    /// Replace the function defination.
    /// </summary>
    /// <param name="index">function index.</param>
    /// <param name="function">the entry function defination.</param>
    public void Replace(int index, BaseFunction function)
    {
        CompilerServices.InferenceType(function);
        var old = Functions[index];
        if (ReferenceEquals(old, function))
        {
            return;
        }

        old.ReplaceAllUsesWith(function);
        var isOldEntry = object.ReferenceEquals(old, _entry);
        ReplaceOperandAt(index, function);
        if (isOldEntry)
        {
            Entry = function;
        }
    }

    /// <summary>
    /// Remove function .
    /// </summary>
    /// <param name="function">function.</param>
    public void Remove(BaseFunction function)
    {
        RemoveOperand(function);
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context)
        => functor.VisitIRModule(this, context);

    public IRModule With(BaseFunction[]? functions = null, BaseFunction? entry = null) => new IRModule(functions ?? Functions.ToArray(), entry ?? Entry);

    protected override void OnOperandsReplaced()
    {
        if (_entry is not null && !SpanUtility.ReferenceContains(Functions, _entry))
        {
            _entry.IsEntry = false;
            _entry = null;
        }

        base.OnOperandsReplaced();
    }
}
