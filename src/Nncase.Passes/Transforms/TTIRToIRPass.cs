// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Triton;
using Nncase.Passes.Rules;
using Nncase.TIR;

namespace Nncase.Passes.Transforms;

/// <summary>
/// TTIR to IR pass.
/// </summary>
public sealed class TTIRToIRPass : FunctionPass
{
    /// <summary>
    /// Initializes a new instance of the <see cref="TTIRToIRPass"/> class.
    /// </summary>
    public TTIRToIRPass()
    {
    }

    /// <inheritdoc/>
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction pre, RunPassContext options)
    {
        if (pre is PrimFunction pf)
        {
            var stores = StoreCollector.Collect(pf.Body);
            var returns = ReturnCollector.Collect(pf.Body);
            var body = stores.Count > 0
                ? stores.Count == 1 ? stores[0] : new IR.Tuple(stores.ToArray())
                : BuildReturnBody(returns);
            var newBody = new IRBlock(body, pf.Parameters);
            return Task.FromResult<BaseFunction>(new Function(pf.Name, newBody));
        }

        return Task.FromResult(pre);
    }

    private static BaseExpr BuildReturnBody(IReadOnlyList<Return> returns)
    {
        if (returns.Count == 0)
        {
            return new IR.Tuple();
        }

        if (returns.Count != 1)
        {
            throw new InvalidOperationException($"Expected at most one return in Triton helper function, got {returns.Count}.");
        }

        var values = returns[0].Values.ToArray();
        return values.Length switch
        {
            0 => new IR.Tuple(),
            1 => values[0],
            _ => new IR.Tuple(values),
        };
    }

    private sealed class StoreCollector : ExprWalker<List<BaseExpr>>
    {
        public static IReadOnlyList<BaseExpr> Collect(BaseExpr expr)
        {
            var stores = new List<BaseExpr>();
            new StoreCollector().Visit(expr, stores);
            return stores;
        }

        protected override Unit VisitLeafCall(Call expr, List<BaseExpr> context)
        {
            if (expr.Target is IR.Triton.Store)
            {
                context.Add(expr);
            }

            return base.VisitLeafCall(expr, context);
        }
    }

    private sealed class ReturnCollector : ExprWalker<List<Return>>
    {
        public static IReadOnlyList<Return> Collect(BaseExpr expr)
        {
            var returns = new List<Return>();
            new ReturnCollector().Visit(expr, returns);
            return returns;
        }

        protected override Unit VisitLeafReturn(Return expr, List<Return> context)
        {
            context.Add(expr);
            return base.VisitLeafReturn(expr, context);
        }
    }
}
