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
            var newBody = new IRBlock(stores.Count == 1 ? stores[0] : new IR.Tuple(stores.ToArray()), pf.Parameters);
            return Task.FromResult<BaseFunction>(new Function(pf.Name, newBody));
        }

        return Task.FromResult(pre);
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
}
