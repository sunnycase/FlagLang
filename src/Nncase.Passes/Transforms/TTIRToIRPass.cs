// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
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
            var returns = ReturnCollector.Collect(pf.Body);
            var body = returns.Count == 0 ? pf.Body : BuildReturnBody(pf.Body, returns);
            var newBody = new IRBlock(body, pf.Parameters);
            return Task.FromResult<BaseFunction>(new Function(pf.Name, newBody));
        }

        return Task.FromResult(pre);
    }

    private static BaseExpr BuildReturnBody(Sequential body, IReadOnlyList<Return> returns)
    {
        if (returns.Count == 1 && body.HasTerminator && ReferenceEquals(body.Fields[^1], returns[0]))
        {
            return BuildTerminalReturnBody(body, returns[0]);
        }

        if (!IsTopLevelReturnBody(body))
        {
            throw new InvalidOperationException("Triton helper return conversion requires returns to be either a terminal void return or a side-effect-free terminal value return.");
        }

        if (returns.Count != 1)
        {
            throw new InvalidOperationException($"Expected at most one return in Triton helper function, got {returns.Count}.");
        }

        return PackReturnValues(returns[0]);
    }

    private static BaseExpr BuildTerminalReturnBody(Sequential body, Return ret)
    {
        if (ret.Values.Length == 0)
        {
            return body.With(fields: body.Fields[..^1].ToArray());
        }

        if (HasSideEffectingPrefix(body))
        {
            throw new InvalidOperationException("Triton helper value return conversion rejects side-effecting statements before the terminal return.");
        }

        return PackReturnValues(ret);
    }

    private static BaseExpr PackReturnValues(Return ret)
    {
        var values = ret.Values.ToArray();
        return values.Length switch
        {
            0 => new IR.Tuple(),
            1 => values[0],
            _ => new IR.Tuple(values),
        };
    }

    private static bool HasSideEffectingPrefix(Sequential body)
    {
        for (var i = 0; i < body.Count - 1; i++)
        {
            if (HasSideEffect(body[i]))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasSideEffect(BaseExpr expr) => expr switch
    {
        Return => true,
        For => true,
        IfThenElse => true,
        Call { Target: IR.Triton.Store } => true,
        Call { Target: IR.Buffers.BufferStore } => true,
        Call { Target: TIR.Store } => true,
        _ => false,
    };

    private static bool IsTopLevelReturnBody(Sequential body)
    {
        if (body.Count == 0)
        {
            return false;
        }

        foreach (var field in body.Fields)
        {
            if (field is not Return)
            {
                return false;
            }
        }

        return true;
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
