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
            var body = returns.Count == 0 ? BuildEffectRoot(pf.Body.Fields) : BuildReturnBody(pf.Body, returns);
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
            return BuildEffectRoot(body.Fields[..^1]);
        }

        var returnValue = PackReturnValues(ret);
        return BuildMemoryDependency(body.Fields[..^1]) is { } dependency
            ? IR.F.Tensors.Depend(dependency, returnValue)
            : returnValue;
    }

    private static BaseExpr BuildEffectRoot(ReadOnlySpan<Expr> fields)
    {
        return BuildMemoryDependency(fields) ?? (fields.Length == 0 ? new IR.Tuple() : fields[^1]);
    }

    private static BaseExpr? BuildMemoryDependency(ReadOnlySpan<Expr> fields)
    {
        var memoryRoots = new List<Expr>();
        foreach (var field in fields)
        {
            if (HasMemorySideEffect(field))
            {
                memoryRoots.Add(field);
            }
        }

        return memoryRoots.Count switch
        {
            0 => null,
            1 => memoryRoots[0],
            _ => new IR.Tuple(memoryRoots.ToArray()),
        };
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

    private static bool HasMemorySideEffect(BaseExpr expr) => expr switch
    {
        Call { Target: IR.Triton.Store } => true,
        Call { Target: IR.Buffers.BufferStore } => true,
        Call { Target: TIR.Store } => true,
        Call { Target: TIR.Memcopy } => true,
        Call { Target: IR.Affine.Scatter } => true,
        If @if => HasMemorySideEffect(@if.Then.Body) || HasMemorySideEffect(@if.Else.Body) || HasMemorySideEffect(@if.Arguments),
        Let let => HasMemorySideEffect(let.Expression) || HasMemorySideEffect(let.Body),
        IRBlock block => HasMemorySideEffect(block.Body),
        Sequential sequential => HasMemorySideEffect(sequential.Fields),
        For @for => HasMemorySideEffect(@for.Body),
        IfThenElse ifThenElse => HasMemorySideEffect(ifThenElse.Then) || HasMemorySideEffect(ifThenElse.Else),
        _ => false,
    };

    private static bool HasMemorySideEffect(ReadOnlySpan<BaseExpr> exprs)
    {
        foreach (var expr in exprs)
        {
            if (HasMemorySideEffect(expr))
            {
                return true;
            }
        }

        return false;
    }

    private static bool HasMemorySideEffect(ReadOnlySpan<Expr> exprs)
    {
        foreach (var expr in exprs)
        {
            if (HasMemorySideEffect(expr))
            {
                return true;
            }
        }

        return false;
    }

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
