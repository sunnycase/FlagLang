// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using Nncase.IR;

namespace Nncase.Tests.TestFixture;

public static class ExprEvaluationExtensions
{
    public static Expr UnwrapBody(this Expr expr)
    {
        var current = expr;
        while (true)
        {
            switch (current)
            {
                case Function function:
                    current = function.Body;
                    break;
                case IRBlock block when block.Body is Expr nested:
                    current = nested;
                    break;
                default:
                    return current;
            }
        }
    }

    public static IValue EvaluateUnwrapped(this Expr expr)
    {
        return expr.UnwrapBody().Evaluate();
    }

    public static IValue EvaluateUnwrapped(this Expr expr, IReadOnlyDictionary<IVar, IValue> feedDict)
    {
        return expr.UnwrapBody().Evaluate(feedDict);
    }
}
