// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR;

/// <summary>
/// the Callable Expr.
/// </summary>
public abstract class Callable : Expr
{
    public Callable(string name, BaseExpr[] operands)
        : base(operands)
    {
        Name = name;
    }

    public string Name { get; set; }
}
