// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.IR.Logics;

public abstract partial class LogicalExpr : BaseExpr
{
    public static readonly LogicalConst True = new LogicalConst(true);

    public static readonly LogicalConst False = new LogicalConst(false);

    /// <summary>
    /// Initializes a new instance of the <see cref="LogicalExpr"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    protected LogicalExpr(IEnumerable<BaseExpr> operands)
        : base(operands)
    {
    }
}
