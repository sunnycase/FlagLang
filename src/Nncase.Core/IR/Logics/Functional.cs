// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Logics;
using Nncase.IR.Shapes;

namespace Nncase.IR.F;

/// <summary>
/// Logics functional helper.
/// </summary>
public static class Logics
{
    /// <summary>
    /// Call logical and.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <returns>Result expression.</returns>
    public static LogicalAnd LogicalAnd(params LogicalExpr[] operands) => new LogicalAnd(operands);

    public static LogicalAnd LogicalAnd(LogicalExpr left, LogicalExpr right) => new LogicalAnd([left, right]);

    /// <summary>
    /// Call logical or.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <returns>Result expression.</returns>
    public static LogicalOr LogicalOr(params LogicalExpr[] operands) => new LogicalOr(operands);

    public static LogicalOr LogicalOr(LogicalExpr left, LogicalExpr right) => new LogicalOr([left, right]);
}
