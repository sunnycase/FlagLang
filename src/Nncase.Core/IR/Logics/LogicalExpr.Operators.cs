// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;

namespace Nncase.IR.Logics;

public partial class LogicalExpr
{
    public static implicit operator LogicalExpr(bool value) => value ? True : False;

    public static LogicalAnd operator &(LogicalExpr left, LogicalExpr right) => IR.F.Logics.LogicalAnd(left, right);

    public static LogicalOr operator |(LogicalExpr left, LogicalExpr right) => IR.F.Logics.LogicalOr(left, right);
}
