// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR.Logics;

namespace Nncase.IR;

public abstract partial class Dimension
{
    /// <summary>
    /// GreaterEqual.
    /// </summary>
    public static DimCompare operator >=(Dimension lhs, Dimension rhs) => IR.F.Shapes.GreaterOrEqual(lhs, rhs);

    /// <summary>
    /// GreaterThan.
    /// </summary>
    public static DimCompare operator >(Dimension lhs, Dimension rhs) => IR.F.Shapes.GreaterThan(lhs, rhs);

    /// <summary>
    /// LessEqual.
    /// </summary>
    public static DimCompare operator <=(Dimension lhs, Dimension rhs) => IR.F.Shapes.LowerOrEqual(lhs, rhs);

    /// <summary>
    /// LessThan.
    /// </summary>
    public static DimCompare operator <(Dimension lhs, Dimension rhs) => IR.F.Shapes.LowerThan(lhs, rhs);
}
