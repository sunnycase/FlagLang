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
/// Shapes functional helper.
/// </summary>
public static class Shapes
{
    public static Call AsTensor(Dimension input) => new Call(new AsTensor(), input);

    public static Call AsTensor(Shape input) => new Call(new AsTensor(), input);

    /// <summary>
    /// Call compare.
    /// </summary>
    /// <param name="compareOp">Compare operator.</param>
    /// <param name="lhs">Left operand.</param>
    /// <param name="rhs">Right operand.</param>
    /// <returns>Result expression.</returns>
    public static DimCompare Compare(CompareOp compareOp, Dimension lhs, Dimension rhs) => new DimCompare(compareOp, lhs, rhs);

    /// <summary>
    /// Call equal.
    /// </summary>
    public static DimCompare Equal(Dimension lhs, Dimension rhs) => Compare(CompareOp.Equal, lhs, rhs);

    /// <summary>
    /// call not equal.
    /// </summary>
    public static DimCompare NotEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.NotEqual, lhs, rhs);

    /// <summary>
    /// call less than.
    /// </summary>
    public static DimCompare LowerThan(Dimension lhs, Dimension rhs) => Compare(CompareOp.LowerThan, lhs, rhs);

    /// <summary>
    /// call less equal.
    /// </summary>
    public static DimCompare LowerOrEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.LowerOrEqual, lhs, rhs);

    /// <summary>
    /// call greater equal.
    /// </summary>
    public static DimCompare GreaterOrEqual(Dimension lhs, Dimension rhs) => Compare(CompareOp.GreaterOrEqual, lhs, rhs);

    /// <summary>
    /// call greater than.
    /// </summary>
    public static DimCompare GreaterThan(Dimension lhs, Dimension rhs) => Compare(CompareOp.GreaterThan, lhs, rhs);
}
