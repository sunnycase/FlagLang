// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using NetFabric.Hyperlinq;
using Nncase;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.IR.Triton;
using Nncase.Passes;
using Nncase.PatternMatch;
using Nncase.Utilities;
using static Nncase.IR.TypePatternUtility;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes.Rules.Triton;

[RuleGenerator]
public sealed partial class StoreToAffineScatter : IRewriteRule
{
    public IPattern Pattern { get; } =
        Nncase.PatternMatch.F.Triton.IsStore(
            target_name: "store",
            "call",
            _ => true,
            IsWildcard("ptr"),
            IsWildcard("value"),
            IsWildcard("mask"));

    public Expr? GetReplace(Expr call, IR.Triton.Store store, Expr ptr, Expr value, Expr mask)
    {
        var generator = new TritonAffineUtility.ReadDimGenerator();
        var ptrBaseAndReadDim = generator.GeneratePtrBaseAndReadDim(ptr);
        if (ptrBaseAndReadDim is null)
        {
            return null;
        }

        var constraintExpr = generator.GenerateMask(mask);
        if (constraintExpr is null)
        {
            return null;
        }

        var (relation, symbols) = TritonAffineUtility.GenerateReadMap(ptrBaseAndReadDim.ReadDim, constraintExpr);
        if (relation is null || symbols is null)
        {
            return null;
        }

        return IR.F.Affine.Scatter(value, ptrBaseAndReadDim.PtrBase, relation, symbols);
    }
}
