// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Logics;
using Nncase.IR.Shapes;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes
{
    /// <summary>
    /// Lowers <see cref="TIR.NTT.AffineGather"/> and <see cref="TIR.NTT.AffineScatter"/> into canonical buffer loops
    /// so downstream codegen paths can be reused.
    /// </summary>
    public sealed class NTTAffineIOLoweringPass : FunctionPass
    {
        private readonly AffineIOLoweringRewriter _rewriter = new();

        /// <inheritdoc/>
        protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context) => Task.FromResult(input switch
        {
            PrimFunction primFunc => LowerPrimFunction(primFunc),
            PrimFunctionWrapper { Target: PrimFunction target } wrapper => LowerWrapper(wrapper, target),
            _ => input,
        });

        private PrimFunction LowerPrimFunction(PrimFunction func)
        {
            var newBody = (Sequential)_rewriter.Visit(func.Body, default);
            return ReferenceEquals(newBody, func.Body) ? func : func.With(body: newBody);
        }

        private BaseFunction LowerWrapper(PrimFunctionWrapper wrapper, PrimFunction target)
        {
            var loweredTarget = LowerPrimFunction(target);
            return ReferenceEquals(loweredTarget, target) ? wrapper : wrapper.With(target: loweredTarget);
        }

        private sealed class AffineIOLoweringRewriter : ExprRewriter<Unit>
        {
            protected override BaseExpr RewriteLeafCall(Call expr, Unit context)
            {
                return expr.Target switch
                {
                    TIR.NTT.AffineGather gather => LowerGather(expr, gather, context),
                    TIR.NTT.AffineScatter scatter => LowerScatter(expr, scatter, context),
                    _ => base.RewriteLeafCall(expr, context),
                };
            }

            private Expr LowerGather(Call call, TIR.NTT.AffineGather gather, Unit context)
            {
                var source = (Expr)Visit(call[TIR.NTT.AffineGather.Source], context);
                _ = Visit(call[TIR.NTT.AffineGather.DefaultValue], context);
                var output = RequireBuffer(Visit(call[TIR.NTT.AffineGather.Output], context));

                ValidateRelation(gather.Relation, output.Dimensions.Length);

                var extents = output.Dimensions.ToArray();
                var symbolMap = BuildSymbolMap(gather.Relation, gather.Symbols);
                return BuildLoopNest(extents, loopVars =>
                {
                    var address = EvaluateAddress(gather.Relation, loopVars, extents, symbolMap);
                    var loaded = T.Load(source, address);
                    return T.BufferStore(output, loopVars.AsExprs(), loaded);
                });
            }

            private Expr LowerScatter(Call call, TIR.NTT.AffineScatter scatter, Unit context)
            {
                var source = RequireBuffer(Visit(call[TIR.NTT.AffineScatter.Source], context));
                var dest = (Expr)Visit(call[TIR.NTT.AffineScatter.Dest], context);

                ValidateRelation(scatter.Relation, source.Dimensions.Length);

                var extents = source.Dimensions.ToArray();
                var symbolMap = BuildSymbolMap(scatter.Relation, scatter.Symbols);
                return BuildLoopNest(extents, loopVars =>
                {
                    var value = T.BufferLoad(source, loopVars.AsExprs());
                    var address = EvaluateAddress(scatter.Relation, loopVars, extents, symbolMap);
                    return T.Store(dest, address, value);
                });
            }

            private Expr BuildLoopNest(Dimension[] extents, Func<DimVar[], Expr> bodyFactory)
            {
                var loopVars = new DimVar[extents.Length];
                return BuildLoopNestRecursive(extents, loopVars, 0, bodyFactory);
            }

            private Expr BuildLoopNestRecursive(IReadOnlyList<Dimension> extents, DimVar[] loopVars, int axis, Func<DimVar[], Expr> bodyFactory)
            {
                if (axis == loopVars.Length)
                {
                    return bodyFactory(loopVars);
                }

                var range = new TIR.Range(Dimension.Zero, extents[axis], Dimension.One);
                var loopBuilder = T.Serial(out loopVars[axis], range, $"d{axis}");
                var inner = BuildLoopNestRecursive(extents, loopVars, axis + 1, bodyFactory);
                return loopBuilder.Body(inner).Build();
            }

            private void ValidateRelation(AffineRelation relation, int expectedRank)
            {
                if (relation.Constraint != LogicalExpr.True)
                {
                    throw new NotSupportedException("Masked affine IO is not supported yet.");
                }

                if (relation.Symbols.Length != 0)
                {
                    throw new NotSupportedException("Symbolic affine IO is not supported yet.");
                }

                if (relation.Domains.Length != expectedRank)
                {
                    throw new NotSupportedException($"Domain rank {relation.Domains.Length} does not match buffer rank {expectedRank}.");
                }
            }

            private Dimension EvaluateAddress(AffineRelation relation, DimVar[] loopVars, IReadOnlyList<Dimension> extents, IReadOnlyDictionary<AffineSymbol, Dimension>? symbolMap)
            {
                var domainValues = new Dimension[loopVars.Length];
                for (int i = 0; i < loopVars.Length; i++)
                {
                    domainValues[i] = loopVars[i];
                }

                return EvaluateAffineExpr(relation.Results[0], domainValues, extents, symbolMap);
            }

            private Dimension EvaluateAffineExpr(AffineExpr expr, IReadOnlyList<Dimension> dims, IReadOnlyList<Dimension> extents, IReadOnlyDictionary<AffineSymbol, Dimension>? symbols)
            {
                return expr switch
                {
                    AffineConstant constant => constant.Value,
                    AffineDim dim => dims[dim.Position],
                    AffineExtent extent => extents[extent.Position],
                    AffineSymbol symbol => symbols is null ? throw new NotSupportedException("Symbolic relations require bound symbol map.") : symbols[symbol],
                    AffineAddBinary add => EvaluateAffineExpr(add.Lhs, dims, extents, symbols) + EvaluateAffineExpr(add.Rhs, dims, extents, symbols),
                    AffineMulBinary mul => EvaluateAffineExpr(mul.Lhs, dims, extents, symbols) * EvaluateAffineExpr(mul.Rhs, dims, extents, symbols),
                    AffineDivBinary div => ApplyDivBinary(div.BinaryOp, EvaluateAffineExpr(div.Lhs, dims, extents, symbols), EvaluateAffineExpr(div.Rhs, dims, extents, symbols)),
                    _ => throw new NotSupportedException($"Unsupported affine expression node {expr.GetType().Name}"),
                };
            }

            private Dimension ApplyDivBinary(AffineDivBinaryOp op, Dimension lhs, Dimension rhs) => op switch
            {
                AffineDivBinaryOp.FloorDiv => lhs / rhs,
                AffineDivBinaryOp.CeilDiv => Dimension.CeilDiv(lhs, rhs),
                AffineDivBinaryOp.Mod => lhs % rhs,
                _ => throw new ArgumentOutOfRangeException(nameof(op), $"Unsupported affine division operator {op}"),
            };

            private IReadOnlyDictionary<AffineSymbol, Dimension>? BuildSymbolMap(AffineRelation relation, RankedShape symbols)
            {
                if (relation.Symbols.Length == 0)
                {
                    return null;
                }

                var dims = symbols.Dimensions.ToArray();
                if (dims.Length != relation.Symbols.Length)
                {
                    throw new InvalidOperationException("Symbol payload does not match relation requirement.");
                }

                var map = new Dictionary<AffineSymbol, Dimension>(dims.Length, ReferenceEqualityComparer.Instance);
                for (int i = 0; i < dims.Length; i++)
                {
                    map.Add(relation.Symbols[i], dims[i]);
                }

                return map;
            }

            private TIR.Buffer RequireBuffer(BaseExpr expr)
            {
                if (expr is not TIR.Buffer buffer)
                {
                    throw new NotSupportedException("Affine IO lowering expects buffer arguments.");
                }

                return buffer;
            }
        }
    }
}

internal static class DimVarExtensions
{
    public static Expr[] AsExprs(this IReadOnlyList<DimVar> loopVars)
    {
        var result = new Expr[loopVars.Count];
        for (int i = 0; i < loopVars.Count; i++)
        {
            result[i] = global::Nncase.IR.F.Shapes.AsTensor(loopVars[i]);
        }

        return result;
    }
}
