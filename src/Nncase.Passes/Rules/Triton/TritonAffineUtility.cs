// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using Nncase;
using Nncase.IR;
using Nncase.IR.Affine;
using Nncase.IR.Distributed;
using Nncase.IR.Logics;
using Nncase.IR.Math;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Utilities;

namespace Nncase.Passes.Rules.Triton;

public static class TritonAffineUtility
{
    public static (AffineRelation? Relation, RankedShape? Symbols) GenerateReadMap(Dimension readDim, LogicalExpr constraint)
    {
        var generator = new AffineAddressGenerator();
        return generator.Generate(readDim, constraint);
    }

    public class ReadDimGenerator : ExprVisitor<BaseExpr?, Unit>
    {
        private readonly List<(DimVar Dim, Expr Value)> _ptrBases = new();
        private readonly List<DimVar> _domains = new();

        public ReadDimGenerator()
        {
        }

        public PtrBaseAndReadDim? GeneratePtrBaseAndReadDim(Expr ptr)
        {
            var addressDim = Visit(ptr) as Dimension;
            if (addressDim is null || _ptrBases.Count > 1)
            {
                // cannot handle multiple ptr bases or null address dimension
                return null;
            }
            else if (_ptrBases.Count == 0)
            {
                return new(Tensor.FromPointer(0, ptr.CheckedTensorType.DType), addressDim!);
            }
            else if (_ptrBases.Count == 1)
            {
                var ptrBase = _ptrBases[0];
                if (ReferenceEquals(ptrBase.Dim, addressDim))
                {
                    return new(ptrBase.Value, Dimension.Zero);
                }
                else if (addressDim is DimSum sum && sum.Operands.ReferenceContains(ptrBase.Dim))
                {
                    return new(ptrBase.Value, sum - ptrBase.Dim);
                }
            }

            return null;
        }

        public LogicalExpr? GenerateMask(Expr mask)
        {
            if (mask is None)
            {
                return LogicalExpr.True;
            }

            return Visit(mask) as LogicalExpr;
        }

        protected override BaseExpr? DefaultVisitLeaf(BaseExpr expr) => null;

        protected override BaseExpr? VisitLeafAsDim(AsDim expr) => Visit(expr.Dim);

        protected override BaseExpr? VisitLeafProgramIdDim(ProgramIdDim expr) => expr;

        protected override BaseExpr? VisitLeafCall(Call expr) => expr.Target switch
        {
            AsTensor => Visit(expr[AsTensor.Input]),
            Cast => Visit(expr[Cast.Input]),

            Binary binary => VisitBinary(expr, binary),
            Range => VisitIRRange(expr),

            Compare compare => VisitCompare(expr, compare),
            _ => null,
        };

        protected override Dimension? VisitLeafTensorConst(TensorConst expr)
        {
            if (expr.Value.Length == 1 && expr.Value.ElementType.IsIntegral())
            {
                return expr.Value.ToScalar<long>();
            }

            return null;
        }

        protected override Dimension? VisitLeafVar(Var expr)
        {
            if (expr.CheckedType is TensorType tt && tt.Shape.IsScalar && (tt.DType.IsIntegral() || tt.DType.IsPointer()))
            {
                var dimVar = new DimVar(expr.Name);
                if (tt.DType.IsPointer())
                {
                    _ptrBases.Add((dimVar, expr));
                }

                return dimVar;
            }

            return null;
        }

        private Dimension? VisitBinary(Call expr, Binary binary)
        {
            var left = Visit(expr[Binary.Lhs]) as Dimension;
            var right = Visit(expr[Binary.Rhs]) as Dimension;
            if (left is null || right is null)
            {
                return null;
            }

            return binary.BinaryOp switch
            {
                BinaryOp.Add => left + right,
                BinaryOp.Sub => left - right,
                BinaryOp.Mul => left * right,
                BinaryOp.FloorDiv => left / right,
                _ => null,
            };
        }

        private DimCompare? VisitCompare(Call expr, Compare compare)
        {
            var left = Visit(expr[Compare.Lhs]) as Dimension;
            var right = Visit(expr[Compare.Rhs]) as Dimension;
            if (left is null || right is null)
            {
                return null;
            }

            return compare.CompareOp switch
            {
                CompareOp.Equal => IR.F.Shapes.Equal(left, right),
                CompareOp.NotEqual => IR.F.Shapes.NotEqual(left, right),
                CompareOp.LowerThan => IR.F.Shapes.LowerThan(left, right),
                CompareOp.LowerOrEqual => IR.F.Shapes.LowerOrEqual(left, right),
                CompareOp.GreaterThan => IR.F.Shapes.GreaterThan(left, right),
                CompareOp.GreaterOrEqual => IR.F.Shapes.GreaterOrEqual(left, right),
                _ => null,
            };
        }

        private Dimension? VisitIRRange(Call expr)
        {
            if (Visit(expr[Range.Begin]) is DimConst start
                && Visit(expr[Range.Step]) is Dimension step)
            {
                var offset = AddDomain();
                if (Visit(expr[Range.End]) is DimConst end)
                {
                    offset.Metadata.Range = new(start.Value, end.Value - 1);
                }

                return start + (offset * step);
            }

            return null;
        }

        private DimVar AddDomain()
        {
            var domain = new DimVar($"d{_domains.Count}");
            _domains.Add(domain);
            return domain;
        }

        public record PtrBaseAndReadDim(Expr PtrBase, Dimension ReadDim);
    }

    private class AffineAddressGenerator : ExprVisitor<AffineExpr?, Unit>
    {
        private readonly List<AffineDim> _domains = new();
        private readonly List<(AffineSymbol Symbol, Dimension Value)> _symbols = new();

        public (AffineRelation? Relation, RankedShape? Symbols) Generate(Dimension dim, LogicalExpr constraint)
        {
            var affineExpr = Visit(dim);
            if (affineExpr is null)
            {
                return (null, null);
            }

            var results = new AffineExpr[] { affineExpr };
            var relation = new AffineRelation(_domains.ToArray(), _symbols.Select(s => s.Symbol).ToArray(), results, constraint);
            return (relation, new RankedShape(_symbols.Select(s => s.Value).ToArray()));
        }

        protected override AffineExpr? DefaultVisitLeaf(BaseExpr expr) => null;

        protected override AffineExpr VisitLeafProgramIdDim(ProgramIdDim expr) => AddSymbol(expr);

        protected override AffineExpr VisitLeafDimVar(DimVar expr)
        {
            var domain = IR.F.Affine.Dim(_domains.Count);
            _domains.Add(domain);
            return domain;
        }

        protected override AffineExpr VisitLeafDimConst(DimConst expr) => expr.Value;

        protected override AffineExpr? VisitLeafDimSum(DimSum expr)
        {
            AffineExpr? result = null;
            foreach (var operand in expr.Operands)
            {
                var affineOperand = Visit(operand);
                if (affineOperand is null)
                {
                    return null;
                }
                else
                {
                    if (result is null)
                    {
                        result = affineOperand;
                    }
                    else
                    {
                        result += affineOperand;
                    }
                }
            }

            if (expr.Bias != 0)
            {
                if (result is null)
                {
                    result = expr.Bias;
                }
                else
                {
                    result += expr.Bias;
                }
            }

            return result;
        }

        protected override AffineExpr? VisitLeafDimProduct(DimProduct expr)
        {
            AffineExpr? linearFactor = null;
            long scale = 1;
            foreach (var factor in expr.Operands)
            {
                var affineFactor = Visit(factor);
                if (affineFactor is null)
                {
                    return null;
                }

                if (affineFactor is AffineConstant constant)
                {
                    scale *= constant.Value;
                    continue;
                }

                if (linearFactor is not null)
                {
                    return null;
                }

                linearFactor = affineFactor;
            }

            if (linearFactor is null)
            {
                linearFactor = scale;
                scale = 1;
            }

            scale *= expr.Scale;

            if (scale != 1)
            {
                linearFactor = linearFactor * scale;
            }

            return linearFactor;
        }

        private AffineSymbol AddSymbol(Dimension value)
        {
            var symbol = IR.F.Affine.Symbol(_symbols.Count);
            _symbols.Add((symbol, value));
            return symbol;
        }
    }
}
