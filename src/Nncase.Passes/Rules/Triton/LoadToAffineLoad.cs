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
using isl = IntegerSetLibrary;

namespace Nncase.Passes.Rules.Triton;

[RuleGenerator]
public sealed partial class LoadToAffineLoad : IRewriteRule
{
    public IPattern Pattern { get; } =
        Nncase.PatternMatch.F.Triton.IsLoad(
            target_name: "load",
            "call",
            _ => true,
            IsWildcard("ptr"),
            IsWildcard("mask"),
            IsWildcard("other"));

    public Expr? GetReplace(Expr call, IR.Triton.Load load, Expr ptr, Expr mask, Expr other)
    {
        var generator = new ReadDimGenerator();
        var ptrBaseAndReadDim = generator.GeneratePtrBaseAndReadDim(ptr);
        if (ptrBaseAndReadDim is null)
        {
            return null;
        }

        var readMap = GenerateReadMap(ptrBaseAndReadDim.ReadDim);
        if (readMap is null)
        {
            return null;
        }

        return IR.F.Affine.Gather(ptrBaseAndReadDim.PtrBase, readMap, ptr.CheckedShape);
    }

    private AffineRelation? GenerateReadMap(Dimension readDim)
    {
        var generator = new AffineAddressGenerator();
        return generator.Generate(readDim);
    }

    private class ReadDimGenerator : ExprVisitor<Dimension?, Unit>
    {
        private readonly List<(DimVar Dim, Expr Value)> _ptrBases = new();
        private readonly List<DimVar> _domains = new();

        public ReadDimGenerator()
        {
        }

        public PtrBaseAndReadDim? GeneratePtrBaseAndReadDim(Expr ptr)
        {
            var addressDim = Visit(ptr);
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

        protected override Dimension? DefaultVisitLeaf(BaseExpr expr) => null;

        protected override Dimension? VisitLeafAsDim(AsDim expr) => Visit(expr.Dim);

        protected override Dimension? VisitLeafProgramIdDim(ProgramIdDim expr) => expr;

        protected override Dimension? VisitLeafCall(Call expr) => expr.Target switch
        {
            AsTensor => Visit(expr[AsTensor.Input]),
            Cast => Visit(expr[Cast.Input]),

            Binary binary => VisitBinary(expr, binary),
            Range => VisitIRRange(expr),
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
            var left = Visit(expr[Binary.Lhs]);
            var right = Visit(expr[Binary.Rhs]);
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
        private readonly List<(AffineSymbol Symbol, BaseExpr Value)> _symbols = new();

        public AffineRelation? Generate(Dimension dim)
        {
            var affineExpr = Visit(dim);
            if (affineExpr is null)
            {
                return null;
            }

            var results = new AffineExpr[] { affineExpr };
            return new AffineRelation(_domains.ToArray(), _symbols.Select(s => s.Symbol).ToArray(), results);
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
            AffineExpr? result = null;
            foreach (var factor in expr.Operands)
            {
                if (Visit(factor) is AffineConstantOrSymbol affineFactor)
                {
                    if (result is null)
                    {
                        result = affineFactor;
                    }
                    else
                    {
                        result *= affineFactor;
                    }
                }
                else
                {
                    return null;
                }
            }

            if (expr.Scale != 1)
            {
                if (result is null)
                {
                    result = expr.Scale;
                }
                else
                {
                    result *= expr.Scale;
                }
            }

            return result!;
        }

        private AffineSymbol AddSymbol(BaseExpr value)
        {
            var symbol = IR.F.Affine.Symbol(_symbols.Count);
            _symbols.Add((symbol, value));
            return symbol;
        }
    }
}
