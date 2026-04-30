// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Affine;
using Nncase.IR.Affine.Builders;

namespace Nncase.IR.F;

public static class Affine
{
    public static AffineDim Dim(int position) => new AffineDim(position);

    public static AffineDim[] Dims(int count) => Enumerable.Range(0, count).Select(Dim).ToArray();

    public static AffineExtent Extent(int position) => new AffineExtent(position);

    public static AffineDomain Domain(int position) => new AffineDomain(Dim(position), Extent(position));

    public static AffineDomain[] Domains(int count) => Enumerable.Range(0, count).Select(Domain).ToArray();

    public static AffineSymbol Symbol(int position) => new AffineSymbol(position);

    public static AffineSymbol[] Symbols(int count) => Enumerable.Range(0, count).Select(Symbol).ToArray();

    public static AffineDivBinary FloorDiv(this AffineExpr lhs, AffineConstantOrSymbol rhs) =>
        new AffineDivBinary(AffineDivBinaryOp.FloorDiv, lhs, rhs);

    public static string ToString(AffineDivBinaryOp binaryOp, AffineExpr lhs, AffineExpr rhs) => binaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => $"floor({lhs} / {rhs})",
        AffineDivBinaryOp.CeilDiv => $"ceil({lhs} / {rhs})",
        AffineDivBinaryOp.Mod => $"({lhs} % {rhs})",
        _ => throw new ArgumentOutOfRangeException(nameof(binaryOp)),
    };

    public static string GetDisplayString(AffineDivBinaryOp binaryOp, AffineExpr lhs, AffineExpr rhs, ReadOnlySpan<AffineSymbol> symbols) => binaryOp switch
    {
        AffineDivBinaryOp.FloorDiv => $"floor({lhs.GetDisplayString(symbols)} / {rhs.GetDisplayString(symbols)})",
        AffineDivBinaryOp.CeilDiv => $"ceil({lhs.GetDisplayString(symbols)} / {rhs.GetDisplayString(symbols)})",
        AffineDivBinaryOp.Mod => $"({lhs.GetDisplayString(symbols)} % {rhs.GetDisplayString(symbols)})",
        _ => throw new ArgumentOutOfRangeException(nameof(binaryOp)),
    };

    public static Call Gather(Expr source, AffineRelation relation, RankedShape symbols, Shape shape, Expr defaultValue) => new Call(new Gather(relation, symbols, shape, new IRArray<SBP>(), new Placement(new IRArray<int>(), string.Empty)), source, defaultValue);

    public static Call Gather(Expr source, AffineRelation relation, RankedShape symbols, Shape shape, Expr defaultValue, IRArray<SBP> ndsbp, Placement placement) => new Call(new Gather(relation, symbols, shape, ndsbp, placement), source, defaultValue);

    public static Call Scatter(Expr source, Expr dest, AffineRelation relation, RankedShape symbols) => new Call(new Scatter(relation, symbols), source, dest);

    public static For For(int memoryLevel, AffineMap domain, Expr body) => new For(memoryLevel, domain, body);

    public static IGridBuilder Grid() => new GridBuilder();
}
