// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.IR.Distributed;

public sealed class ProgramIdDim : Dimension, IEquatable<ProgramIdDim?>
{
    public ProgramIdDim(int axis)
        : base(Array.Empty<Expr>())
    {
        Axis = axis;
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    public int Axis { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitProgramIdDim(this, context);

    public ProgramIdDim With(int? axis = null) => new(axis ?? Axis);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ProgramIdDim);

    /// <inheritdoc/>
    public bool Equals(ProgramIdDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Axis == other.Axis;
    }

    public override string ToString() => $"pid[{Axis}]";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => Axis.GetHashCode();
}
