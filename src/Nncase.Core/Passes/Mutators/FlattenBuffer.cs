// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reactive;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Passes.Mutators;

/// <summary>
/// Flatten the multi-dimensional BufferLoad and BufferStore to single-dimensional typed Load/Store.
/// </summary>
public sealed class FlattenBuffer : ExprRewriter
{
    private readonly Dictionary<IVar, BaseExpr> _letBindings = new(ReferenceEqualityComparer.Instance);

    /// <inheritdoc/>
    protected internal override BaseExpr VisitLet(Let expr, Unit context)
    {
        var expression = Visit(expr.Expression, context);
        if (!ReferenceEquals(expr.Expression, expression))
        {
            expr.ReplaceOperandAt(1, expression);
            SetMutated();
        }

        var hadBinding = _letBindings.TryGetValue(expr.Var, out var oldBinding);
        _letBindings[expr.Var] = expression;
        BaseExpr body;
        try
        {
            body = Visit(expr.Body, context);
        }
        finally
        {
            if (hadBinding)
            {
                _letBindings[expr.Var] = oldBinding!;
            }
            else
            {
                _letBindings.Remove(expr.Var);
            }
        }

        if (body is not Sequential sequential)
        {
            throw new InvalidOperationException($"FlattenBuffer expects let body to remain Sequential, got {body.GetType().Name}.");
        }

        if (!ReferenceEquals(expr.Body, sequential))
        {
            expr.ReplaceOperandAt(2, sequential);
            SetMutated();
        }

        if (IsPureViewBinding(expression) && !ContainsReference(sequential, expr.Var))
        {
            return sequential;
        }

        return expr;
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafBlock(Block expr)
    {
        // TODO: put the unfold block into this.
        if (expr.Predicate is TensorConst tc && tc.Value.ToScalar<bool>() == true)
        {
            return expr.Body;
        }

        return T.Nop();
    }

    /// <inheritdoc/>
    protected override Expr RewriteLeafCall(Call expr)
    {
        if (expr.Target is IR.Buffers.BufferLoad)
        {
            var indices = (IR.Tuple)expr[IR.Buffers.BufferLoad.Indices];
            var access = ResolveAccess(expr[IR.Buffers.BufferLoad.Input], TupleIndices(indices), "BufferLoad");
            return T.Load(access.Handle, access.Index);
        }
        else if (expr.Target is IR.Buffers.BufferStore)
        {
            var indices = (IR.Tuple)expr[IR.Buffers.BufferStore.Indices];
            var access = ResolveAccess(expr[IR.Buffers.BufferStore.Input], TupleIndices(indices), "BufferStore");
            return T.Store(access.Handle, access.Index, (Expr)expr[IR.Buffers.BufferStore.Value]);
        }
        else if (expr.Target is TIR.Load && IsViewAccess(expr[TIR.Load.Handle]))
        {
            var access = ResolveAccess(expr[TIR.Load.Handle], new[] { expr[TIR.Load.Index].AsDim() }, "TIR.Load");
            return T.Load(access.Handle, access.Index);
        }
        else if (expr.Target is TIR.Store && IsViewAccess(expr[TIR.Store.Handle]))
        {
            var access = ResolveAccess(expr[TIR.Store.Handle], new[] { expr[TIR.Store.Index].AsDim() }, "TIR.Store");
            return T.Store(access.Handle, access.Index, (Expr)expr[TIR.Store.Value]);
        }
        else if (expr.Target is IR.Buffers.MatchBuffer && expr.Arguments[0] is TIR.Buffer { MemSpan: { Start: DimConst or DimVar } })
        {
            // remove the all fixed match operation.
            return T.Nop();
        }

        return expr;
    }

    private static bool ContainsReference(BaseExpr expr, IVar var) =>
        ExprCollector.Collect(expr).Any(candidate => ReferenceEquals(candidate, var));

    private static bool IsPureViewBinding(BaseExpr expr) =>
        expr is TIR.Buffer ||
        expr is Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView };

    private static Call ReinterpretHandle(TIR.Buffer input) =>
        new(new IR.Tensors.Cast(new PointerType(input.ElemType), CastMode.Reinterpret), input.MemSpan);

    private static Dimension LinearIndex(TIR.Buffer input, IReadOnlyList<Dimension> indices)
    {
        if (indices.Count != input.Rank)
        {
            throw new InvalidOperationException($"FlattenBuffer expects {input.Rank} indices for buffer {input.Name}, got {indices.Count}.");
        }

        return Enumerable.Range(0, indices.Count)
            .Aggregate((Dimension)0, (acc, i) => acc + (input.Strides[i] * indices[i]));
    }

    private static Dimension[] TupleIndices(IR.Tuple indices)
    {
        var result = new Dimension[indices.Count];
        for (int i = 0; i < indices.Count; i++)
        {
            result[i] = indices[i].AsDim();
        }

        return result;
    }

    private bool IsViewAccess(BaseExpr expr) =>
        expr is TIR.Buffer ||
        expr is Call { Target: IR.Buffers.BufferSubview or IR.Buffers.AllocateBufferView } ||
        (expr is IVar var && _letBindings.ContainsKey(var));

    private Access ResolveAccess(BaseExpr input, IReadOnlyList<Dimension> indices, string role)
    {
        if (input is IVar var && _letBindings.TryGetValue(var, out var binding))
        {
            return ResolveAccess(binding, indices, role);
        }

        if (input is TIR.Buffer buffer)
        {
            return new Access(ReinterpretHandle(buffer), LinearIndex(buffer, indices));
        }

        if (input is Call { Target: IR.Buffers.AllocateBufferView } allocateBufferView)
        {
            return ResolveAccess(allocateBufferView[IR.Buffers.AllocateBufferView.Buffer], indices, role);
        }

        if (input is Call { Target: IR.Buffers.BufferSubview } subview)
        {
            var offsets = GetRankedShape(subview, IR.Buffers.BufferSubview.Offset, role);
            var shape = GetRankedShape(subview, IR.Buffers.BufferSubview.Shape, role);
            if (offsets.Length != shape.Length)
            {
                throw new InvalidOperationException($"{role} subview offset rank {offsets.Length} does not match shape rank {shape.Length}.");
            }

            var parentIndices = CombineSubviewIndices(offsets, indices, role);
            return ResolveAccess(subview[IR.Buffers.BufferSubview.Buffer], parentIndices, role);
        }

        if (IsPointerLike(input))
        {
            return indices.Count switch
            {
                0 => new Access(input, Dimension.Zero),
                1 => new Access(input, indices[0]),
                _ => throw new InvalidOperationException($"{role} raw pointer access must use a single linear index, got {indices.Count}."),
            };
        }

        throw new NotSupportedException($"{role} cannot be flattened from {input.GetType().Name} with type {input.CheckedType}.");
    }

    private Dimension[] GetRankedShape(Call call, ParameterInfo parameter, string role)
    {
        if (call[parameter] is not RankedShape shape)
        {
            throw new InvalidOperationException($"{role} requires ranked BufferSubview {parameter.Name}, got {call[parameter].GetType().Name}.");
        }

        return shape.ToArray();
    }

    private Dimension[] CombineSubviewIndices(IReadOnlyList<Dimension> offsets, IReadOnlyList<Dimension> indices, string role)
    {
        if (offsets.Count == indices.Count)
        {
            var combined = new Dimension[offsets.Count];
            for (int i = 0; i < combined.Length; i++)
            {
                combined[i] = offsets[i] + indices[i];
            }

            return combined;
        }

        if (offsets.Count == 0 && indices.Count == 1)
        {
            return new[] { indices[0] };
        }

        throw new InvalidOperationException($"{role} cannot flatten rank-{offsets.Count} subview with {indices.Count} access indices.");
    }

    private bool IsPointerLike(BaseExpr expr)
    {
        try
        {
            return expr.CheckedDataType is PointerType;
        }
        catch (InvalidOperationException)
        {
            return false;
        }
    }

    private sealed record Access(BaseExpr Handle, Dimension Index);
}
