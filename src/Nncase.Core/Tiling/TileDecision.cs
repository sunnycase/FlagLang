// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Globalization;
using System.Linq;
using Nncase.IR;
using Nncase.TIR;

namespace Nncase.Tiling;

public sealed record TileLifetime(int Start, int End)
{
    public override string ToString() => $"[{Start}, {End}]";
}

public sealed record TileCapacity(long RequestedBytes, long? BudgetBytes, string Source)
{
    public override string ToString()
    {
        var budget = BudgetBytes.HasValue ? BudgetBytes.Value.ToString(CultureInfo.InvariantCulture) : "unknown";
        return $"requested={RequestedBytes.ToString(CultureInfo.InvariantCulture)}, budget={budget}, source={Source}";
    }
}

public sealed record TileDecision(
    string Id,
    string OpKind,
    Shape TileShape,
    DistributionLayout? DistributionLayout,
    StorageLayout StorageLayout,
    BufferStorage Storage,
    TileLifetime Lifetime,
    long ByteSize,
    TileCapacity Capacity,
    bool RequiresSynchronization,
    string Reason)
{
    public string ToDumpString()
    {
        var distribution = DistributionLayout is null
            ? "none"
            : $"{DistributionLayout.Kind}: {DistributionLayout.GlobalToOwnerLocal}";
        return string.Join(
            Environment.NewLine,
            [
                $"- id: {Id}",
                $"  op: {OpKind}",
                $"  tile_shape: {TileShape}",
                $"  distribution: {distribution}",
                $"  storage_layout: {StorageLayout.Kind}: {StorageLayout.LogicalToPhysical}",
                $"  storage: {Storage}",
                $"  lifetime: {Lifetime}",
                $"  bytes: {ByteSize.ToString(CultureInfo.InvariantCulture)}",
                $"  capacity: {Capacity}",
                $"  requires_sync: {RequiresSynchronization}",
                $"  reason: {Reason}",
            ]);
    }
}

public static class TileDecisionMetadata
{
    public const string DecisionAttributeKey = "nncase.tiling.decision";

    public const string RequiredAttributeKey = "nncase.tiling.required";

    public static void Set(BaseExpr expr, TileDecision decision)
    {
        expr.Metadata.Attributes[DecisionAttributeKey] = decision;
        expr.Metadata.Attributes[RequiredAttributeKey] = true;
    }

    public static void MarkRequired(BaseExpr expr)
    {
        expr.Metadata.Attributes[RequiredAttributeKey] = true;
    }

    public static bool IsRequired(BaseExpr expr) =>
        expr.Metadata.Attributes.TryGetValue(RequiredAttributeKey, out var value) && value is true;

    public static bool TryGet(BaseExpr expr, out TileDecision decision)
    {
        if (expr.Metadata.Attributes.TryGetValue(DecisionAttributeKey, out var value) && value is TileDecision found)
        {
            decision = found;
            return true;
        }

        decision = null!;
        return false;
    }

    public static TileDecision Require(BaseExpr expr, string context)
    {
        if (TryGet(expr, out var decision))
        {
            return decision;
        }

        throw new InvalidOperationException($"{context} requires a tile decision for {FormatExpr(expr)}, but none was attached by the tiling pass.");
    }

    private static string FormatExpr(BaseExpr expr)
    {
        var type = expr is Expr typed ? typed.CheckedType.ToString() : "<non-expr>";
        return $"{expr.GetType().Name}({type})";
    }
}
