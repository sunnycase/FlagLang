// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.Utilities;

namespace Nncase.IR;

public enum HierarchyKind : byte
{
    Parallel = 0,
    SMT = 1,
}

/// <summary>
/// Per-thread element ownership order inside a Triton blocked tile.
/// </summary>
public enum TritonThreadElementOrder : byte
{
    Contiguous,
    Strided,
}

[JsonConverter(typeof(SBPConverter))]
public abstract record SBP
{
    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPPartial P(ReduceOp op = ReduceOp.Sum) => new SBPPartial(op);

    public static SBPSplit S(IRArray<int> axes) => new SBPSplit(axes);

    public static SBPSplit S(params int[] axes) => new SBPSplit(axes);
}

public sealed record SBPSplit(IRArray<int> Axes) : SBP
{
    public override string ToString() => $"S({string.Join(",", Axes)})";
}

public sealed record SBPPartial(ReduceOp Op) : SBP
{
    public override string ToString() => $"P({Op})";
}

public sealed record SBPBroadCast : SBP
{
    public static readonly SBPBroadCast Instance = new SBPBroadCast();

    public override string ToString() => "B";
}

public class SBPConverter : JsonConverter<SBP>
{
    public override SBP Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        string? typeDiscriminator = null;
        SBPSplit? sbpSplit = null;
        SBPPartial? sbpPartial = null;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string? propertyName = reader.GetString();
                reader.Read(); // Move to property value

                switch (propertyName)
                {
                    case "$type":
                        typeDiscriminator = reader.GetString();
                        break;
                    case "Axes":
                        int[] axes = JsonSerializer.Deserialize<int[]>(ref reader, options)!;
                        var irAxes = new IRArray<int>(axes);
                        if (typeDiscriminator == "S")
                        {
                            sbpSplit = new SBPSplit(irAxes);
                        }
                        else
                        {
                            throw new InvalidDataException("Axes must be used in SBP split");
                        }

                        break;
                    case "Op":
                        ReduceOp partialOp = JsonSerializer.Deserialize<ReduceOp>(ref reader, options);
                        sbpPartial = new SBPPartial(partialOp);
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }
        }

        switch (typeDiscriminator)
        {
            case "B":
                return SBP.B;
            case "P":
                return sbpPartial!;
            case "S":
                return sbpSplit!;
            default:
                throw new JsonException($"Unknown '$type' discriminator: {typeDiscriminator}");
        }
    }

    public override void Write(Utf8JsonWriter writer, SBP value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        if (value is SBPBroadCast)
        {
            writer.WriteString("$type", "B");
        }
        else if (value is SBPPartial partialValue)
        {
            writer.WriteString("$type", "P");
            writer.WriteString("Op", partialValue.Op.ToString());
        }
        else if (value is SBPSplit splitValue)
        {
            writer.WriteString("$type", "S");
            writer.WritePropertyName("Axes");
            JsonSerializer.Serialize(writer, splitValue.Axes.ToArray(), options);
        }
        else
        {
            throw new JsonException($"Unknown SBP type: {value.GetType()}");
        }

        writer.WriteEndObject();
    }
}

// public sealed record Placement(Placement.DeviceKind Kind, IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind)
public sealed record Placement(IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind = HierarchyKind.Parallel)
{
    // public enum DeviceKind : uint
    // {
    //     CPU = 0,
    // }
    public int Rank => Hierarchy.Count;

    public bool HasWarp => Name.Contains('w', StringComparison.Ordinal);

    public override string ToString() => $"[{string.Join(',', Hierarchy.Zip(Name).Select(t => t.Second.ToString() + ':' + t.First.ToString()))}]";
}

public abstract record IndexExpr
{
    public static IndexExpr Var(string name) => new IndexVar(name);

    public static IndexExpr Const(long value) => new IndexConst(value);

    public static IndexExpr Add(params IndexExpr[] terms) => new IndexAdd(terms);

    public static IndexExpr Mul(params IndexExpr[] factors) => new IndexMul(factors);

    public static IndexExpr FloorDiv(IndexExpr value, IndexExpr divisor) => new IndexFloorDiv(value, divisor);

    public static IndexExpr Mod(IndexExpr value, IndexExpr divisor) => new IndexMod(value, divisor);
}

public sealed record IndexVar(string Name) : IndexExpr
{
    public override string ToString() => Name;
}

public sealed record IndexConst(long Value) : IndexExpr
{
    public override string ToString() => Value.ToString(CultureInfo.InvariantCulture);
}

public sealed record IndexAny : IndexExpr
{
    public static readonly IndexAny Instance = new();

    public override string ToString() => "any";
}

public sealed record IndexNamedPrimitive(string Name, IRArray<IndexExpr> Arguments) : IndexExpr
{
    public override string ToString() => $"{Name}({string.Join(",", Arguments)})";
}

public sealed record IndexAdd(IRArray<IndexExpr> Terms) : IndexExpr
{
    public override string ToString() => string.Join("+", Terms.Select(FormatTerm));

    private static string FormatTerm(IndexExpr expr) => expr is IndexAdd ? $"({expr})" : expr.ToString();
}

public sealed record IndexMul(IRArray<IndexExpr> Factors) : IndexExpr
{
    public override string ToString() => string.Join("*", Factors.Select(FormatFactor));

    private static string FormatFactor(IndexExpr expr) => expr is IndexAdd ? $"({expr})" : expr.ToString();
}

public sealed record IndexFloorDiv(IndexExpr Value, IndexExpr Divisor) : IndexExpr
{
    public override string ToString() => $"floor({Value}/{Divisor})";
}

public sealed record IndexMod(IndexExpr Value, IndexExpr Divisor) : IndexExpr
{
    public override string ToString() => $"{Format(Value)}%{Divisor}";

    private static string Format(IndexExpr expr) => expr is IndexVar or IndexConst or IndexFloorDiv ? expr.ToString() : $"({expr})";
}

public sealed record IndexMapBinding(string Name, IndexExpr Expr)
{
    public static IndexMapBinding Any(string name) => new(name, IndexAny.Instance);

    public override string ToString() => $"{Name}={Expr}";
}

public sealed record IndexMapDescriptor(
    string Name,
    IRArray<string> Inputs,
    IRArray<IndexMapBinding> Outputs,
    IRArray<string> InputDomain,
    IRArray<string> OutputDomain,
    string Predicate = "true",
    string? Inverse = null)
{
    public override string ToString()
    {
        var input = string.Join(", ", Inputs);
        var output = string.Join(", ", Outputs);
        return $"{Name}: ({input}) -> ({output}) where {Predicate}";
    }
}

public static class LayoutVerifier
{
    private const long MaxFiniteCompositionPoints = 16384;

    public static void Verify(DistributionLayout layout)
    {
        VerifyMap(layout.GlobalToOwnerLocal, layout.OwnerLocalToGlobal.Name, layout.Kind);
        VerifyMap(layout.OwnerLocalToGlobal, layout.GlobalToOwnerLocal.Name, layout.Kind);
        if (layout.LocalShape.IsUnranked)
        {
            throw new InvalidOperationException($"DistributionLayout {layout.Kind} has unranked local shape.");
        }
    }

    public static void Verify(DistributionLayout distributionLayout, StorageLayout storageLayout)
    {
        Verify(distributionLayout);
        VerifyMap(storageLayout.LogicalToPhysical, storageLayout.LogicalToPhysical.Name, storageLayout.Kind);
        if (storageLayout.LogicalShape != distributionLayout.LocalShape && storageLayout.ViewMap is null)
        {
            throw new InvalidOperationException($"StorageLayout {storageLayout.Kind} logical shape {storageLayout.LogicalShape} does not match DistributionLayout {distributionLayout.Kind} local shape {distributionLayout.LocalShape} and has no view map.");
        }

        if (storageLayout.ViewMap is { } viewMap)
        {
            VerifyMap(viewMap, distributionLayout.GlobalToOwnerLocal.Name, storageLayout.Kind);
        }
    }

    public static void Verify(DistributedType distributedType, string context)
    {
        var distributionLayout = distributedType.DistributionLayout;
        var storageLayout = distributedType.StorageLayout;
        Verify(distributionLayout, storageLayout);
        VerifyOwnerBounds(distributionLayout, distributedType.Placement, context);
        if (distributedType.ExplicitDistributionLayout is not null)
        {
            VerifyForwardInverseComposition(distributionLayout, context, requireProof: true);
        }
    }

    public static void Verify(DistributionLayout distributionLayout, StorageLayout storageLayout, Placement placement)
    {
        Verify(distributionLayout, storageLayout);
        VerifyOwnerBounds(distributionLayout, placement, $"layout {distributionLayout.Kind}");
        VerifyForwardInverseComposition(distributionLayout, $"layout {distributionLayout.Kind}", requireProof: true);
    }

    public static void VerifyEquivalentForView(DistributedType input, DistributedType output, string context)
    {
        Verify(input, $"{context} input");
        Verify(output, $"{context} output");

        if (GetViewCompatibilityReason(input, output) is { } reason)
        {
            ThrowViewCompatibility(input, output, context, reason);
        }
    }

    public static void VerifyEquivalentForBitcast(DistributedType input, DistributedType output, string context)
    {
        VerifyBitcastLayout(input, output, input, context, "input");

        if (input.TensorType.DType.SizeInBytes != output.TensorType.DType.SizeInBytes)
        {
            ThrowBitcastCompatibility(
                input,
                output,
                context,
                $"element sizes differ: input={input.TensorType.DType}({input.TensorType.DType.SizeInBytes} bytes), output={output.TensorType.DType}({output.TensorType.DType.SizeInBytes} bytes)");
        }

        if (input.TensorType.Shape != output.TensorType.Shape)
        {
            ThrowBitcastCompatibility(input, output, context, $"logical shapes differ: input={input.TensorType.Shape}, output={output.TensorType.Shape}");
        }

        VerifyBitcastLayout(input, output, output, context, "output");

        if (GetViewCompatibilityReason(input, output) is { } reason)
        {
            ThrowBitcastCompatibility(input, output, context, reason);
        }
    }

    public static void VerifyEquivalentForD2DTransfer(DistributedType input, DistributedType output, string context)
    {
        VerifyD2DTransferLayout(input, output, input, context, "input");
        VerifyD2DTransferLayout(input, output, output, context, "output");

        if (input.TensorType != output.TensorType)
        {
            ThrowD2DTransferCompatibility(input, output, context, $"tensor types differ: input={input.TensorType}, output={output.TensorType}");
        }

        if (GetViewCompatibilityReason(input, output) is { } reason)
        {
            ThrowD2DTransferCompatibility(input, output, context, reason);
        }
    }

    public static void VerifyEquivalentToLegacyAxisPolicies(DistributedType distributedType, string context)
    {
        if (!distributedType.HasExplicitLayout)
        {
            return;
        }

        var legacyType = new DistributedType(
            distributedType.TensorType,
            distributedType.AxisPolicies,
            distributedType.Placement,
            distributedType.Partial);
        VerifyEquivalentForView(distributedType, legacyType, $"{context} legacy AxisPolicies compatibility");
    }

    private static void VerifyMap(IndexMapDescriptor map, string expectedInverse, string layoutKind)
    {
        if (map.Outputs.Count == 0)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} has no outputs.");
        }

        if (map.Inverse != expectedInverse)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} must declare inverse {expectedInverse}, but got {map.Inverse ?? "<none>"}.");
        }

        if (map.InputDomain.Count == 0 || map.OutputDomain.Count == 0)
        {
            throw new InvalidOperationException($"Layout {layoutKind} map {map.Name} must declare input and output domains.");
        }

        foreach (var output in map.Outputs)
        {
            VerifyExpr(output.Expr, layoutKind, map.Name);
        }
    }

    private static void VerifyExpr(IndexExpr expr, string layoutKind, string mapName)
    {
        switch (expr)
        {
            case IndexVar:
            case IndexConst:
            case IndexAny:
                return;
            case IndexAdd add:
                foreach (var term in add.Terms)
                {
                    VerifyExpr(term, layoutKind, mapName);
                }

                return;
            case IndexMul mul:
                foreach (var factor in mul.Factors)
                {
                    VerifyExpr(factor, layoutKind, mapName);
                }

                return;
            case IndexFloorDiv floorDiv:
                VerifyExpr(floorDiv.Value, layoutKind, mapName);
                VerifyExpr(floorDiv.Divisor, layoutKind, mapName);
                return;
            case IndexMod mod:
                VerifyExpr(mod.Value, layoutKind, mapName);
                VerifyExpr(mod.Divisor, layoutKind, mapName);
                return;
            case IndexNamedPrimitive named:
                throw new NotSupportedException($"Layout {layoutKind} map {mapName} uses unsupported named primitive {named.Name}; add forward, inverse, domain, cost, and codegen lowering before using it.");
            default:
                throw new NotSupportedException($"Layout {layoutKind} map {mapName} uses unsupported index expression {expr.GetType().Name}.");
        }
    }

    private static void VerifyOwnerBounds(DistributionLayout layout, Placement placement, string context)
    {
        var forwardOutputBounds = ParseDomainBounds(layout.GlobalToOwnerLocal.OutputDomain, $"{context} map {layout.GlobalToOwnerLocal.Name} output domain", failOnUnsupported: false);
        var inverseInputBounds = ParseDomainBounds(layout.OwnerLocalToGlobal.InputDomain, $"{context} map {layout.OwnerLocalToGlobal.Name} input domain", failOnUnsupported: false);
        foreach (var owner in layout.GlobalToOwnerLocal.Outputs.Where(output => output.Name.StartsWith("owner", StringComparison.Ordinal)))
        {
            var ownerAxisText = owner.Name["owner".Length..];
            if (!int.TryParse(ownerAxisText, NumberStyles.None, CultureInfo.InvariantCulture, out var axis))
            {
                throw new InvalidOperationException($"{context}: layout {layout.Kind} owner coordinate {owner.Name} must use owner<axis> naming.");
            }

            if (axis < 0 || axis >= placement.Rank)
            {
                throw new InvalidOperationException($"{context}: layout {layout.Kind} owner coordinate {owner.Name} is outside placement rank {placement.Rank} for placement {placement}.");
            }

            VerifyExactBound(forwardOutputBounds, owner.Name, 0, placement.Hierarchy[axis], $"{context}: layout {layout.Kind} map {layout.GlobalToOwnerLocal.Name} output domain");
            VerifyExactBound(inverseInputBounds, owner.Name, 0, placement.Hierarchy[axis], $"{context}: layout {layout.Kind} map {layout.OwnerLocalToGlobal.Name} input domain");

            if (!layout.OwnerLocalToGlobal.Inputs.Contains(owner.Name))
            {
                throw new InvalidOperationException($"{context}: layout {layout.Kind} inverse map {layout.OwnerLocalToGlobal.Name} must consume owner coordinate {owner.Name}.");
            }
        }
    }

    private static void VerifyForwardInverseComposition(DistributionLayout layout, string context, bool requireProof)
    {
        if (!TryVerifyForwardInverseComposition(layout.GlobalToOwnerLocal, layout.OwnerLocalToGlobal, context, requireProof, out var reason) && requireProof)
        {
            throw new InvalidOperationException($"{context}: layout {layout.Kind} cannot prove finite inverse composition {layout.OwnerLocalToGlobal.Name}({layout.GlobalToOwnerLocal.Name}(.)): {reason}.");
        }

        if (!TryVerifyForwardInverseComposition(layout.OwnerLocalToGlobal, layout.GlobalToOwnerLocal, context, requireProof, out reason) && requireProof)
        {
            throw new InvalidOperationException($"{context}: layout {layout.Kind} cannot prove finite reverse composition {layout.GlobalToOwnerLocal.Name}({layout.OwnerLocalToGlobal.Name}(.)): {reason}.");
        }
    }

    private static bool TryVerifyForwardInverseComposition(
        IndexMapDescriptor forward,
        IndexMapDescriptor inverse,
        string context,
        bool failOnUnsupported,
        out string reason)
    {
        var inputBounds = ParseDomainBounds(forward.InputDomain, $"{context} map {forward.Name} input domain", failOnUnsupported);
        var outputBounds = ParseDomainBounds(forward.OutputDomain, $"{context} map {forward.Name} output domain", failOnUnsupported);
        if (!TryBuildFiniteDomains(forward.Inputs, inputBounds, out var domains, out reason))
        {
            return false;
        }

        var pointCount = domains.Aggregate(1L, (product, domain) => checked(product * global::System.Math.Max(0, domain.MaxExclusive!.Value - domain.Min!.Value)));
        if (pointCount > MaxFiniteCompositionPoints)
        {
            reason = $"input domain has {pointCount} points, exceeding finite proof cap {MaxFiniteCompositionPoints}";
            return false;
        }

        var inverseOutputs = inverse.Outputs.ToDictionary(output => output.Name, output => output.Expr, StringComparer.Ordinal);
        foreach (var inputValues in EnumerateFiniteDomains(forward.Inputs, domains))
        {
            var forwardValues = EvaluateMapOutputs(forward, inputValues, context);
            VerifyConcreteValuesInDomain(forwardValues, outputBounds, $"{context} map {forward.Name} output domain");
            var reconstructed = EvaluateMapOutputs(inverse, forwardValues, context);
            foreach (var input in forward.Inputs)
            {
                if (!inverseOutputs.ContainsKey(input))
                {
                    reason = $"inverse map {inverse.Name} does not reconstruct input {input}";
                    return false;
                }

                if (!reconstructed.TryGetValue(input, out var actual))
                {
                    reason = $"inverse map {inverse.Name} output {input} is not concrete";
                    return false;
                }

                var expected = inputValues[input];
                if (actual != expected)
                {
                    reason = $"input {input}={expected} reconstructs as {actual}";
                    return false;
                }
            }
        }

        reason = string.Empty;
        return true;
    }

    private static Dictionary<string, long> EvaluateMapOutputs(IndexMapDescriptor map, IReadOnlyDictionary<string, long> inputs, string context)
    {
        var values = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var output in map.Outputs)
        {
            if (output.Expr is IndexAny)
            {
                continue;
            }

            values.Add(output.Name, EvaluateIndexExpr(output.Expr, inputs, $"{context} map {map.Name} output {output.Name}"));
        }

        return values;
    }

    private static long EvaluateIndexExpr(IndexExpr expr, IReadOnlyDictionary<string, long> inputs, string context) => expr switch
    {
        IndexVar var when inputs.TryGetValue(var.Name, out var value) => value,
        IndexVar var => throw new NotSupportedException($"{context}: missing concrete value for index variable {var.Name}."),
        IndexConst constant => constant.Value,
        IndexAny => throw new NotSupportedException($"{context}: cannot evaluate unconstrained `any` index expression."),
        IndexAdd add => add.Terms.Aggregate(0L, (sum, term) => checked(sum + EvaluateIndexExpr(term, inputs, context))),
        IndexMul mul => mul.Factors.Aggregate(1L, (product, factor) => checked(product * EvaluateIndexExpr(factor, inputs, context))),
        IndexFloorDiv floorDiv => FloorDiv(EvaluateIndexExpr(floorDiv.Value, inputs, context), EvaluateIndexExpr(floorDiv.Divisor, inputs, context), context),
        IndexMod mod => Mod(EvaluateIndexExpr(mod.Value, inputs, context), EvaluateIndexExpr(mod.Divisor, inputs, context), context),
        IndexNamedPrimitive named => throw new NotSupportedException($"{context}: cannot evaluate named primitive {named.Name}; add a structured evaluator before using this layout."),
        _ => throw new NotSupportedException($"{context}: cannot evaluate index expression {expr.GetType().Name}."),
    };

    private static long FloorDiv(long value, long divisor, string context)
    {
        if (divisor <= 0)
        {
            throw new InvalidOperationException($"{context}: floor division requires a positive divisor, got {divisor}.");
        }

        return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
    }

    private static long Mod(long value, long divisor, string context)
    {
        if (divisor <= 0)
        {
            throw new InvalidOperationException($"{context}: modulo requires a positive divisor, got {divisor}.");
        }

        var result = value % divisor;
        return result < 0 ? result + divisor : result;
    }

    private static bool TryBuildFiniteDomains(
        IRArray<string> inputs,
        IReadOnlyDictionary<string, DomainBounds> bounds,
        out DomainBounds[] domains,
        out string reason)
    {
        domains = new DomainBounds[inputs.Count];
        for (int i = 0; i < inputs.Count; i++)
        {
            var input = inputs[i];
            if (!bounds.TryGetValue(input, out var bound) || bound.Min is null || bound.MaxExclusive is null)
            {
                reason = $"input {input} has no finite constant domain";
                return false;
            }

            if (bound.MaxExclusive.Value < bound.Min.Value)
            {
                reason = $"input {input} has invalid domain [{bound.Min.Value}, {bound.MaxExclusive.Value})";
                return false;
            }

            domains[i] = bound;
        }

        reason = string.Empty;
        return true;
    }

    private static IEnumerable<Dictionary<string, long>> EnumerateFiniteDomains(IRArray<string> inputs, DomainBounds[] domains)
    {
        var values = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var assignment in EnumerateFiniteDomains(inputs, domains, 0, values))
        {
            yield return assignment;
        }
    }

    private static IEnumerable<Dictionary<string, long>> EnumerateFiniteDomains(IRArray<string> inputs, DomainBounds[] domains, int index, Dictionary<string, long> values)
    {
        if (index == inputs.Count)
        {
            yield return new Dictionary<string, long>(values, StringComparer.Ordinal);
            yield break;
        }

        var domain = domains[index];
        for (var value = domain.Min!.Value; value < domain.MaxExclusive!.Value; value++)
        {
            values[inputs[index]] = value;
            foreach (var assignment in EnumerateFiniteDomains(inputs, domains, index + 1, values))
            {
                yield return assignment;
            }
        }
    }

    private static Dictionary<string, DomainBounds> ParseDomainBounds(IRArray<string> domains, string context, bool failOnUnsupported)
    {
        var bounds = new Dictionary<string, DomainBounds>(StringComparer.Ordinal);
        foreach (var domain in domains)
        {
            foreach (var condition in domain.Split("&&", StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
            {
                if (!TryParseDomainCondition(condition, out var name, out var min, out var maxExclusive))
                {
                    if (failOnUnsupported)
                    {
                        throw new NotSupportedException($"{context}: unsupported domain condition `{condition}`. Supported forms are `0<=var<N`, `0<=var`, and `var<N` joined by `&&`.");
                    }

                    continue;
                }

                MergeDomainBound(bounds, name, min, maxExclusive, context);
            }
        }

        return bounds;
    }

    private static bool TryParseDomainCondition(string condition, out string name, out long? min, out long? maxExclusive)
    {
        var text = new string(condition.Where(ch => !char.IsWhiteSpace(ch)).ToArray());
        name = string.Empty;
        min = null;
        maxExclusive = null;
        if (text.StartsWith("0<=", StringComparison.Ordinal))
        {
            var rest = text["0<=".Length..];
            var upperIndex = rest.IndexOf('<', StringComparison.Ordinal);
            if (upperIndex < 0)
            {
                name = rest;
                min = 0;
                return IsValidDomainVariable(name);
            }

            name = rest[..upperIndex];
            if (!long.TryParse(rest[(upperIndex + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out var upper))
            {
                return false;
            }

            min = 0;
            maxExclusive = upper;
            return IsValidDomainVariable(name);
        }

        var lessThanIndex = text.IndexOf('<', StringComparison.Ordinal);
        if (lessThanIndex > 0 && !text.Contains("<=", StringComparison.Ordinal))
        {
            name = text[..lessThanIndex];
            if (!long.TryParse(text[(lessThanIndex + 1)..], NumberStyles.Integer, CultureInfo.InvariantCulture, out var upper))
            {
                return false;
            }

            maxExclusive = upper;
            return IsValidDomainVariable(name);
        }

        return false;
    }

    private static bool IsValidDomainVariable(string name) =>
        name.Length > 0 && name.All(ch => char.IsLetterOrDigit(ch) || ch == '_');

    private static void MergeDomainBound(Dictionary<string, DomainBounds> bounds, string name, long? min, long? maxExclusive, string context)
    {
        bounds.TryGetValue(name, out var existing);
        var merged = new DomainBounds(
            MergeLower(existing.Min, min),
            MergeUpper(existing.MaxExclusive, maxExclusive));
        if (merged is { Min: { } lower, MaxExclusive: { } upper } && lower > upper)
        {
            throw new InvalidOperationException($"{context}: domain for {name} is empty after merging bounds [{lower}, {upper}).");
        }

        bounds[name] = merged;
    }

    private static long? MergeLower(long? lhs, long? rhs) => (lhs, rhs) switch
    {
        (null, null) => null,
        ({ } value, null) => value,
        (null, { } value) => value,
        ({ } left, { } right) => global::System.Math.Max(left, right),
    };

    private static long? MergeUpper(long? lhs, long? rhs) => (lhs, rhs) switch
    {
        (null, null) => null,
        ({ } value, null) => value,
        (null, { } value) => value,
        ({ } left, { } right) => global::System.Math.Min(left, right),
    };

    private static void VerifyExactBound(IReadOnlyDictionary<string, DomainBounds> bounds, string name, long expectedMin, long expectedMaxExclusive, string context)
    {
        if (!bounds.TryGetValue(name, out var bound) || bound.Min != expectedMin || bound.MaxExclusive != expectedMaxExclusive)
        {
            var actual = bounds.TryGetValue(name, out var actualBound)
                ? $"[{actualBound.Min?.ToString(CultureInfo.InvariantCulture) ?? "-inf"}, {actualBound.MaxExclusive?.ToString(CultureInfo.InvariantCulture) ?? "+inf"})"
                : "<missing>";
            throw new InvalidOperationException($"{context}: coordinate {name} must have domain [{expectedMin}, {expectedMaxExclusive}), got {actual}.");
        }
    }

    private static void VerifyConcreteValuesInDomain(IReadOnlyDictionary<string, long> values, IReadOnlyDictionary<string, DomainBounds> bounds, string context)
    {
        foreach (var (name, value) in values)
        {
            if (!bounds.TryGetValue(name, out var bound))
            {
                continue;
            }

            if ((bound.Min is { } min && value < min) || (bound.MaxExclusive is { } maxExclusive && value >= maxExclusive))
            {
                throw new InvalidOperationException($"{context}: coordinate {name} value {value} is outside declared domain [{bound.Min?.ToString(CultureInfo.InvariantCulture) ?? "-inf"}, {bound.MaxExclusive?.ToString(CultureInfo.InvariantCulture) ?? "+inf"}).");
            }
        }
    }

    private static void ThrowViewCompatibility(DistributedType input, DistributedType output, string context, string reason) =>
        throw new NotSupportedException(
            $"{context} requires layout-equivalent distributed view/bitcast operands. Reason: {reason}. " +
            $"Input={FormatDistributedTypeForView(input)}; Output={FormatDistributedTypeForView(output)}");

    private static void ThrowBitcastCompatibility(DistributedType input, DistributedType output, string context, string reason) =>
        throw new NotSupportedException(
            $"{context} cannot preserve explicit distributed layouts across bitcast. Reason: {reason}. " +
            $"InputDType={input.TensorType.DType}, OutputDType={output.TensorType.DType}, " +
            $"InputShape={input.TensorType.Shape}, OutputShape={output.TensorType.Shape}. " +
            $"Input={FormatDistributedTypeForView(input)}; Output={FormatDistributedTypeForView(output)}");

    private static void ThrowD2DTransferCompatibility(DistributedType input, DistributedType output, string context, string reason) =>
        throw new NotSupportedException(
            $"{context} requires explicit-layout distributed-to-distributed transfer proof. Reason: {reason}. " +
            $"InputDType={input.TensorType.DType}, OutputDType={output.TensorType.DType}, " +
            $"InputShape={input.TensorType.Shape}, OutputShape={output.TensorType.Shape}. " +
            $"Input={FormatDistributedTypeForView(input)}; Output={FormatDistributedTypeForView(output)}");

    private static void VerifyBitcastLayout(DistributedType input, DistributedType output, DistributedType type, string context, string role)
    {
        try
        {
            Verify(type, $"{context} {role}");
        }
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
        {
            ThrowBitcastCompatibility(input, output, context, $"{role} layout verification failed: {ex.Message}");
        }
    }

    private static void VerifyD2DTransferLayout(DistributedType input, DistributedType output, DistributedType type, string context, string role)
    {
        try
        {
            Verify(type, $"{context} {role}");
        }
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
        {
            ThrowD2DTransferCompatibility(input, output, context, $"{role} layout verification failed: {ex.Message}");
        }
    }

    private static string? GetViewCompatibilityReason(DistributedType input, DistributedType output)
    {
        if (input.Partial != output.Partial)
        {
            return $"partial flags differ: input={input.Partial}, output={output.Partial}";
        }

        if (!IsSamePlacement(input.Placement, output.Placement))
        {
            return $"placements differ: input={input.Placement}, output={output.Placement}";
        }

        if (!input.AxisPolicies.SequenceEqual(output.AxisPolicies))
        {
            return $"AxisPolicies differ: input=({string.Join(',', input.AxisPolicies)}), output=({string.Join(',', output.AxisPolicies)})";
        }

        if (!IsSameDistributionLayout(input.DistributionLayout, output.DistributionLayout))
        {
            return "explicit distribution layouts are not equivalent";
        }

        if (!IsSameStorageLayout(input.StorageLayout, output.StorageLayout))
        {
            return "explicit storage layouts are not equivalent";
        }

        return null;
    }

    private static string FormatDistributedTypeForView(DistributedType type) =>
        $"Shape={type.TensorType.Shape}, AxisPolicies=({string.Join(',', type.AxisPolicies)}), Placement={type.Placement}, Partial={type.Partial}, " +
        $"Distribution={FormatDistributionLayout(type.DistributionLayout)}, Storage={FormatStorageLayout(type.StorageLayout)}";

    private static string FormatDistributionLayout(DistributionLayout layout) =>
        $"{layout.Kind}, LocalShape={layout.LocalShape}, Valid={layout.ValidPredicate}, Attrs=[{FormatStringArray(layout.Attributes)}], " +
        $"GlobalToOwnerLocal={layout.GlobalToOwnerLocal}, OwnerLocalToGlobal={layout.OwnerLocalToGlobal}";

    private static string FormatStorageLayout(StorageLayout layout) =>
        $"{layout.Kind}, LogicalShape={layout.LogicalShape}, Valid={layout.ValidPredicate}, Attrs=[{FormatStringArray(layout.Attributes)}], " +
        $"LogicalToPhysical={layout.LogicalToPhysical}, ViewMap={layout.ViewMap?.ToString() ?? "<none>"}";

    private static string FormatStringArray(IRArray<string>? values) =>
        values is { Count: > 0 } ? string.Join(", ", values) : string.Empty;

    private static bool IsSamePlacement(Placement lhs, Placement rhs) =>
        lhs.Name == rhs.Name &&
        lhs.HierarchyKind == rhs.HierarchyKind &&
        lhs.Hierarchy.SequenceEqual(rhs.Hierarchy);

    private static bool IsSameDistributionLayout(DistributionLayout lhs, DistributionLayout rhs) =>
        lhs.Kind == rhs.Kind &&
        lhs.LocalShape == rhs.LocalShape &&
        lhs.ValidPredicate == rhs.ValidPredicate &&
        IsSameStringArray(lhs.Attributes, rhs.Attributes) &&
        IsSameIndexMap(lhs.GlobalToOwnerLocal, rhs.GlobalToOwnerLocal) &&
        IsSameIndexMap(lhs.OwnerLocalToGlobal, rhs.OwnerLocalToGlobal);

    private static bool IsSameStorageLayout(StorageLayout lhs, StorageLayout rhs) =>
        lhs.Kind == rhs.Kind &&
        lhs.LogicalShape == rhs.LogicalShape &&
        lhs.ValidPredicate == rhs.ValidPredicate &&
        IsSameStringArray(lhs.Attributes, rhs.Attributes) &&
        IsSameIndexMap(lhs.LogicalToPhysical, rhs.LogicalToPhysical) &&
        IsSameNullableIndexMap(lhs.ViewMap, rhs.ViewMap);

    private static bool IsSameNullableIndexMap(IndexMapDescriptor? lhs, IndexMapDescriptor? rhs)
    {
        if (lhs is null || rhs is null)
        {
            return lhs is null && rhs is null;
        }

        return IsSameIndexMap(lhs, rhs);
    }

    private static bool IsSameIndexMap(IndexMapDescriptor lhs, IndexMapDescriptor rhs) =>
        lhs.Name == rhs.Name &&
        lhs.Predicate == rhs.Predicate &&
        lhs.Inverse == rhs.Inverse &&
        IsSameStringArray(lhs.Inputs, rhs.Inputs) &&
        IsSameStringArray(lhs.InputDomain, rhs.InputDomain) &&
        IsSameStringArray(lhs.OutputDomain, rhs.OutputDomain) &&
        lhs.Outputs.Count == rhs.Outputs.Count &&
        lhs.Outputs.ToArray().Zip(rhs.Outputs.ToArray()).All(pair =>
            pair.First.Name == pair.Second.Name && IsSameIndexExpr(pair.First.Expr, pair.Second.Expr));

    private static bool IsSameStringArray(IRArray<string>? lhs, IRArray<string>? rhs)
    {
        if (lhs is null || rhs is null)
        {
            return lhs is null && rhs is null;
        }

        return lhs.Value.SequenceEqual(rhs.Value);
    }

    private static bool IsSameIndexExpr(IndexExpr lhs, IndexExpr rhs)
    {
        if (lhs.GetType() != rhs.GetType())
        {
            return false;
        }

        return (lhs, rhs) switch
        {
            (IndexVar l, IndexVar r) => l.Name == r.Name,
            (IndexConst l, IndexConst r) => l.Value == r.Value,
            (IndexAny, IndexAny) => true,
            (IndexAdd l, IndexAdd r) => l.Terms.Count == r.Terms.Count &&
                l.Terms.ToArray().Zip(r.Terms.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            (IndexMul l, IndexMul r) => l.Factors.Count == r.Factors.Count &&
                l.Factors.ToArray().Zip(r.Factors.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            (IndexFloorDiv l, IndexFloorDiv r) => IsSameIndexExpr(l.Value, r.Value) && IsSameIndexExpr(l.Divisor, r.Divisor),
            (IndexMod l, IndexMod r) => IsSameIndexExpr(l.Value, r.Value) && IsSameIndexExpr(l.Divisor, r.Divisor),
            (IndexNamedPrimitive l, IndexNamedPrimitive r) => l.Name == r.Name &&
                l.Arguments.Count == r.Arguments.Count &&
                l.Arguments.ToArray().Zip(r.Arguments.ToArray()).All(pair => IsSameIndexExpr(pair.First, pair.Second)),
            _ => false,
        };
    }

    private readonly record struct DomainBounds(long? Min, long? MaxExclusive);
}

public sealed record DistributionLayout(
    string Kind,
    IndexMapDescriptor GlobalToOwnerLocal,
    IndexMapDescriptor OwnerLocalToGlobal,
    Shape LocalShape,
    string ValidPredicate = "true",
    IRArray<string>? Attributes = null)
{
    public static DistributionLayout FromAxisPolicies(TensorType tensorType, IRArray<SBP> axisPolicies, Placement placement)
    {
        if (axisPolicies.Count != tensorType.Shape.Rank)
        {
            throw new InvalidOperationException($"Axis policy rank {axisPolicies.Count} does not match tensor rank {tensorType.Shape.Rank} for shape {tensorType.Shape}.");
        }

        if (!DistributedUtility.IsDistributable(tensorType, axisPolicies.ToArray(), placement))
        {
            throw new InvalidOperationException($"Axis policies ({string.Join(',', axisPolicies)}) are not distributable for shape {tensorType.Shape} and placement {placement}.");
        }

        if (!DistributedUtility.TryGetDividedTensorType(new DistributedType(tensorType, axisPolicies, placement), out var localTensorType))
        {
            throw new InvalidOperationException($"Failed to derive local tensor type for shape {tensorType.Shape}, policies ({string.Join(',', axisPolicies)}), placement {placement}.");
        }

        var rank = tensorType.Shape.Rank;
        var globalInputs = Enumerable.Range(0, rank).Select(i => $"g{i}").ToArray();
        var ownerLocalOutputs = BuildSbpOwnerLocalOutputs(axisPolicies, placement, localTensorType.Shape, rank);
        var ownerInputs = Enumerable.Range(0, placement.Rank).Select(i => $"owner{i}").ToArray();
        var localInputs = Enumerable.Range(0, rank).Select(i => $"l{i}").ToArray();
        var globalOutputs = BuildSbpGlobalOutputs(axisPolicies, placement, localTensorType.Shape, rank);

        return new DistributionLayout(
            "SBP",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                globalInputs,
                ownerLocalOutputs,
                BuildShapeDomain("g", tensorType.Shape, rank),
                BuildOwnerLocalDomain(placement, localTensorType.Shape, rank),
                BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
                "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ownerInputs.Concat(localInputs).ToArray(),
                globalOutputs,
                BuildOwnerLocalDomain(placement, localTensorType.Shape, rank),
                BuildShapeDomain("g", tensorType.Shape, rank),
                BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
                "GlobalToOwnerLocal"),
            localTensorType.Shape,
            BuildSbpValidPredicate(placement, localTensorType.Shape, rank),
            axisPolicies.Select((sbp, i) => $"axis{i}:{sbp}").ToArray());
    }

    public static DistributionLayout TritonBlocked(Shape tensorShape, TritonBlockedLayout layout)
    {
        if (tensorShape.Rank != 1)
        {
            throw new NotSupportedException($"TritonBlocked DistributionLayout currently supports rank-1 tensors only, but got shape {tensorShape}.");
        }

        layout.Validate();
        var threadsPerCTA = layout.ThreadsPerWarp * layout.WarpsPerCTA;
        var elementsPerCTA = layout.SizePerThread * threadsPerCTA;
        var localShape = new RankedShape(layout.SizePerThread);
        var ctaDomain = tensorShape[0].IsFixed
            ? $"0<=cta<{CeilDiv(tensorShape[0].FixedValue, elementsPerCTA)}"
            : "0<=cta";
        var ownerLocalDomain = $"{ctaDomain} && 0<=warp<{layout.WarpsPerCTA} && 0<=lane<{layout.ThreadsPerWarp} && 0<=elem<{layout.SizePerThread}";
        return new DistributionLayout(
            "TritonBlocked",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"g{i}").ToArray(),
                BuildTritonBlockedOwnerLocalOutputs(layout, threadsPerCTA, elementsPerCTA),
                BuildShapeDomain("g", tensorShape, tensorShape.Rank),
                [ownerLocalDomain],
                $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
                "OwnerLocalToGlobal"),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["cta", "warp", "lane", "elem"],
                BuildTritonBlockedGlobalOutputs(layout, threadsPerCTA, elementsPerCTA),
                [ownerLocalDomain],
                BuildShapeDomain("g", tensorShape, tensorShape.Rank),
                $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
                "GlobalToOwnerLocal"),
            localShape,
            $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
            layout.ToAttributes());
    }

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LocalShape={LocalShape}, Valid={ValidPredicate}{attrs}";
    }

    private static IndexMapBinding[] BuildSbpOwnerLocalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var ownerOutputs = new IndexMapBinding[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            ownerOutputs[i] = IndexMapBinding.Any($"owner{i}");
        }

        var localOutputs = new List<IndexMapBinding>();
        for (int dim = 0; dim < rank; dim++)
        {
            var globalIndex = IndexExpr.Var($"g{dim}");
            if (axisPolicies[dim] is not SBPSplit split)
            {
                localOutputs.Add(new IndexMapBinding($"l{dim}", globalIndex));
            }
            else
            {
                var localExtent = DimensionExpr(localShape[dim]);
                localOutputs.Add(new IndexMapBinding($"l{dim}", IndexExpr.Mod(globalIndex, localExtent)));

                var stride = 1;
                foreach (var axis in split.Axes)
                {
                    ValidatePlacementAxis(axis, placement);
                    ownerOutputs[axis] = new IndexMapBinding(
                        $"owner{axis}",
                        IndexExpr.Mod(
                            IndexExpr.FloorDiv(globalIndex, Product(localExtent, stride)),
                            IndexExpr.Const(placement.Hierarchy[axis])));
                    stride *= placement.Hierarchy[axis];
                }
            }
        }

        return ownerOutputs.Concat(localOutputs).ToArray();
    }

    private static IndexMapBinding[] BuildSbpGlobalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var outputs = new List<IndexMapBinding>();
        for (int dim = 0; dim < rank; dim++)
        {
            var localIndex = IndexExpr.Var($"l{dim}");
            if (axisPolicies[dim] is not SBPSplit split)
            {
                outputs.Add(new IndexMapBinding($"g{dim}", localIndex));
            }
            else
            {
                var localExtent = DimensionExpr(localShape[dim]);
                var ownerLinear = BuildOwnerLinearExpression(split, placement);
                outputs.Add(new IndexMapBinding($"g{dim}", IndexExpr.Add(IndexExpr.Mul(ownerLinear, localExtent), localIndex)));
            }
        }

        return outputs.ToArray();
    }

    private static string BuildSbpValidPredicate(Placement placement, Shape localShape, int rank)
    {
        var predicates = new List<string>();
        for (int axis = 0; axis < placement.Rank; axis++)
        {
            predicates.Add($"0<=owner{axis}<{placement.Hierarchy[axis]}");
        }

        for (int dim = 0; dim < rank; dim++)
        {
            predicates.Add($"0<=l{dim}<{FormatDimension(localShape[dim])}");
        }

        return string.Join(" && ", predicates);
    }

    private static IndexExpr BuildOwnerLinearExpression(SBPSplit split, Placement placement)
    {
        var stride = 1;
        var terms = new List<IndexExpr>();
        foreach (var axis in split.Axes)
        {
            ValidatePlacementAxis(axis, placement);
            var owner = IndexExpr.Var($"owner{axis}");
            terms.Add(stride == 1 ? owner : IndexExpr.Mul(owner, IndexExpr.Const(stride)));
            stride *= placement.Hierarchy[axis];
        }

        return terms.Count == 1 ? terms[0] : IndexExpr.Add(terms.ToArray());
    }

    private static IndexExpr Product(IndexExpr extent, int multiplier) => multiplier == 1 ? extent : IndexExpr.Mul(extent, IndexExpr.Const(multiplier));

    private static IndexExpr DimensionExpr(Dimension dimension) => dimension.IsFixed ? IndexExpr.Const(dimension.FixedValue) : IndexExpr.Var(FormatDimension(dimension));

    private static string[] BuildShapeDomain(string prefix, Shape shape, int rank) =>
        Enumerable.Range(0, rank).Select(i => $"0<={prefix}{i}<{FormatDimension(shape[i])}").ToArray();

    private static string[] BuildOwnerLocalDomain(Placement placement, Shape localShape, int rank) =>
        Enumerable.Range(0, placement.Rank).Select(axis => $"0<=owner{axis}<{placement.Hierarchy[axis]}")
            .Concat(Enumerable.Range(0, rank).Select(dim => $"0<=l{dim}<{FormatDimension(localShape[dim])}"))
            .ToArray();

    private static long CeilDiv(long value, long divisor)
    {
        if (divisor <= 0)
        {
            throw new InvalidOperationException($"CeilDiv requires a positive divisor, got {divisor}.");
        }

        return (value + divisor - 1) / divisor;
    }

    private static string FormatDimension(Dimension dimension) => dimension.IsFixed ? dimension.FixedValue.ToString(CultureInfo.InvariantCulture) : dimension.ToString();

    private static void ValidatePlacementAxis(int axis, Placement placement)
    {
        if (axis < 0 || axis >= placement.Rank)
        {
            throw new InvalidOperationException($"Split axis {axis} is outside placement rank {placement.Rank} for placement {placement}.");
        }

        if (placement.Hierarchy[axis] <= 0)
        {
            throw new InvalidOperationException($"Placement axis {axis} has invalid hierarchy {placement.Hierarchy[axis]} for placement {placement}.");
        }
    }

    private static IndexMapBinding[] BuildTritonBlockedOwnerLocalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA)
    {
        var g0 = IndexExpr.Var("g0");
        var elementInCta = IndexExpr.Mod(g0, IndexExpr.Const(elementsPerCTA));
        return layout.ThreadElementOrder switch
        {
            TritonThreadElementOrder.Contiguous =>
            [
                new("cta", IndexExpr.FloorDiv(g0, IndexExpr.Const(elementsPerCTA))),
                new("warp", IndexExpr.FloorDiv(elementInCta, IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp))),
                new("lane", IndexExpr.FloorDiv(IndexExpr.Mod(g0, IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp)), IndexExpr.Const(layout.SizePerThread))),
                new("elem", IndexExpr.Mod(g0, IndexExpr.Const(layout.SizePerThread))),
            ],
            TritonThreadElementOrder.Strided =>
            [
                new("cta", IndexExpr.FloorDiv(g0, IndexExpr.Const(elementsPerCTA))),
                new("warp", IndexExpr.FloorDiv(IndexExpr.Mod(g0, IndexExpr.Const(threadsPerCTA)), IndexExpr.Const(layout.ThreadsPerWarp))),
                new("lane", IndexExpr.Mod(IndexExpr.Mod(g0, IndexExpr.Const(threadsPerCTA)), IndexExpr.Const(layout.ThreadsPerWarp))),
                new("elem", IndexExpr.FloorDiv(elementInCta, IndexExpr.Const(threadsPerCTA))),
            ],
            _ => throw new NotSupportedException($"Unsupported Triton thread element order {layout.ThreadElementOrder}."),
        };
    }

    private static IndexMapBinding[] BuildTritonBlockedGlobalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA) => layout.ThreadElementOrder switch
    {
        TritonThreadElementOrder.Contiguous =>
        [
            new(
                "g0",
                IndexExpr.Add(
                    IndexExpr.Mul(IndexExpr.Var("cta"), IndexExpr.Const(elementsPerCTA)),
                    IndexExpr.Mul(IndexExpr.Var("warp"), IndexExpr.Const(layout.SizePerThread * layout.ThreadsPerWarp)),
                    IndexExpr.Mul(IndexExpr.Var("lane"), IndexExpr.Const(layout.SizePerThread)),
                    IndexExpr.Var("elem"))),
        ],
        TritonThreadElementOrder.Strided =>
        [
            new(
                "g0",
                IndexExpr.Add(
                    IndexExpr.Mul(IndexExpr.Var("cta"), IndexExpr.Const(elementsPerCTA)),
                    IndexExpr.Mul(IndexExpr.Var("warp"), IndexExpr.Const(layout.ThreadsPerWarp)),
                    IndexExpr.Var("lane"),
                    IndexExpr.Mul(IndexExpr.Var("elem"), IndexExpr.Const(threadsPerCTA)))),
        ],
        _ => throw new NotSupportedException($"Unsupported Triton thread element order {layout.ThreadElementOrder}."),
    };
}

public sealed record TritonBlockedLayout(
    int SizePerThread,
    int ThreadsPerWarp,
    int WarpsPerCTA,
    IRArray<int> Order,
    IRArray<int> CTAsPerCGA,
    IRArray<int> CTASplitNum,
    IRArray<int> CTAOrder,
    TritonThreadElementOrder ThreadElementOrder = TritonThreadElementOrder.Contiguous)
{
    public IRArray<string> ToAttributes() =>
    [
        $"sizePerThread={SizePerThread}",
        $"threadsPerWarp={ThreadsPerWarp}",
        $"warpsPerCTA={WarpsPerCTA}",
        $"threadElementOrder={ThreadElementOrder}",
        $"order=[{string.Join(",", Order)}]",
        $"ctasPerCGA=[{string.Join(",", CTAsPerCGA)}]",
        $"ctaSplitNum=[{string.Join(",", CTASplitNum)}]",
        $"ctaOrder=[{string.Join(",", CTAOrder)}]",
    ];

    public void Validate()
    {
        if (SizePerThread <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout SizePerThread must be positive, got {SizePerThread}.");
        }

        if (ThreadsPerWarp <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout ThreadsPerWarp must be positive, got {ThreadsPerWarp}.");
        }

        if (WarpsPerCTA <= 0)
        {
            throw new InvalidOperationException($"Triton blocked layout WarpsPerCTA must be positive, got {WarpsPerCTA}.");
        }
    }
}

public sealed record StorageLayout(
    string Kind,
    Shape LogicalShape,
    IndexMapDescriptor LogicalToPhysical,
    string ValidPredicate = "true",
    IRArray<string>? Attributes = null,
    IndexMapDescriptor? ViewMap = null)
{
    public static StorageLayout Identity(Shape localShape) =>
        new(
            "Identity",
            localShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                Enumerable.Range(0, localShape.Rank).Select(i => $"l{i}").ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => new IndexMapBinding($"p{i}", IndexExpr.Var($"l{i}"))).ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => $"0<=l{i}<{FormatDimension(localShape[i])}").ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => $"0<=p{i}<{FormatDimension(localShape[i])}").ToArray(),
                "true",
                "LogicalToPhysical"));

    public static StorageLayout SharedBlock(Shape tensorShape, DistributionLayout distributionLayout)
    {
        if (tensorShape.IsUnranked)
        {
            throw new InvalidOperationException("SharedBlock storage layout requires a ranked tensor shape.");
        }

        return new(
            "SharedBlock",
            tensorShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"g{i}").ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => new IndexMapBinding($"p{i}", IndexExpr.Var($"g{i}"))).ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"0<=g{i}<{FormatDimension(tensorShape[i])}").ToArray(),
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"0<=p{i}<{FormatDimension(tensorShape[i])}").ToArray(),
                "true",
                "LogicalToPhysical"),
            distributionLayout.ValidPredicate,
            ["scope:block_local"],
            distributionLayout.OwnerLocalToGlobal);
    }

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LogicalShape={LogicalShape}, Valid={ValidPredicate}{attrs}";
    }

    private static string FormatDimension(Dimension dimension) => dimension.IsFixed ? dimension.FixedValue.ToString(CultureInfo.InvariantCulture) : dimension.ToString();
}

public sealed record DistributedType(
    TensorType TensorType,
    IRArray<SBP> AxisPolicies,
    Placement Placement,
    bool Partial = false,
    DistributionLayout? ExplicitDistributionLayout = null,
    StorageLayout? ExplicitStorageLayout = null) : IRType
{
    public DistributionLayout DistributionLayout => ExplicitDistributionLayout ?? DistributionLayout.FromAxisPolicies(TensorType, AxisPolicies, Placement);

    public StorageLayout StorageLayout => ExplicitStorageLayout ?? StorageLayout.Identity(DistributionLayout.LocalShape);

    public bool HasExplicitLayout => ExplicitDistributionLayout is not null || ExplicitStorageLayout is not null;

    public static DistributedType FromLayouts(
        TensorType tensorType,
        Placement placement,
        DistributionLayout distributionLayout,
        StorageLayout? storageLayout = null,
        IRArray<SBP>? axisPolicies = null,
        bool partial = false)
    {
        storageLayout ??= StorageLayout.Identity(distributionLayout.LocalShape);
        LayoutVerifier.Verify(distributionLayout, storageLayout, placement);
        return new DistributedType(
            tensorType,
            axisPolicies ?? Enumerable.Range(0, tensorType.Shape.Rank).Select(_ => SBP.B).ToArray(),
            placement,
            partial,
            distributionLayout,
            storageLayout);
    }

    public override string ToString() => $"{TensorType}, ({string.Join(',', AxisPolicies)}), {Placement}, Layout: {DistributionLayout.Kind}, Storage: {StorageLayout.Kind}, Partial: {Partial}";
}
