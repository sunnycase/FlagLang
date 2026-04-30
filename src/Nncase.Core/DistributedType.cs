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

public sealed record IndexMapDescriptor(string Name, IRArray<string> Inputs, IRArray<string> Outputs, string Predicate = "true")
{
    public override string ToString()
    {
        var input = string.Join(", ", Inputs);
        var output = string.Join(", ", Outputs);
        return $"{Name}: ({input}) -> ({output}) where {Predicate}";
    }
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
            new IndexMapDescriptor("GlobalToOwnerLocal", globalInputs, ownerLocalOutputs),
            new IndexMapDescriptor("OwnerLocalToGlobal", ownerInputs.Concat(localInputs).ToArray(), globalOutputs),
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
        var localShape = new RankedShape(elementsPerCTA);
        return new DistributionLayout(
            "TritonBlocked",
            new IndexMapDescriptor(
                "GlobalToOwnerLocal",
                Enumerable.Range(0, tensorShape.Rank).Select(i => $"g{i}").ToArray(),
                BuildTritonBlockedOwnerLocalOutputs(layout, threadsPerCTA, elementsPerCTA)),
            new IndexMapDescriptor(
                "OwnerLocalToGlobal",
                ["cta", "warp", "lane", "elem"],
                BuildTritonBlockedGlobalOutputs(layout, threadsPerCTA, elementsPerCTA)),
            localShape,
            $"0<=lane<{layout.ThreadsPerWarp} && 0<=warp<{layout.WarpsPerCTA} && 0<=elem<{layout.SizePerThread}",
            layout.ToAttributes());
    }

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LocalShape={LocalShape}, Valid={ValidPredicate}{attrs}";
    }

    private static string[] BuildSbpOwnerLocalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var ownerOutputs = new string[placement.Rank];
        for (int i = 0; i < placement.Rank; i++)
        {
            ownerOutputs[i] = $"owner{i}=any";
        }

        var localOutputs = new List<string>();
        for (int dim = 0; dim < rank; dim++)
        {
            if (axisPolicies[dim] is not SBPSplit split)
            {
                localOutputs.Add($"l{dim}=g{dim}");
            }
            else
            {
                var localExtent = FormatDimension(localShape[dim]);
                localOutputs.Add($"l{dim}=g{dim}%{localExtent}");

                var stride = 1;
                foreach (var axis in split.Axes)
                {
                    ValidatePlacementAxis(axis, placement);
                    ownerOutputs[axis] = $"owner{axis}=floor(g{dim}/{FormatProduct(localExtent, stride)})%{placement.Hierarchy[axis]}";
                    stride *= placement.Hierarchy[axis];
                }
            }
        }

        return ownerOutputs.Concat(localOutputs).ToArray();
    }

    private static string[] BuildSbpGlobalOutputs(IRArray<SBP> axisPolicies, Placement placement, Shape localShape, int rank)
    {
        var outputs = new List<string>();
        for (int dim = 0; dim < rank; dim++)
        {
            if (axisPolicies[dim] is not SBPSplit split)
            {
                outputs.Add($"g{dim}=l{dim}");
            }
            else
            {
                var localExtent = FormatDimension(localShape[dim]);
                var ownerLinear = BuildOwnerLinearExpression(split, placement);
                outputs.Add($"g{dim}={FormatOwnerLocalProduct(ownerLinear, localExtent)}+l{dim}");
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

    private static string BuildOwnerLinearExpression(SBPSplit split, Placement placement)
    {
        var stride = 1;
        var terms = new List<string>();
        foreach (var axis in split.Axes)
        {
            ValidatePlacementAxis(axis, placement);
            terms.Add(stride == 1 ? $"owner{axis}" : $"owner{axis}*{stride}");
            stride *= placement.Hierarchy[axis];
        }

        return string.Join("+", terms);
    }

    private static string FormatOwnerLocalProduct(string ownerLinear, string localExtent) =>
        ownerLinear.Contains('+', StringComparison.Ordinal) ? $"({ownerLinear})*{localExtent}" : $"{ownerLinear}*{localExtent}";

    private static string FormatProduct(string extent, int multiplier) => multiplier == 1 ? extent : $"{extent}*{multiplier}";

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

    private static string[] BuildTritonBlockedOwnerLocalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA) => layout.ThreadElementOrder switch
    {
        TritonThreadElementOrder.Contiguous =>
        [
            $"cta=floor(g0/{elementsPerCTA})",
            $"warp=floor((g0%{elementsPerCTA})/{layout.SizePerThread * layout.ThreadsPerWarp})",
            $"lane=floor((g0%{layout.SizePerThread * layout.ThreadsPerWarp})/{layout.SizePerThread})",
            $"elem=g0%{layout.SizePerThread}",
        ],
        TritonThreadElementOrder.Strided =>
        [
            $"cta=floor(g0/{elementsPerCTA})",
            $"warp=floor((g0%{threadsPerCTA})/{layout.ThreadsPerWarp})",
            $"lane=(g0%{threadsPerCTA})%{layout.ThreadsPerWarp}",
            $"elem=floor((g0%{elementsPerCTA})/{threadsPerCTA})",
        ],
        _ => throw new NotSupportedException($"Unsupported Triton thread element order {layout.ThreadElementOrder}."),
    };

    private static string[] BuildTritonBlockedGlobalOutputs(TritonBlockedLayout layout, int threadsPerCTA, int elementsPerCTA) => layout.ThreadElementOrder switch
    {
        TritonThreadElementOrder.Contiguous =>
        [
            $"g0=cta*{elementsPerCTA}+warp*{layout.SizePerThread * layout.ThreadsPerWarp}+lane*{layout.SizePerThread}+elem",
        ],
        TritonThreadElementOrder.Strided =>
        [
            $"g0=cta*{elementsPerCTA}+warp*{layout.ThreadsPerWarp}+lane+elem*{threadsPerCTA}",
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
    IRArray<string>? Attributes = null)
{
    public static StorageLayout Identity(Shape localShape) =>
        new(
            "Identity",
            localShape,
            new IndexMapDescriptor(
                "LogicalToPhysical",
                Enumerable.Range(0, localShape.Rank).Select(i => $"l{i}").ToArray(),
                Enumerable.Range(0, localShape.Rank).Select(i => $"p{i}=l{i}").ToArray()));

    public override string ToString()
    {
        var attrs = Attributes is { Count: > 0 } ? $", Attrs=[{string.Join(", ", Attributes)}]" : string.Empty;
        return $"{Kind}, LogicalShape={LogicalShape}, Valid={ValidPredicate}{attrs}";
    }
}

public sealed record DistributedType(TensorType TensorType, IRArray<SBP> AxisPolicies, Placement Placement, bool Partial = false) : IRType
{
    public DistributionLayout DistributionLayout => DistributionLayout.FromAxisPolicies(TensorType, AxisPolicies, Placement);

    public StorageLayout StorageLayout => StorageLayout.Identity(DistributionLayout.LocalShape);

    public override string ToString() => $"{TensorType}, ({string.Join(',', AxisPolicies)}), {Placement}, Layout: {DistributionLayout.Kind}, Storage: {StorageLayout.Kind}, Partial: {Partial}";
}
