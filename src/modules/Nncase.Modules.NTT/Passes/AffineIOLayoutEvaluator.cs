// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Nncase.IR;
using Nncase.IR.Distributed;
using Nncase.IR.Shapes;
using Nncase.TIR;

namespace Nncase.Passes;

internal static class AffineIOLayoutEvaluator
{
    public static Expr[] GetStorageIndices(TIR.Buffer buffer, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> domainValues)
    {
        if (buffer.DistributedType is not DistributedType distributedType)
        {
            return loopVars.AsExprs();
        }

        ValidateDistributedBuffer(buffer, distributedType);
        var storageLayout = distributedType.StorageLayout;
        if (IsIdentityStorageMap(storageLayout.LogicalToPhysical, loopVars.Count))
        {
            return loopVars.AsExprs();
        }

        var bindings = BuildOwnerLocalBindings(buffer, distributedType, loopVars);
        if (storageLayout.ViewMap is not null)
        {
            var viewValues = EvaluateMap(storageLayout.ViewMap, bindings, $"storage view map {storageLayout.ViewMap.Name} for buffer {buffer.Name}");
            foreach (var value in viewValues)
            {
                bindings[value.Name] = value.Value;
            }
        }
        else
        {
            for (int i = 0; i < domainValues.Count; i++)
            {
                bindings[$"g{i}"] = domainValues[i];
            }
        }

        return EvaluateMap(storageLayout.LogicalToPhysical, bindings, $"storage map {storageLayout.LogicalToPhysical.Name} for buffer {buffer.Name}")
            .Select(value => value.Value)
            .ToArray()
            .AsExprs();
    }

    public static Dimension[] GetIterationExtents(TIR.Buffer buffer)
    {
        var globalExtents = buffer.Dimensions.ToArray();
        if (buffer.DistributedType is not DistributedType distributedType)
        {
            return globalExtents;
        }

        ValidateDistributedBuffer(buffer, distributedType);
        if (distributedType.ExplicitDistributionLayout is not null)
        {
            return GetExplicitIterationExtents(buffer, distributedType);
        }

        return globalExtents.Select((extent, axis) => GetLocalShardExtent(distributedType, axis, extent)).ToArray();
    }

    public static Dimension[] GetDomainValues(TIR.Buffer buffer, IReadOnlyList<DimVar> loopVars, IReadOnlyList<Dimension> globalExtents)
    {
        if (buffer.DistributedType is not DistributedType distributedType)
        {
            return loopVars.Select<DimVar, Dimension>(loopVar => loopVar).ToArray();
        }

        ValidateDistributedBuffer(buffer, distributedType);
        if (distributedType.ExplicitDistributionLayout is not null)
        {
            return GetExplicitDomainValues(buffer, distributedType, loopVars);
        }

        var domainValues = new Dimension[loopVars.Count];
        for (int axis = 0; axis < loopVars.Count; axis++)
        {
            domainValues[axis] = loopVars[axis] + GetShardOffset(distributedType, axis, globalExtents[axis]);
        }

        return domainValues;
    }

    public static Dimension GetTritonBlockedLocalDomainValue(DistributionLayout distributionLayout, DimVar elem, string context)
    {
        var layout = ParseTritonBlockedLayout(distributionLayout, context);
        if (distributionLayout.LocalShape is not { IsUnranked: false, Rank: 1 } || distributionLayout.LocalShape[0] != layout.SizePerThread)
        {
            throw new NotSupportedException($"{context} requires TritonBlocked LocalShape=[sizePerThread], got LocalShape={distributionLayout.LocalShape}, sizePerThread={layout.SizePerThread}.");
        }

        var threadId = IR.F.Distributed.ThreadId();
        var threadsPerCTA = (Dimension)(layout.ThreadsPerWarp * layout.WarpsPerCTA);
        return layout.ThreadElementOrder switch
        {
            TritonThreadElementOrder.Contiguous => (threadId * layout.SizePerThread) + elem,
            TritonThreadElementOrder.Strided => threadId + (elem * threadsPerCTA),
            _ => throw new NotSupportedException($"{context} does not support TritonBlocked thread element order {layout.ThreadElementOrder}."),
        };
    }

    private static bool IsIdentityStorageMap(IndexMapDescriptor map, int rank)
    {
        if (map.Inputs.Count != rank || map.Outputs.Count != rank)
        {
            return false;
        }

        for (int i = 0; i < rank; i++)
        {
            if (map.Inputs[i] != $"l{i}" ||
                map.Outputs[i].Name != $"p{i}" ||
                map.Outputs[i].Expr is not IndexVar { Name: var name } ||
                name != $"l{i}")
            {
                return false;
            }
        }

        return true;
    }

    private static void ValidateDistributedBuffer(TIR.Buffer buffer, DistributedType distributedType)
    {
        if (distributedType.Partial)
        {
            throw new NotSupportedException("Affine IO layout evaluation cannot directly lower partial distributed buffers. Resolve Partial with a distributed reduction before affine IO lowering.");
        }

        if (distributedType.TensorType.Shape.Rank != buffer.Rank || distributedType.AxisPolicies.Count != buffer.Rank)
        {
            throw new NotSupportedException($"Distributed buffer {buffer.Name} has rank {buffer.Rank}, but its distributed type has tensor rank {distributedType.TensorType.Shape.Rank} and {distributedType.AxisPolicies.Count} axis policies.");
        }

        if (distributedType.ExplicitDistributionLayout is not null)
        {
            LayoutVerifier.Verify(distributedType, $"Affine IO layout evaluation buffer {buffer.Name}");
            if (distributedType.DistributionLayout.LocalShape.Rank != buffer.Rank)
            {
                throw new NotSupportedException($"Distributed buffer {buffer.Name} has rank {buffer.Rank}, but explicit layout {distributedType.DistributionLayout.Kind} has local shape {distributedType.DistributionLayout.LocalShape}.");
            }
        }
    }

    private static Dimension[] GetExplicitIterationExtents(TIR.Buffer buffer, DistributedType distributedType)
    {
        var layout = distributedType.DistributionLayout;
        ValidateSupportedExplicitMap(buffer, layout);
        return layout.LocalShape.ToArray();
    }

    private static Dimension[] GetExplicitDomainValues(TIR.Buffer buffer, DistributedType distributedType, IReadOnlyList<DimVar> loopVars)
    {
        var layout = distributedType.DistributionLayout;
        ValidateSupportedExplicitMap(buffer, layout);
        return EvaluateMap(layout.OwnerLocalToGlobal, BuildOwnerLocalBindings(buffer, distributedType, loopVars), $"distribution map {layout.OwnerLocalToGlobal.Name} for buffer {buffer.Name}")
            .Select(value => value.Value)
            .ToArray();
    }

    private static void ValidateSupportedExplicitMap(TIR.Buffer buffer, DistributionLayout layout)
    {
        if (layout.Kind is not ("SBP" or "TritonBlocked"))
        {
            throw new NotSupportedException($"Affine IO layout evaluation cannot evaluate explicit distribution layout {layout.Kind} for buffer {buffer.Name}. Add a DistributionLayout evaluator for this map instead of falling back to AxisPolicies.");
        }

        if (layout.Kind == "TritonBlocked" && layout.LocalShape is not { IsUnranked: false, Rank: 1 })
        {
            throw new NotSupportedException($"Affine IO layout evaluation supports explicit {layout.Kind} layout only for rank-1 buffers, got {layout.LocalShape} on {buffer.Name}.");
        }
    }

    private static Dimension GetLocalShardExtent(DistributedType distributedType, int tensorAxis, Dimension globalExtent)
    {
        return distributedType.AxisPolicies[tensorAxis] switch
        {
            SBPBroadCast => globalExtent,
            SBPSplit split => GetSplitLocalShardExtent(distributedType, split, globalExtent),
            SBPPartial partial => throw new NotSupportedException($"Affine IO layout evaluation cannot directly lower partial shard policy {partial}. Resolve Partial before affine IO lowering."),
            SBP policy => throw new NotSupportedException($"Unsupported affine IO shard policy {policy.GetType().Name}."),
        };
    }

    private static Dimension GetSplitLocalShardExtent(DistributedType distributedType, SBPSplit split, Dimension globalExtent)
    {
        var maxLocalExtent = Dimension.CeilDiv(globalExtent, GetSplitDivisor(distributedType, split));
        var offset = GetSplitShardOffset(distributedType, split, globalExtent);
        return Dimension.Min(maxLocalExtent, globalExtent - offset);
    }

    private static Dimension GetShardOffset(DistributedType distributedType, int tensorAxis, Dimension globalExtent)
    {
        return distributedType.AxisPolicies[tensorAxis] switch
        {
            SBPBroadCast => Dimension.Zero,
            SBPSplit split => GetSplitShardOffset(distributedType, split, globalExtent),
            SBPPartial partial => throw new NotSupportedException($"Affine IO layout evaluation cannot directly lower partial shard policy {partial}. Resolve Partial before affine IO lowering."),
            SBP policy => throw new NotSupportedException($"Unsupported affine IO shard policy {policy.GetType().Name}."),
        };
    }

    private static Dimension GetSplitShardOffset(DistributedType distributedType, SBPSplit split, Dimension globalExtent)
    {
        var maxLocalExtent = Dimension.CeilDiv(globalExtent, GetSplitDivisor(distributedType, split));
        return Dimension.Min(maxLocalExtent * GetSplitLinearShardIndex(distributedType, split), globalExtent);
    }

    private static Dimension GetSplitDivisor(DistributedType distributedType, SBPSplit split)
    {
        Dimension divisor = Dimension.One;
        foreach (var meshAxis in split.Axes)
        {
            ValidateMeshAxis(distributedType.Placement, meshAxis);
            divisor *= distributedType.Placement.Hierarchy[meshAxis];
        }

        return divisor;
    }

    private static Dimension GetSplitLinearShardIndex(DistributedType distributedType, SBPSplit split)
    {
        Dimension linearIndex = Dimension.Zero;
        foreach (var meshAxis in split.Axes)
        {
            ValidateMeshAxis(distributedType.Placement, meshAxis);
            linearIndex = (linearIndex * distributedType.Placement.Hierarchy[meshAxis]) + GetMeshAxisIndex(distributedType.Placement, meshAxis);
        }

        return linearIndex;
    }

    private static void ValidateMeshAxis(Placement placement, int meshAxis)
    {
        if (meshAxis < 0 || meshAxis >= placement.Rank || meshAxis >= placement.Name.Length)
        {
            throw new NotSupportedException($"Invalid distributed mesh axis {meshAxis} for placement {placement}.");
        }
    }

    private static Dimension GetMeshAxisIndex(Placement placement, int meshAxis)
    {
        return placement.Name[meshAxis] switch
        {
            't' => IR.F.Distributed.ThreadId(),
            'b' => IR.F.Distributed.ProgramId(0),
            var name => throw new NotSupportedException($"Affine IO layout evaluation only supports thread ('t') and block ('b') mesh axes, got '{name}' in placement {placement}."),
        };
    }

    private static TritonBlockedLayout ParseTritonBlockedLayout(DistributionLayout distributionLayout, string context)
    {
        if (distributionLayout.Attributes is null)
        {
            throw new NotSupportedException($"{context} requires TritonBlocked layout attributes.");
        }

        return new TritonBlockedLayout(
            SizePerThread: RequireTritonBlockedIntAttribute(distributionLayout, "sizePerThread"),
            ThreadsPerWarp: RequireTritonBlockedIntAttribute(distributionLayout, "threadsPerWarp"),
            WarpsPerCTA: RequireTritonBlockedIntAttribute(distributionLayout, "warpsPerCTA"),
            Order: RequireTritonBlockedIntListAttribute(distributionLayout, "order"),
            CTAsPerCGA: RequireTritonBlockedIntListAttribute(distributionLayout, "ctasPerCGA"),
            CTASplitNum: RequireTritonBlockedIntListAttribute(distributionLayout, "ctaSplitNum"),
            CTAOrder: RequireTritonBlockedIntListAttribute(distributionLayout, "ctaOrder"),
            ThreadElementOrder: RequireTritonBlockedThreadOrderAttribute(distributionLayout));
    }

    private static int RequireTritonBlockedIntAttribute(DistributionLayout distributionLayout, string name)
    {
        var value = RequireTritonBlockedAttribute(distributionLayout, name);
        return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
            ? parsed
            : throw new NotSupportedException($"TritonBlocked layout attribute {name} must be an integer, got '{value}'.");
    }

    private static IRArray<int> RequireTritonBlockedIntListAttribute(DistributionLayout distributionLayout, string name)
    {
        var value = RequireTritonBlockedAttribute(distributionLayout, name);
        if (!value.StartsWith("[", StringComparison.Ordinal) || !value.EndsWith("]", StringComparison.Ordinal))
        {
            throw new NotSupportedException($"TritonBlocked layout attribute {name} must be an integer list, got '{value}'.");
        }

        var body = value[1..^1];
        if (string.IsNullOrWhiteSpace(body))
        {
            return Array.Empty<int>();
        }

        return body.Split(',', StringSplitOptions.TrimEntries)
            .Select(item => int.TryParse(item, NumberStyles.None, CultureInfo.InvariantCulture, out var parsed)
                ? parsed
                : throw new NotSupportedException($"TritonBlocked layout attribute {name} contains non-integer item '{item}'."))
            .ToArray();
    }

    private static TritonThreadElementOrder RequireTritonBlockedThreadOrderAttribute(DistributionLayout distributionLayout)
    {
        var value = RequireTritonBlockedAttribute(distributionLayout, "threadElementOrder");
        return Enum.TryParse<TritonThreadElementOrder>(value, ignoreCase: false, out var parsed)
            ? parsed
            : throw new NotSupportedException($"TritonBlocked layout has unsupported threadElementOrder '{value}'.");
    }

    private static string RequireTritonBlockedAttribute(DistributionLayout distributionLayout, string name)
    {
        var prefix = name + "=";
        var matches = distributionLayout.Attributes!.Value.Where(attr => attr.StartsWith(prefix, StringComparison.Ordinal)).ToArray();
        return matches.Length == 1
            ? matches[0][prefix.Length..]
            : throw new NotSupportedException($"TritonBlocked layout must provide exactly one {name} attribute, got {matches.Length}.");
    }

    private static Dictionary<string, Dimension> BuildOwnerLocalBindings(TIR.Buffer buffer, DistributedType distributedType, IReadOnlyList<DimVar> loopVars)
    {
        var bindings = new Dictionary<string, Dimension>(StringComparer.Ordinal);
        for (int axis = 0; axis < distributedType.Placement.Rank; axis++)
        {
            bindings[$"owner{axis}"] = GetMeshAxisIndex(distributedType.Placement, axis);
        }

        for (int i = 0; i < loopVars.Count; i++)
        {
            bindings[$"l{i}"] = loopVars[i];
            bindings[$"d{i}"] = loopVars[i];
        }

        if (distributedType.DistributionLayout.Kind == "TritonBlocked")
        {
            BindTritonBlockedOwnerCoordinates(distributedType.DistributionLayout, bindings);
            if (loopVars.Count != 1)
            {
                throw new NotSupportedException($"TritonBlocked affine IO layout evaluation requires one loop coordinate for buffer {buffer.Name}, got {loopVars.Count}.");
            }

            bindings["elem"] = loopVars[0];
        }

        return bindings;
    }

    private static void BindTritonBlockedOwnerCoordinates(DistributionLayout layout, IDictionary<string, Dimension> bindings)
    {
        var tritonLayout = ParseTritonBlockedLayout(layout, "TritonBlocked affine IO layout evaluation");
        var threadId = IR.F.Distributed.ThreadId();
        bindings["cta"] = IR.F.Distributed.ProgramId(0);
        bindings["warp"] = threadId / (Dimension)tritonLayout.ThreadsPerWarp;
        bindings["lane"] = threadId % (Dimension)tritonLayout.ThreadsPerWarp;
    }

    private static IReadOnlyList<(string Name, Dimension Value)> EvaluateMap(IndexMapDescriptor map, IReadOnlyDictionary<string, Dimension> bindings, string context)
    {
        foreach (var input in map.Inputs)
        {
            if (!bindings.ContainsKey(input))
            {
                throw new NotSupportedException($"Cannot evaluate {context}: input {input} is not bound.");
            }
        }

        var outputs = new List<(string Name, Dimension Value)>(map.Outputs.Count);
        foreach (var output in map.Outputs)
        {
            outputs.Add((output.Name, EvaluateIndexExpr(output.Expr, bindings, context)));
        }

        return outputs;
    }

    private static Dimension EvaluateIndexExpr(IndexExpr expr, IReadOnlyDictionary<string, Dimension> bindings, string context)
    {
        return expr switch
        {
            IndexVar variable => bindings.TryGetValue(variable.Name, out var value)
                ? value
                : throw new NotSupportedException($"Cannot evaluate {context}: variable {variable.Name} is not bound."),
            IndexConst constant => constant.Value,
            IndexAdd add => EvaluateIndexAdd(add, bindings, context),
            IndexMul mul => EvaluateIndexMul(mul, bindings, context),
            IndexFloorDiv floorDiv => EvaluateIndexExpr(floorDiv.Value, bindings, context) / EvaluateIndexExpr(floorDiv.Divisor, bindings, context),
            IndexMod mod => EvaluateIndexExpr(mod.Value, bindings, context) % EvaluateIndexExpr(mod.Divisor, bindings, context),
            IndexAny => throw new NotSupportedException($"Cannot evaluate {context}: wildcard index output requires an explicit binding."),
            IndexNamedPrimitive named => throw new NotSupportedException($"Cannot evaluate {context}: named primitive {named.Name} has no affine IO evaluator."),
            _ => throw new NotSupportedException($"Cannot evaluate {context}: unsupported index expression {expr.GetType().Name}."),
        };
    }

    private static Dimension EvaluateIndexAdd(IndexAdd add, IReadOnlyDictionary<string, Dimension> bindings, string context)
    {
        Dimension result = Dimension.Zero;
        foreach (var term in add.Terms)
        {
            result += EvaluateIndexExpr(term, bindings, context);
        }

        return result;
    }

    private static Dimension EvaluateIndexMul(IndexMul mul, IReadOnlyDictionary<string, Dimension> bindings, string context)
    {
        Dimension result = Dimension.One;
        foreach (var factor in mul.Factors)
        {
            result *= EvaluateIndexExpr(factor, bindings, context);
        }

        return result;
    }
}
