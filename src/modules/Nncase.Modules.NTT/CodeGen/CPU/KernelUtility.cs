// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.CommandLine;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;

namespace Nncase.CodeGen.NTT;

public static class KernelUtility
{
    public static string DimensionsToC(bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType) =>
        DimensionsToC("shape", isFixed, dimensions, isType);

    public static string StridesToC(bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType) =>
        DimensionsToC("strides", isFixed, dimensions, isType);

    public static string DimensionsTypeToC(bool isFixed, ReadOnlySpan<Dimension> dimensions) =>
        DimensionsTypeToC("shape", isFixed, dimensions);

    public static string StridesTypeToC(bool isFixed, ReadOnlySpan<Dimension> dimensions) =>
        DimensionsTypeToC("strides", isFixed, dimensions);

    public static string PlacementToC(this Placement placement)
    {
        return $"mesh<topology::thread, {string.Join(',', placement.Hierarchy)}>";
    }

    public static string SBPToC(this SBP value)
    {
        if (value is SBPSplit s)
        {
            return $"S<{string.Join(", ", s.Axes)}>()";
        }
        else
        {
            return "B";
        }
    }

    public static string ShardingToC(DistributedType distributedType)
    {
        LayoutVerifier.Verify(distributedType, "NTT C++ sharding codegen");
        ValidateExplicitLayoutRepresentableBySbp(distributedType, "NTT C++ sharding codegen");
        var placement = distributedType.Placement;
        var ndSBP = distributedType.AxisPolicies;

        var sb = new StringBuilder("make_sharding<mesh<topology::thread, ");
        for (int i = 0; i < placement.Rank; i++)
        {
            var value = placement.Hierarchy[i];
            sb.Append($"{value}");
            if (i != placement.Rank - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append(">>(");
        for (int axis = 0; axis < distributedType.TensorType.Shape.Rank; axis++)
        {
            var value = ndSBP[axis];
            sb.Append(SBPToC(value));
            if (axis != distributedType.TensorType.Shape.Rank - 1)
            {
                sb.Append(", ");
            }
        }

        sb.Append(')');
        return sb.ToString();
    }

    private static void ValidateExplicitLayoutRepresentableBySbp(DistributedType distributedType, string context)
    {
        if (distributedType.ExplicitDistributionLayout is null)
        {
            return;
        }

        DistributionLayout sbpLayout;
        try
        {
            sbpLayout = DistributionLayout.FromAxisPolicies(distributedType.TensorType, distributedType.AxisPolicies, distributedType.Placement);
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            throw CreateExplicitLayoutCodegenException(distributedType, context, $"failed to derive SBP layout from AxisPolicies: {ex.Message}", ex);
        }

        if (!IsEquivalentSbpLayout(distributedType.ExplicitDistributionLayout, sbpLayout))
        {
            throw CreateExplicitLayoutCodegenException(distributedType, context, "explicit DistributionLayout is not exactly representable by the existing make_sharding(... SBP ...) codegen API");
        }
    }

    private static bool IsEquivalentSbpLayout(DistributionLayout explicitLayout, DistributionLayout sbpLayout) =>
        explicitLayout.Kind == "SBP" &&
        explicitLayout.LocalShape == sbpLayout.LocalShape &&
        explicitLayout.ValidPredicate == sbpLayout.ValidPredicate &&
        MapsEqual(explicitLayout.GlobalToOwnerLocal, sbpLayout.GlobalToOwnerLocal) &&
        MapsEqual(explicitLayout.OwnerLocalToGlobal, sbpLayout.OwnerLocalToGlobal);

    private static bool MapsEqual(IndexMapDescriptor lhs, IndexMapDescriptor rhs) =>
        lhs.Name == rhs.Name &&
        lhs.Predicate == rhs.Predicate &&
        lhs.Inverse == rhs.Inverse &&
        lhs.Inputs.SequenceEqual(rhs.Inputs) &&
        lhs.InputDomain.SequenceEqual(rhs.InputDomain) &&
        lhs.OutputDomain.SequenceEqual(rhs.OutputDomain) &&
        lhs.Outputs.Select(output => output.ToString()).SequenceEqual(rhs.Outputs.Select(output => output.ToString()));

    private static NotSupportedException CreateExplicitLayoutCodegenException(DistributedType distributedType, string context, string reason, Exception? inner = null)
    {
        var axisPolicies = string.Join(',', distributedType.AxisPolicies);
        return new NotSupportedException(
            $"{context} cannot lower explicit DistributionLayout through legacy AxisPolicies sharding: {reason}. Layout={distributedType.DistributionLayout.Kind}, Shape={distributedType.TensorType.Shape}, Placement={distributedType.Placement}, AxisPolicies=({axisPolicies}).",
            inner);
    }

    private static string DimensionsToC(string typeName, bool isFixed, ReadOnlySpan<CSymbol> dimensions, bool isType)
    {
        if (isFixed)
        {
            var sb = new StringBuilder($"fixed_{typeName}<");
            AppendDimValues(sb, dimensions);
            sb.Append(isType ? ">" : ">{}");
            return sb.ToString();
        }
        else
        {
            if (isType)
            {
                return $"ranked_{typeName}<{dimensions.Length}>";
            }
            else
            {
                var sb = new StringBuilder($"make_ranked_{typeName}(");
                AppendDimValues(sb, dimensions);
                sb.Append(')');
                return sb.ToString();
            }
        }
    }

    private static string DimensionsTypeToC(string typeName, bool isFixed, ReadOnlySpan<Dimension> dimensions)
    {
        if (isFixed)
        {
            var sb = new StringBuilder($"fixed_{typeName}<");
            AppendDimValues(sb, dimensions);
            sb.Append('>');
            return sb.ToString();
        }
        else
        {
            return $"ranked_{typeName}<{dimensions.Length}>";
        }
    }

    private static void AppendDimValues(StringBuilder sb, ReadOnlySpan<CSymbol> dimensions)
    {
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = dimensions[i].Name;
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }
    }

    private static void AppendDimValues(StringBuilder sb, ReadOnlySpan<Dimension> dimensions)
    {
        for (int i = 0; i < dimensions.Length; i++)
        {
            var value = dimensions[i].FixedValue;
            sb.Append(value);
            if (i != dimensions.Length - 1)
            {
                sb.Append(", ");
            }
        }
    }
}
