// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using Nncase.CostModel;
using Nncase.IR;
using Nncase.IR.Tensors;
using OrtKISharp;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Bitcast"/>.
/// </summary>
public class BitcastEvaluator : IEvaluator<Bitcast>, ITypeInferencer<Bitcast>, ICostEvaluator<Bitcast>, IMetricEvaluator<Bitcast>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Bitcast cast)
    {
        var input = context.GetArgumentValue(cast, Bitcast.Input).AsTensor();
        return Value.FromTensor(input.CastTo(cast.NewType, CastMode.Reinterpret));
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Bitcast target)
    {
        var input = context.CheckArgumentType<IRType>(target, Bitcast.Input);
        return input switch
        {
            TensorType t => Visit(target, t),
            DistributedType d => Visit(target, d),
            _ => new InvalidType(input.GetType().ToString()),
        };
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Bitcast target)
    {
        return new()
        {
            [CostFactorNames.CPUCycles] = 1,
        };
    }

    public Metric Visit(IMetricEvaluateContext context, Bitcast target)
    {
        return new();
    }

    private static bool TryScaleLastDimensionForBitcast(Dimension elementCount, int srcSize, int destSize, out Dimension scaledLastDim, out string reason)
    {
        scaledLastDim = Dimension.Unknown;
        reason = string.Empty;
        if (elementCount.IsFixed)
        {
            var totalBytes = checked(elementCount.FixedValue * srcSize);
            if (totalBytes % destSize != 0)
            {
                reason = $"Bitcast from {srcSize}-byte elements to {destSize}-byte elements requires the last dimension byte size {totalBytes} to be divisible by {destSize}.";
                return false;
            }

            scaledLastDim = totalBytes / destSize;
            return true;
        }

        if (srcSize % destSize == 0)
        {
            scaledLastDim = elementCount * (srcSize / destSize);
            return true;
        }

        reason = $"Bitcast from {srcSize}-byte elements to {destSize}-byte elements requires a statically divisible last dimension when widening element size.";
        return false;
    }

    private IRType Visit(Bitcast target, TensorType input)
    {
        if (input.Shape is not RankedShape rankedInShape)
        {
            return new InvalidType(input.ToString());
        }

        var srcSize = input.DType.SizeInBytes;
        var destSize = target.NewType.SizeInBytes;
        var newDimensions = rankedInShape.Dimensions.ToArray();

        if (srcSize != destSize)
        {
            var elementCount = newDimensions.Length == 0 ? Dimension.One : newDimensions[^1];
            if (!TryScaleLastDimensionForBitcast(elementCount, srcSize, destSize, out var scaledLastDim, out var reason))
            {
                return new InvalidType(reason);
            }

            if (newDimensions.Length == 0)
            {
                newDimensions = [scaledLastDim];
            }
            else
            {
                newDimensions[^1] = scaledLastDim;
            }
        }

        return new TensorType(target.NewType, newDimensions);
    }

    private IRType Visit(Bitcast target, DistributedType input)
    {
        var tensorType = Visit(target, input.TensorType);
        if (tensorType is not TensorType outTensorType)
        {
            return tensorType;
        }

        var invalid = new InvalidType(input.ToString());
        var ndsbp = new SBP[input.AxisPolicies.Count];
        for (int i = 0; i < ndsbp.Length; i++)
        {
            if (input.AxisPolicies[i] is SBPPartial)
            {
                return invalid;
            }

            ndsbp[i] = input.AxisPolicies[i];
        }

        var output = input.HasExplicitLayout
            ? input with { TensorType = outTensorType }
            : new DistributedType(outTensorType, ndsbp, input.Placement, input.Partial);

        if (input.HasExplicitLayout)
        {
            try
            {
                LayoutVerifier.VerifyEquivalentForBitcast(input, output, "IR.Tensors.Bitcast type inference");
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
            {
                return new InvalidType(ex.Message);
            }
        }

        return output;
    }
}
