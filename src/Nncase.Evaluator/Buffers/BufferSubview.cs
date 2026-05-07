// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Linq;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Buffers;
using Nncase.Utilities;

namespace Nncase.Evaluator.Buffers;

/// <summary>
/// Evaluator for AddressOf.
/// </summary>
public partial class BufferSubviewEvaluator : ITypeInferencer<BufferSubview>
{
    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, BufferSubview target)
    {
        var buffer = context.GetArgument(target, BufferSubview.Buffer);
        var shape = (Shape)context.GetArgument(target, BufferSubview.Shape);
        return buffer.CheckedType switch
        {
            DistributedType distributedType => InferDistributedSubviewType(distributedType, shape),
            TensorType tensorType => new TensorType(tensorType.DType, shape),
            var type => throw new InvalidOperationException($"BufferSubview expects TensorType or DistributedType buffer, got {type}."),
        };
    }

    private static IRType InferDistributedSubviewType(DistributedType distributedType, Shape shape)
    {
        var localTensorType = DistributedUtility.GetDividedTensorType(distributedType);
        if (shape is not RankedShape subviewShape || localTensorType.Shape is not RankedShape localShape)
        {
            throw new InvalidOperationException($"BufferSubview on distributed tensor requires ranked local subview shape, got {shape} from {distributedType}.");
        }

        if (subviewShape.Rank != localShape.Rank)
        {
            throw new InvalidOperationException($"BufferSubview rank {subviewShape.Rank} does not match distributed local shard rank {localShape.Rank}.");
        }

        for (int i = 0; i < subviewShape.Rank; i++)
        {
            var extent = subviewShape[i];
            var localExtent = localShape[i];
            if (!extent.IsFixed || !localExtent.IsFixed)
            {
                throw new InvalidOperationException($"BufferSubview on distributed tensor requires statically bounded local shard extents on axis {i}, got extent {extent} and local extent {localExtent}.");
            }

            if (extent.FixedValue > localExtent.FixedValue)
            {
                throw new InvalidOperationException($"BufferSubview extent {extent} exceeds distributed local shard extent {localExtent} on axis {i}.");
            }
        }

        return new TensorType(localTensorType.DType, shape);
    }
}
