// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase;
using Nncase.IR;

namespace Nncase.Quantization;

/// <summary>
/// <see cref="ICalibrationDatasetProvider"/> that generates random inputs data.
/// </summary>
public sealed class RandomCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RandomCalibrationDatasetProvider"/> class.
    /// </summary>
    /// <param name="vars">Input parameters.</param>
    /// <param name="samplesCount">Samples count.</param>
    public RandomCalibrationDatasetProvider(IReadOnlyList<IVar> vars, int samplesCount)
    {
        Count = samplesCount;
        Samples = Enumerable.Range(0, samplesCount).Select(i =>
        {
            var values = new Dictionary<IVar, IValue>();
            foreach (var var in vars)
            {
                var (dataType, shape) = ResolveVarSpec(var);
                var rawValue = IR.F.Random.Normal(dataType, 0, 1, 0, shape).Evaluate();
                var finalValue = var.CheckedType is DataType ? new ScalarValue(rawValue.AsTensor()) : rawValue;
                values.Add(var, finalValue);
            }

            return values;
        }).ToAsyncEnumerable();
    }

    /// <inheritdoc/>
    public int? Count { get; }

    /// <inheritdoc/>
    public IAsyncEnumerable<IReadOnlyDictionary<IVar, IValue>> Samples { get; }

    private static (DataType DataType, long[] Shape) ResolveVarSpec(IVar var)
    {
        return var.CheckedType switch
        {
            TensorType tensorType => (tensorType.DType, ToConcreteShape(tensorType.Shape)),
            DistributedType distributedType => (distributedType.TensorType.DType, ToConcreteShape(distributedType.TensorType.Shape)),
            DataType dataType => (dataType, Array.Empty<long>()),
            _ => throw new NotSupportedException($"Random calibration does not support parameter type: {var.CheckedType}"),
        };
    }

    private static long[] ToConcreteShape(Shape shape)
    {
        if (shape.IsScalar)
        {
            return Array.Empty<long>();
        }

        if (!shape.IsRanked)
        {
            throw new InvalidOperationException("Random calibration inputs must have a ranked shape.");
        }

        var dims = new long[shape.Rank];
        for (int i = 0; i < dims.Length; i++)
        {
            var dimension = shape[i];
            dims[i] = dimension.IsFixed ? dimension.FixedValue : 1;
        }

        return dims;
    }
}
