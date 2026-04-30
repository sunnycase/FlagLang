// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.CostModel;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Evaluator.Tensors;

/// <summary>
/// Evaluator for <see cref="Depend"/>.
/// </summary>
public sealed class DependEvaluator : IEvaluator<Depend>, ITypeInferencer<Depend>, IOpPrinter<Depend>, ICostEvaluator<Depend>, IMetricEvaluator<Depend>
{
    /// <inheritdoc/>
    public IValue Visit(IEvaluateContext context, Depend target)
    {
        _ = context.GetArgumentValue(target, Depend.Dependencies);
        return context.GetArgumentValue(target, Depend.Value);
    }

    /// <inheritdoc/>
    public IRType Visit(ITypeInferenceContext context, Depend target)
    {
        var dependencies = context.CheckArgumentType<IRType>(target, Depend.Dependencies);
        if (dependencies is InvalidType invalidDependencies)
        {
            return invalidDependencies;
        }

        return context.CheckArgumentType<IRType>(target, Depend.Value);
    }

    /// <inheritdoc/>
    public string Visit(IPrintOpContext context, Depend target)
    {
        return $"Depend({context.GetArgument(target, Depend.Dependencies)}, {context.GetArgument(target, Depend.Value)})";
    }

    /// <inheritdoc/>
    public Cost Visit(ICostEvaluateContext context, Depend target)
    {
        return new() { [CostFactorNames.CPUCycles] = 1 };
    }

    /// <inheritdoc/>
    public Metric Visit(IMetricEvaluateContext context, Depend target) => Metric.Zero;
}
