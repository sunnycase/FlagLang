// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Triton;

/// <summary>
/// Triton module.
/// </summary>
internal class TritonModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<LoadEvaluator>(reuse: Reuse.Singleton);
        registrator.RegisterManyInterface<StoreEvaluator>(reuse: Reuse.Singleton);
    }
}
