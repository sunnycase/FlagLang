// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc;
using Nncase.Hosting;

namespace Nncase.Evaluator.Affine;

/// <summary>
/// Affine module.
/// </summary>
internal class AffineModule : IApplicationPart
{
    public void ConfigureServices(IRegistrator registrator)
    {
        registrator.RegisterManyInterface<GatherEvaluator>(reuse: Reuse.Singleton);
    }
}
