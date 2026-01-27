// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

public sealed class FlattenIRBlockPass : FunctionPass
{
    protected override Task<BaseFunction> RunCoreAsync(BaseFunction input, RunPassContext context)
    {
        if (input is Function func)
        {
            var block = func.Body;
            while (block.Body is IRBlock inner)
            {
                var innerParams = inner.Parameters.ToArray();
                var outerParams = block.Parameters.ToArray();
                if (!innerParams.SequenceEqual(outerParams, ReferenceEqualityComparer.Instance))
                {
                    break;
                }

                block = block.With(body: inner.Body, parameters: outerParams);
            }

            if (!ReferenceEquals(block, func.Body))
            {
                return Task.FromResult((BaseFunction)func.With(body: block));
            }
        }

        return Task.FromResult(input);
    }
}
