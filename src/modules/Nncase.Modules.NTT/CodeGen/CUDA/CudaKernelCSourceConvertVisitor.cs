// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.Targets;

namespace Nncase.CodeGen.NTT;

internal sealed class CudaKernelCSourceConvertVisitor : KernelCSourceConvertVisitor
{
    public CudaKernelCSourceConvertVisitor(NTTTargetOptions targetOptions)
        : base(targetOptions)
    {
    }
}
