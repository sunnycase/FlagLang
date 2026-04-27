// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Data;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.Passes;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.WithMarker;
using Nncase.Passes.Transforms;

namespace Nncase.Compiler.Interop;

public static unsafe partial class CApi
{
    [UnmanagedCallersOnly]
    private static IntPtr CompileOptionsCreate()
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CompileOptions()
        {
            DumpDir = "dump",
            DumpFlags = Diagnostics.DumpFlags.Compile | Diagnostics.DumpFlags.PassIR,
        }));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompileSessionCreate(IntPtr targetHandle, IntPtr compileOptionsHandle)
    {
        var target = Get<ITarget>(targetHandle);
        var compileOptions = Get<CompileOptions>(compileOptionsHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(CompileSession.Create(target, compileOptions)));
    }

    [UnmanagedCallersOnly]
    private static IntPtr CompileSessionCreatePassManager(IntPtr sessionHandle, byte* namePtr, nuint nameLen)
    {
        var session = Get<CompileSession>(sessionHandle);
        var name = ToString(namePtr, nameLen);
        var pm = session.CreatePassManager(name);
        return GCHandle.ToIntPtr(GCHandle.Alloc(pm));
    }

    [UnmanagedCallersOnly]
    private static byte CompilerServices_InferenceType(IntPtr exprPtr)
    {
        var expr = Get<IR.BaseExpr>(exprPtr);
        var ret = CompilerServices.InferenceType(expr);
        CompilerServices.DumpIR(expr, "infer", "dump", Diagnostics.PrinterFlags.Normal);

        return ret ? (byte)1 : (byte)0;
    }

    [UnmanagedCallersOnly]
    private static void PassManagerAddOptimizeTTIR(IntPtr pmHandle, int capability)
    {
        var pm = Get<IPassManager>(pmHandle);
        TargetIndependentPass(pm);
    }

    [UnmanagedCallersOnly]
    private static IntPtr PassManagerRun(IntPtr pmHandle, IntPtr moduleHandle)
    {
        var pm = Get<IPassManager>(pmHandle);
        var module = Get<IR.IRModule>(moduleHandle);
        var result = pm.RunAsync(module).ConfigureAwait(false).GetAwaiter().GetResult();
        return GCHandle.ToIntPtr(GCHandle.Alloc(result));
    }

    private static void TargetIndependentPass(IPassManager passManager)
    {
        passManager.Add<TTIRToIRPass>();
        passManager.AddWithName<DataflowPass>("TargetIndependent").Configure(c =>
        {
            c.Add<UnbroadcastBinaryLhs>();
            c.Add<UnbroadcastBinaryRhs>();
            c.Add<UnbroadcastCompareLhs>();
            c.Add<UnbroadcastCompareRhs>();

            c.Add<Passes.Rules.Triton.LoadToAffineGather>();
            c.Add<Passes.Rules.Triton.StoreToAffineScatter>();
        });

        passManager.CompileSession.Target.RegisterTargetInDependentPass(passManager, passManager.CompileSession.CompileOptions);

        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
    }
}
