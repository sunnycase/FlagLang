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
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Nncase.Diagnostics;
using Nncase.Passes;
using Nncase.Passes.Rules.Lower;
using Nncase.Passes.Rules.Neutral;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Passes.Rules.WithMarker;
using Nncase.Passes.Transforms;
using Nncase.Targets;

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

    [UnmanagedCallersOnly]
    private static IntPtr IRModuleCompileToCubin(IntPtr moduleHandle, byte* optionsJsonPtr, nuint optionsJsonLength)
    {
        try
        {
            return IRModuleCompileToCubinCore(moduleHandle, optionsJsonPtr, optionsJsonLength);
        }
        catch (Exception ex)
        {
            var json = JsonSerializer.Serialize(new Dictionary<string, object?>
            {
                ["error"] = ex.ToString(),
            });
            return GCHandle.ToIntPtr(GCHandle.Alloc(new NativeCudaCompileResult(Array.Empty<byte>(), json)));
        }
    }

    private static IntPtr IRModuleCompileToCubinCore(IntPtr moduleHandle, byte* optionsJsonPtr, nuint optionsJsonLength)
    {
        var module = Get<IR.IRModule>(moduleHandle);
        var request = NativeCudaCompileRequest.Parse(ToString(optionsJsonPtr, optionsJsonLength));
        var dumpDir = request.DumpDir;
        Directory.CreateDirectory(dumpDir);
        EnsureNativeCudaEntry(module);

        var target = CompilerServices.GetTarget(CUDATarget.Kind);
        var compileOptions = new CompileOptions
        {
            DumpDir = dumpDir,
            DumpFlags = DumpFlags.Compile | DumpFlags.PassIR | DumpFlags.Rewrite | DumpFlags.CodeGen,
            TargetOptions = new NTTTargetOptions
            {
                HierarchyNames = "t",
                Hierarchies = new[] { new[] { request.ThreadsPerCta } },
                CudaArchitecture = request.Capability,
                CudaCompiler = request.CudaCompiler,
            },
        };

        using var session = CompileSession.Create(target, compileOptions);
        using var dumpScope = new DumpScope(session.GetRequiredService<IDumpperFactory>().Root, session);
        var originalIr = PrintModule(module);
        var compiledModule = RunNativeCudaPassPipeline(session, module, request.EnableAutoDist);

        using var output = new MemoryStream();
        session.Compiler.Gencode(output);

        var codegenDir = Path.Combine(dumpDir, "CodeGen", CUDATarget.Kind);
        var cubinPath = FindGeneratedCubin(codegenDir, request.Capability);
        var cubin = File.ReadAllBytes(cubinPath);
        var compilerLog = ReadOptional(Path.Combine(codegenDir, "compiler.log"));
        var asm = CollectNativeCudaStages(originalIr, compiledModule, codegenDir, compilerLog);
        var metadata = new Dictionary<string, object?>
        {
            ["name"] = "flaglang_native_entry",
            ["original_entry_name"] = request.EntryName,
            ["shared"] = 0,
            ["num_warps"] = request.NumWarps,
            ["num_ctas"] = request.NumCtas,
            ["cluster_dims"] = request.ClusterDims,
            ["tmem_size"] = 0,
            ["global_scratch_size"] = 0,
            ["global_scratch_align"] = 1,
            ["profile_scratch_size"] = 0,
            ["profile_scratch_align"] = 1,
            ["cuda_compiler"] = request.CudaCompiler,
            ["cuda_arch"] = request.Arch,
            ["enable_auto_dist"] = request.EnableAutoDist,
            ["dump_dir"] = dumpDir,
            ["cubin_path"] = cubinPath,
        };

        var json = JsonSerializer.Serialize(new Dictionary<string, object?>
        {
            ["metadata"] = metadata,
            ["asm"] = asm,
            ["compiler_log"] = compilerLog,
        });
        return GCHandle.ToIntPtr(GCHandle.Alloc(new NativeCudaCompileResult(cubin, json)));
    }

    private static void EnsureNativeCudaEntry(IR.IRModule module)
    {
        if (module.Entry is not null)
        {
            return;
        }

        module.Entry = SelectEntryBaseFunction(module)
            ?? throw new InvalidOperationException("Native CUDA compilation requires an IRModule entry function.");
    }

    private static IR.IRModule RunNativeCudaPassPipeline(CompileSession session, IR.IRModule module, bool enableAutoDist)
    {
        using var scope = new CompileSessionScope(session);
        var compiler = (Nncase.Compiler.Compiler)session.Compiler;
        compiler.ImportIRModule(module);

        var passManager = session.CreatePassManager("NativeCudaPipeline");
        compiler.TargetIndependentPass(passManager);
        session.Target.RegisterPostAutoVectorizePass(passManager, session.CompileOptions);
        if (enableAutoDist)
        {
            compiler.AutoDistributedPass(passManager);
        }

        compiler.AutoTilingPass(passManager);
        NativeCudaTIRPass(session, passManager);
        var compiledModule = passManager.RunAsync(module).ConfigureAwait(false).GetAwaiter().GetResult();
        compiler.ImportIRModule(compiledModule);
        return compiledModule;
    }

    private static void NativeCudaTIRPass(CompileSession session, IPassManager passManager)
    {
        session.Target.RegisterTIRSelectionPass(passManager, session.CompileOptions);
        passManager.Add<AddFunctionToModule>();
        passManager.AddWithName<PrimFuncPass>("RemoveFunctionWrapper").Configure(p =>
        {
            p.Add<Passes.Mutators.RemoveFunctionWrapper>();
        });

        passManager.Add<RemoveUnusedFunctions>();
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
        passManager.Add<BufferizePass>();
    }

    [UnmanagedCallersOnly]
    private static nuint NativeCudaCompileResultGetJson(IntPtr resultHandle, byte* buffer, nuint bufferLength)
    {
        var result = Get<NativeCudaCompileResult>(resultHandle);
        return WriteUtf8(result.Json, buffer, bufferLength);
    }

    [UnmanagedCallersOnly]
    private static nuint NativeCudaCompileResultGetCubin(IntPtr resultHandle, byte* buffer, nuint bufferLength)
    {
        var result = Get<NativeCudaCompileResult>(resultHandle);
        return WriteBytes(result.Cubin, buffer, bufferLength);
    }

    private static nuint WriteBytes(byte[] bytes, byte* buffer, nuint bufferLength)
    {
        if (buffer != null && bufferLength > 0)
        {
            var copyLength = Math.Min((int)bufferLength, bytes.Length);
            bytes.AsSpan(0, copyLength).CopyTo(new Span<byte>(buffer, copyLength));
        }

        return (nuint)bytes.Length;
    }

    private static string FindGeneratedCubin(string codegenDir, int capability)
    {
        var candidates = new[]
        {
            Path.Combine(codegenDir, "build", "nncase_ntt_module.cubin"),
            Path.Combine(codegenDir, "build", $"linked_sm_{capability}.o"),
            Path.Combine(codegenDir, "build", "nncase_ntt_module"),
        };

        foreach (var candidate in candidates)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new FileNotFoundException($"CUDA codegen did not produce a cubin artifact under {codegenDir}.");
    }

    private static Dictionary<string, string> CollectNativeCudaStages(string originalIr, IR.IRModule compiledModule, string codegenDir, string compilerLog)
    {
        var stages = new Dictionary<string, string>
        {
            ["triton_tir"] = originalIr,
            ["after_compile"] = PrintModule(compiledModule),
            ["tir"] = PrintModule(compiledModule),
        };

        AddIfExists(stages, "ntt_cu", Path.Combine(codegenDir, "thread_main.cu"));
        AddIfExists(stages, "kernel_header", Path.Combine(codegenDir, "kernel_functions.h"));
        AddIfExists(stages, "compiler_cmd", Path.Combine(codegenDir, "compiler.cmd"));
        if (!string.IsNullOrWhiteSpace(compilerLog))
        {
            stages["compiler_log"] = compilerLog;
        }

        return stages;
    }

    private static void AddIfExists(IDictionary<string, string> stages, string name, string path)
    {
        if (File.Exists(path))
        {
            stages[name] = File.ReadAllText(path);
        }
    }

    private static string ReadOptional(string path) => File.Exists(path) ? File.ReadAllText(path) : string.Empty;

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

    private sealed record NativeCudaCompileResult(byte[] Cubin, string Json);

    private sealed record NativeCudaCompileRequest(
        string EntryName,
        string Arch,
        int Capability,
        int NumWarps,
        int NumCtas,
        int[] ClusterDims,
        int ThreadsPerCta,
        string DumpDir,
        string CudaCompiler,
        bool EnableAutoDist)
    {
        public static NativeCudaCompileRequest Parse(string json)
        {
            using var document = JsonDocument.Parse(json);
            var root = document.RootElement;
            var capability = GetInt(root, "capability", 80);
            var numWarps = GetInt(root, "num_warps", 4);
            var warpSize = GetInt(root, "warp_size", 32);
            var threadsPerCta = GetInt(root, "threads_per_cta", Math.Max(1, numWarps * warpSize));
            var dumpDir = GetString(root, "dump_dir", string.Empty);
            if (string.IsNullOrWhiteSpace(dumpDir))
            {
                dumpDir = Path.Combine(Path.GetTempPath(), "flaglang-cuda-pipeline", Guid.NewGuid().ToString("N"));
            }

            return new(
                EntryName: GetString(root, "entry_name", string.Empty),
                Arch: GetString(root, "arch", $"sm_{capability}"),
                Capability: capability,
                NumWarps: numWarps,
                NumCtas: GetInt(root, "num_ctas", 1),
                ClusterDims: GetIntArray(root, "cluster_dims", new[] { 1, 1, 1 }),
                ThreadsPerCta: threadsPerCta,
                DumpDir: dumpDir,
                CudaCompiler: GetString(root, "cuda_compiler", Environment.GetEnvironmentVariable("NNCASE_CUDA_COMPILER") ?? "nvcc"),
                EnableAutoDist: GetBool(root, "enable_auto_dist", false));
        }

        private static string GetString(JsonElement root, string name, string fallback) =>
            root.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() ?? fallback : fallback;

        private static int GetInt(JsonElement root, string name, int fallback) =>
            root.TryGetProperty(name, out var value) && value.TryGetInt32(out var result) ? result : fallback;

        private static bool GetBool(JsonElement root, string name, bool fallback) =>
            root.TryGetProperty(name, out var value) && value.ValueKind is JsonValueKind.True or JsonValueKind.False ? value.GetBoolean() : fallback;

        private static int[] GetIntArray(JsonElement root, string name, int[] fallback)
        {
            if (!root.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array)
            {
                return fallback;
            }

            return value.EnumerateArray().Where(item => item.TryGetInt32(out _)).Select(item => item.GetInt32()).ToArray();
        }
    }
}
