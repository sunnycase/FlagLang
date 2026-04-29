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
using Nncase.CodeGen.NTT;
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
        var compiler = (Nncase.Compiler.Compiler)session.Compiler;
        compiler.ImportIRModule(module);
        var nncaseModule = RunNativeCudaImportPass(session, compiler.Module);
        var importedAbi = DescribeNativeCudaEntryAbi(nncaseModule);
        compiler.ImportIRModule(nncaseModule);
        compiler.CompileAsync(enableAutoDistributed: request.EnableAutoDist)
            .ConfigureAwait(false).GetAwaiter().GetResult();
        var compiledModule = compiler.Module;
        EnsureNoTritonLoadStore(compiledModule, "after CompileAsync");

        using var output = new MemoryStream();
        session.Compiler.Gencode(output);

        var codegenDir = Path.Combine(dumpDir, "CodeGen", CUDATarget.Kind);
        var cubinPath = FindGeneratedCubin(codegenDir, request.Capability);
        var cubin = File.ReadAllBytes(cubinPath);
        ValidateGeneratedCubin(cubinPath, cubin, "flaglang_native_entry", request.Cuobjdump);
        var compilerLog = ReadOptional(Path.Combine(codegenDir, "compiler.log"));
        var passDumps = CollectPassDumpNames(dumpDir);
        var asm = CollectNativeCudaStages(originalIr, nncaseModule, compiledModule, dumpDir, codegenDir, compilerLog, passDumps);
        var metadata = new Dictionary<string, object?>
        {
            ["name"] = "flaglang_native_entry",
            ["original_entry_name"] = request.EntryName,
            ["flaglang_abi"] = DescribeNativeCudaAbi(importedAbi, compiledModule, "flaglang_native_entry", request),
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
            ["cuobjdump"] = request.Cuobjdump,
            ["cuda_arch"] = request.Arch,
            ["enable_auto_dist"] = request.EnableAutoDist,
            ["dump_dir"] = dumpDir,
            ["cubin_path"] = cubinPath,
            ["pass_dumps"] = passDumps,
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

    private static IR.IRModule RunNativeCudaImportPass(CompileSession session, IR.IRModule module)
    {
        using var scope = new CompileSessionScope(session);
        var passManager = session.CreatePassManager("NativeCudaImportPass");
        passManager.Add<TTIRToIRPass>();
        passManager.AddWithName<DataflowPass>("TritonLoadStoreToAffineIO").Configure(c =>
        {
            c.Add<Passes.Rules.Triton.LoadToAffineGather>();
            c.Add<Passes.Rules.Triton.StoreToAffineScatter>();
        });
        passManager.Add<InferRangePass>();
        passManager.Add<OptimizeByRangePass>();
        var nncaseModule = passManager.RunAsync(module).ConfigureAwait(false).GetAwaiter().GetResult();
        EnsureNoTritonLoadStore(nncaseModule, "after native CUDA import");
        return nncaseModule;
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

    private static Dictionary<string, string> CollectNativeCudaStages(
        string originalIr,
        IR.IRModule nncaseModule,
        IR.IRModule compiledModule,
        string dumpDir,
        string codegenDir,
        string compilerLog,
        IReadOnlyList<string> passDumps)
    {
        var stages = new Dictionary<string, string>
        {
            ["triton_tir"] = originalIr,
            ["nncase_ir"] = PrintModule(nncaseModule),
            ["after_compile"] = PrintModule(compiledModule),
            ["tir"] = ReadPassDump(dumpDir, "TIRPass") ?? PrintModule(compiledModule),
            ["pass_dumps"] = string.Join(Environment.NewLine, passDumps),
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

    private static void EnsureNoTritonLoadStore(IR.IRModule module, string stage)
    {
        var text = PrintModule(module);
        if (text.Contains("Triton.Load", StringComparison.Ordinal) ||
            text.Contains("Triton.Store", StringComparison.Ordinal) ||
            text.Contains("IR.Triton.Load", StringComparison.Ordinal) ||
            text.Contains("IR.Triton.Store", StringComparison.Ordinal))
        {
            throw new InvalidOperationException($"Native CUDA module still contains Triton load/store {stage}.");
        }
    }

    private static NativeCudaParameterAbi DescribeNativeCudaEntryAbi(IR.IRModule module)
    {
        var entry = SelectEntryBaseFunction(module)
            ?? throw new InvalidOperationException("Native CUDA ABI description requires an entry function.");
        return new NativeCudaParameterAbi(
            GetFunctionParameterNames(entry),
            GetFunctionParameterTypes(entry));
    }

    private static Dictionary<string, object?> DescribeNativeCudaAbi(
        NativeCudaParameterAbi importedAbi,
        IR.IRModule compiledModule,
        string entryName,
        NativeCudaCompileRequest request)
    {
        var entry = SelectNativeCudaEntryPrimFunction(compiledModule);
        var parameterOrder = GetFunctionParameterNames(entry);
        var parameterTypes = GetFunctionParameterTypes(entry);
        if (!parameterOrder.SequenceEqual(importedAbi.ArgumentOrder, StringComparer.Ordinal) ||
            !parameterTypes.SequenceEqual(importedAbi.ArgumentTypes, StringComparer.Ordinal))
        {
            throw new InvalidOperationException(
                "Native CUDA compiled entry ABI does not match post-import entry ABI. " +
                $"Imported: ({string.Join(", ", importedAbi.ArgumentOrder)}) [{string.Join(", ", importedAbi.ArgumentTypes)}]; " +
                $"Compiled: ({string.Join(", ", parameterOrder)}) [{string.Join(", ", parameterTypes)}].");
        }

        var rawParameterOrder = parameterOrder.Select(IR.IRHelpers.GetIdentityName).ToArray();
        var rawParameterTypes = GetRawEntryParameterTypes(entry);
        if (request.RuntimeArgumentOrder.Length != parameterOrder.Length ||
            request.RuntimeArgumentTypes.Length != parameterOrder.Length)
        {
            throw new InvalidOperationException(
                $"Native CUDA runtime ABI metadata has {request.RuntimeArgumentOrder.Length} names and {request.RuntimeArgumentTypes.Length} types, but compiled entry has {parameterOrder.Length} parameters.");
        }

        return new Dictionary<string, object?>
        {
            ["entry"] = entryName,
            ["wrapped_entry"] = request.EntryName,
            ["argument_count"] = parameterOrder.Length,
            ["imported_argument_order"] = importedAbi.ArgumentOrder,
            ["imported_argument_types"] = importedAbi.ArgumentTypes,
            ["argument_order"] = parameterOrder,
            ["argument_types"] = parameterTypes,
            ["raw_argument_order"] = rawParameterOrder,
            ["raw_argument_types"] = rawParameterTypes,
            ["runtime_argument_count"] = request.RuntimeArgumentOrder.Length,
            ["runtime_argument_order"] = request.RuntimeArgumentOrder,
            ["runtime_argument_types"] = request.RuntimeArgumentTypes,
            ["wrapper"] = "triton_raw_args_to_thread_main",
        };
    }

    private static Nncase.TIR.PrimFunction SelectNativeCudaEntryPrimFunction(IR.IRModule module)
    {
        var entry = SelectEntryBaseFunction(module);
        if (entry is Nncase.TIR.PrimFunction primFunction)
        {
            return primFunction;
        }

        if (entry is IR.PrimFunctionWrapper wrapper)
        {
            return wrapper.Target;
        }

        var primFunctions = module.Functions.ToArray()
            .OfType<Nncase.TIR.PrimFunction>()
            .Where(function => !function.Name.Contains("device_func", StringComparison.Ordinal))
            .ToArray();
        return primFunctions.Length == 1
            ? primFunctions[0]
            : throw new InvalidOperationException($"Native CUDA compilation expected one compiled entry PrimFunction, got {primFunctions.Length}.");
    }

    private static string[] GetFunctionParameterNames(IR.BaseFunction? function)
    {
        return function switch
        {
            IR.Function f => f.Parameters.ToArray().Select(p => p.Name).ToArray(),
            IR.Fusion f => f.Parameters.ToArray().Select(p => p.Name).ToArray(),
            IR.PrimFunctionWrapper f => f.Target.Parameters.ToArray().Select(p => p.Name).ToArray(),
            Nncase.TIR.PrimFunction f => f.Parameters.ToArray().Select(p => p.Name).ToArray(),
            _ => throw new InvalidOperationException(
                $"Native CUDA ABI description does not support function node '{function?.GetType().FullName ?? "<null>"}'."),
        };
    }

    private static string[] GetFunctionParameterTypes(IR.BaseFunction? function)
    {
        return function switch
        {
            IR.Function f => f.Parameters.ToArray().Select(p => p.CheckedDataType.ToString()).ToArray(),
            IR.Fusion f => f.Parameters.ToArray().Select(p => p.CheckedDataType.ToString()).ToArray(),
            IR.PrimFunctionWrapper f => f.Target.Parameters.ToArray().Select(p => p.CheckedDataType.ToString()).ToArray(),
            Nncase.TIR.PrimFunction f => f.Parameters.ToArray().Select(p => p.CheckedDataType.ToString()).ToArray(),
            _ => throw new InvalidOperationException(
                $"Native CUDA ABI description does not support function node '{function?.GetType().FullName ?? "<null>"}'."),
        };
    }

    private static string[] GetRawEntryParameterTypes(Nncase.TIR.PrimFunction function) =>
        function.Parameters.ToArray().Select(RawEntryParamType).ToArray();

    private static string RawEntryParamType(IR.IVar input)
    {
        if (input.CheckedDataType is PointerType pointerType)
        {
            return $"{pointerType.ElemType.ToC()} *";
        }

        if (input.CheckedDataType is PrimType)
        {
            return input.CheckedDataType.ToC();
        }

        return "std::byte *";
    }

    private static void ValidateGeneratedCubin(string cubinPath, byte[] cubin, string entryName, string cuobjdumpPath)
    {
        if (cubin.Length < 64 ||
            cubin[0] != 0x7f ||
            cubin[1] != (byte)'E' ||
            cubin[2] != (byte)'L' ||
            cubin[3] != (byte)'F')
        {
            throw new InvalidDataException($"CUDA codegen produced a non-ELF cubin artifact: {cubinPath}");
        }

        if (!CubinContainsEntrySymbol(cubinPath, entryName, cuobjdumpPath))
        {
            throw new InvalidDataException($"Generated cubin does not contain entry symbol '{entryName}': {cubinPath}");
        }
    }

    private static bool CubinContainsEntrySymbol(string cubinPath, string entryName, string cuobjdumpPath)
    {
        var cuobjdump = TryRunProcess(cuobjdumpPath, "--dump-elf", cubinPath);
        return !string.IsNullOrWhiteSpace(cuobjdump) && cuobjdump.Contains(entryName, StringComparison.Ordinal);
    }

    private static IReadOnlyList<string> CollectPassDumpNames(string dumpDir)
    {
        if (!Directory.Exists(dumpDir))
        {
            return Array.Empty<string>();
        }

        return Directory.EnumerateDirectories(dumpDir)
            .Select(Path.GetFileName)
            .Where(name => !string.IsNullOrWhiteSpace(name))
            .Cast<string>()
            .Where(name => char.IsDigit(name[0]) || name.Equals("NativeCudaImportPass", StringComparison.Ordinal))
            .OrderBy(name => name, StringComparer.Ordinal)
            .ToArray();
    }

    private static string? ReadPassDump(string dumpDir, string passName)
    {
        if (!Directory.Exists(dumpDir))
        {
            return null;
        }

        var passDir = Directory.EnumerateDirectories(dumpDir)
            .Where(dir => Path.GetFileName(dir).Contains(passName, StringComparison.Ordinal))
            .OrderBy(dir => dir, StringComparer.Ordinal)
            .LastOrDefault();
        if (passDir is null)
        {
            return null;
        }

        var dumpFile = Directory.EnumerateFiles(passDir, "*.*", SearchOption.AllDirectories)
            .Where(path => path.EndsWith(".il", StringComparison.Ordinal) || path.EndsWith(".script", StringComparison.Ordinal))
            .OrderBy(path => path, StringComparer.Ordinal)
            .LastOrDefault(path => Path.GetFileName(path).StartsWith("End", StringComparison.Ordinal) ||
                                   path.Contains("End_", StringComparison.Ordinal));
        return dumpFile is null ? null : File.ReadAllText(dumpFile);
    }

    private static string TryRunProcess(string fileName, params string[] arguments)
    {
        try
        {
            var startInfo = new ProcessStartInfo(fileName)
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
            };
            foreach (var argument in arguments)
            {
                startInfo.ArgumentList.Add(argument);
            }

            using var process = Process.Start(startInfo);
            if (process is null)
            {
                return string.Empty;
            }

            var output = process.StandardOutput.ReadToEnd();
            process.WaitForExit(10_000);
            return process.ExitCode == 0 ? output : string.Empty;
        }
        catch
        {
            return string.Empty;
        }
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

    private sealed record NativeCudaParameterAbi(string[] ArgumentOrder, string[] ArgumentTypes);

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
        string Cuobjdump,
        string[] RuntimeArgumentOrder,
        string[] RuntimeArgumentTypes,
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
                dumpDir = Path.Combine("dump", "flaglang-cuda-pipeline", Guid.NewGuid().ToString("N"));
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
                Cuobjdump: GetString(root, "cuobjdump", Environment.GetEnvironmentVariable("TRITON_CUOBJDUMP_PATH") ?? "cuobjdump"),
                RuntimeArgumentOrder: GetStringArray(root, "runtime_argument_order", Array.Empty<string>()),
                RuntimeArgumentTypes: GetStringArray(root, "runtime_argument_types", Array.Empty<string>()),
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

        private static string[] GetStringArray(JsonElement root, string name, string[] fallback)
        {
            if (!root.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Array)
            {
                return fallback;
            }

            return value.EnumerateArray()
                .Where(item => item.ValueKind == JsonValueKind.String)
                .Select(item => item.GetString() ?? string.Empty)
                .Where(item => !string.IsNullOrWhiteSpace(item))
                .ToArray();
        }
    }
}
