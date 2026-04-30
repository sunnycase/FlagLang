// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using Nncase.IR;
using Nncase.Schedule;
using Nncase.TIR;

namespace Nncase.CodeGen.NTT;

/// <summary>
/// the csource code compiler.
/// </summary>
public class CSourceCompiler
{
    private static string? _vcVarPath;

    private readonly bool _isCUDA;
    private readonly string _cudaCompiler;
    private readonly int _cudaArchitecture;

    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _exe = string.Empty;

    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _arch = string.Empty;

    /// <summary>
    /// compiler exe name.
    /// </summary>
    private string _ext = string.Empty;

    public CSourceCompiler(bool isCUDA, string cudaCompiler = "nvcc", int cudaArchitecture = 80)
    {
        _isCUDA = isCUDA;
        _cudaCompiler = string.IsNullOrWhiteSpace(cudaCompiler) ? "nvcc" : cudaCompiler;
        _cudaArchitecture = cudaArchitecture <= 0 ? 80 : cudaArchitecture;
        PlatformSpecific();
        ArchSpecific();
    }

    protected static string? VCVarPath => _vcVarPath ??= FindVCVarPath();

    protected string Exe
    {
        get => _exe;
    }

    protected string Arch
    {
        get => _arch;
    }

    protected string Ext
    {
        get => _ext;
    }

    /// <summary>
    /// compile the source txt, write to the out_path.
    /// </summary>
    /// <param name="sourcePath"> c source code.</param>
    /// <param name="outPath"> out .so path. </param>
    /// <returns> outPath. </returns>
    public string Compile(string sourcePath, string outPath)
    {
        var errMsg = new StringBuilder(8192);
        using (var errWriter = new StringWriter(errMsg))
        {
            using (var proc = new Process())
            {
                proc.StartInfo.FileName = Exe;
                proc.StartInfo.Arguments = ArgumentsSpecific(sourcePath, outPath);
                proc.StartInfo.WorkingDirectory = Directory.GetCurrentDirectory();
                proc.StartInfo.RedirectStandardError = true;
                proc.StartInfo.RedirectStandardOutput = true;
                Directory.CreateDirectory(sourcePath);
                File.WriteAllText(Path.Join(sourcePath, "compiler.cmd"), $"{proc.StartInfo.FileName} {proc.StartInfo.Arguments}");
                proc.OutputDataReceived += (sender, e) =>
                {
                    try
                    {
                        errWriter.WriteLine(e.Data);
                    }
                    catch (ArgumentException)
                    {
                    }
                };
                proc.ErrorDataReceived += (sender, e) =>
                {
                    try
                    {
                        errWriter.WriteLine(e.Data);
                    }
                    catch (ArgumentException)
                    {
                    }
                };
                proc.Start();
                proc.BeginErrorReadLine();
                proc.BeginOutputReadLine();
                proc.WaitForExit();
                File.WriteAllText(Path.Join(sourcePath, "compiler.log"), errMsg.ToString());
                if (proc.ExitCode != 0)
                {
                    throw new InvalidOperationException(errMsg.ToString());
                }
            }
        }

        if (_isCUDA)
        {
            GenerateCudaPtxDump(sourcePath);

            var linkedCubin = Path.Join(sourcePath, "build", $"linked_sm_{_cudaArchitecture}.o");
            var namedCubin = Path.Join(sourcePath, "build", "nncase_ntt_module.cubin");
            if (File.Exists(linkedCubin))
            {
                File.Copy(linkedCubin, namedCubin, overwrite: true);
                return namedCubin;
            }
        }

        return outPath;
    }

    /// <summary>
    /// create the temp dll file and compile source
    /// <see cref="Compile(string, string)"/>.
    /// </summary>
    public string Compile(string sourcePath) => Compile(sourcePath, Path.Join(sourcePath, "build", Path.GetFileName(sourcePath)));

    private static string? FindVCVarPath()
    {
        var vsDir = Environment.GetEnvironmentVariable("VSAPPIDDIR");
        if (!string.IsNullOrEmpty(vsDir))
        {
            return Path.Combine(vsDir, "..\\..\\VC\\Auxiliary\\Build\\vcvarsall.bat");
        }
        else
        {
            var vsWhereDir = Path.Combine(Environment.GetEnvironmentVariable("ProgramFiles(x86)")!, "Microsoft Visual Studio\\Installer\\vswhere");
            if (string.IsNullOrEmpty(vsWhereDir))
            {
                return null;
            }

            using (var proc = new Process())
            {
                proc.StartInfo.FileName = vsWhereDir;
                proc.StartInfo.Arguments = "-prerelease -latest -property installationPath";
                proc.StartInfo.RedirectStandardOutput = true;
                proc.Start();
                proc.WaitForExit();
                vsDir = proc.StandardOutput.ReadLine()!;
                return Path.Combine(vsDir, "VC\\Auxiliary\\Build\\vcvarsall.bat");
            }
        }
    }

    private static string FindNinjaBuildBlock(string buildNinja, string buildOutputSuffix)
    {
        var lines = File.ReadAllLines(buildNinja);
        for (var i = 0; i < lines.Length; i++)
        {
            if (!lines[i].StartsWith("build ", StringComparison.Ordinal) ||
                !lines[i].Contains(buildOutputSuffix, StringComparison.Ordinal))
            {
                continue;
            }

            var block = new StringBuilder(lines[i]);
            for (var j = i + 1; j < lines.Length && lines[j].StartsWith("  ", StringComparison.Ordinal); j++)
            {
                block.AppendLine();
                block.Append(lines[j]);
            }

            return block.ToString();
        }

        throw new InvalidOperationException($"CUDA PTX dump could not find the Ninja build block for '{buildOutputSuffix}' in {buildNinja}.");
    }

    private static string GetNinjaVariable(string block, string name)
    {
        var prefix = $"  {name} =";
        foreach (var line in block.Split(Environment.NewLine))
        {
            if (line.StartsWith(prefix, StringComparison.Ordinal))
            {
                return line[prefix.Length..].Trim();
            }
        }

        throw new InvalidOperationException($"CUDA PTX dump could not find Ninja variable '{name}' for thread_main.cu.");
    }

    private static void WriteProcessLine(StringWriter writer, string? line)
    {
        try
        {
            writer.WriteLine(line);
        }
        catch (ArgumentException)
        {
        }
    }

    private static string QuotePath(string path) => $"\"{path}\"";

    /// <summary>
    /// select current pattern's exe.
    /// </summary>
    /// <exception cref="NotSupportedException">NotSupportedException.</exception>
    private void PlatformSpecific()
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            _exe = "/bin/bash";
            _ext = "so";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            _exe = "/bin/bash";
            _ext = "dylib";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _exe = "cmd";
            _ext = "dll";
        }

        if (System.Environment.GetEnvironmentVariable("NNCASE_CPU_COMPILER") is string exe)
        {
            _exe = exe;
        }
    }

    private void ArchSpecific()
    {
        _arch = RuntimeInformation.OSArchitecture switch
        {
            Architecture.X64 => RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ? "x86-64" : "x86_64",
            Architecture.Arm64 => "arm64",
            _ => throw new NotSupportedException(RuntimeInformation.OSArchitecture.ToString()),
        };
    }

    private void GenerateCudaPtxDump(string sourcePath)
    {
        sourcePath = Path.GetFullPath(sourcePath);
        var buildNinja = Path.Join(sourcePath, "build", "build.ninja");
        var threadMain = Path.Join(sourcePath, "thread_main.cu");
        var outputPtx = Path.Join(sourcePath, "build", "thread_main.ptx");
        if (!File.Exists(buildNinja))
        {
            throw new FileNotFoundException($"CUDA PTX dump requires the generated Ninja build file: {buildNinja}");
        }

        if (!File.Exists(threadMain))
        {
            throw new FileNotFoundException($"CUDA PTX dump requires the generated CUDA entry source: {threadMain}");
        }

        var compileBlock = FindNinjaBuildBlock(buildNinja, "thread_main.cu.o");
        var defines = GetNinjaVariable(compileBlock, "DEFINES");
        var includes = GetNinjaVariable(compileBlock, "INCLUDES");
        var flags = GetNinjaVariable(compileBlock, "FLAGS");
        var arguments = $"-forward-unknown-to-host-compiler {defines} {includes} {flags} -x cu -ptx {QuotePath(threadMain)} -o {QuotePath(outputPtx)}";
        RunCudaPtxCompiler(sourcePath, arguments, outputPtx);
    }

    private void RunCudaPtxCompiler(string sourcePath, string arguments, string outputPtx)
    {
        var logPath = Path.Join(sourcePath, "ptx_compiler.log");
        var cmdPath = Path.Join(sourcePath, "ptx_compiler.cmd");
        var errMsg = new StringBuilder(8192);
        using var errWriter = new StringWriter(errMsg);
        using var proc = new Process();
        proc.StartInfo.FileName = _cudaCompiler;
        proc.StartInfo.Arguments = arguments;
        proc.StartInfo.WorkingDirectory = sourcePath;
        proc.StartInfo.RedirectStandardError = true;
        proc.StartInfo.RedirectStandardOutput = true;
        File.WriteAllText(cmdPath, $"{proc.StartInfo.FileName} {proc.StartInfo.Arguments}");
        proc.OutputDataReceived += (_, e) => WriteProcessLine(errWriter, e.Data);
        proc.ErrorDataReceived += (_, e) => WriteProcessLine(errWriter, e.Data);
        proc.Start();
        proc.BeginErrorReadLine();
        proc.BeginOutputReadLine();
        proc.WaitForExit();
        File.WriteAllText(logPath, errMsg.ToString());
        if (proc.ExitCode != 0)
        {
            throw new InvalidOperationException(errMsg.ToString());
        }

        if (!File.Exists(outputPtx))
        {
            throw new FileNotFoundException($"CUDA PTX compiler completed without producing PTX: {outputPtx}");
        }
    }

    private string ArgumentsSpecific(string sourcePath, string outPath)
    {
        string archConfig = string.Empty;
        if (_isCUDA)
        {
            archConfig = $"-DCMAKE_CUDA_ARCHITECTURES={_cudaArchitecture} -DCMAKE_CUDA_COMPILER={_cudaCompiler}";
        }
        else
        {
            archConfig = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ?
            "-DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl" : string.Empty;
        }

#if DEBUG
        var config = "Release";
#else
        var config = "Release";
#endif
        var script = $"""
            cd {sourcePath} &&
            cmake -E remove_directory build &&
            cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE={config} {archConfig} &&
            cmake --build build --config {config}
            """.Replace("\r\n", " ", StringComparison.Ordinal);

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            return $"-c \"{script}\"";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            return $"-c \"{script}\"";
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            var vcVarPath = FindVCVarPath();
            if (!string.IsNullOrEmpty(vcVarPath))
            {
                return $"/C \"(\"{vcVarPath}\" x64) && {script}\"";
            }

            return $"/C {script}";
        }

        throw new NotSupportedException("Only Support Linux/Osx/Windows");
    }
}
