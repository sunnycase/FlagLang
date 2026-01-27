// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using DryIoc;
using DryIoc.Microsoft.DependencyInjection;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Nncase;
using Nncase.Compiler;
using Nncase.Compiler.Hosting;
using Nncase.Hosting;

namespace Microsoft.Extensions.Hosting;

/// <summary>
/// Compiler host builder extentensions.
/// </summary>
public static class CompilerHostBuilderExtensions
{
    /// <summary>
    /// Configure compiler host builder.
    /// </summary>
    /// <param name="hostBuilder">Host builder.</param>
    /// <param name="configureCompiler">Configure compiler builder.</param>
    /// <returns>Created host builder.</returns>
    public static IHostBuilder ConfigureCompiler(this IHostBuilder hostBuilder, Action<ICompilerBuilder>? configureCompiler = null)
    {
        hostBuilder.UseServiceProviderFactory(new DryIocServiceProviderFactory())
            .ConfigureContainer<Container>(ConfigureBuiltinModules)
            .ConfigureServices(ConfigureServices)
            .ConfigureLogging(ConfigureLogging);

        hostBuilder.ConfigureContainer<Container>(x => configureCompiler?.Invoke(new CompilerBuilder(x)));
        return hostBuilder;
    }

    private static void ConfigureBuiltinModules(Container builder)
    {
        builder.AddCore()
            .AddCodeGen()
            .AddDiagnostics()
            .AddEGraph()
            .AddEvaluator()
            .AddGraph()
            .AddPasses()
            .AddSchedule();

        // Add modules
        builder.AddNTT();
    }

    private static void ConfigureServices(HostBuilderContext context, IServiceCollection services)
    {
        services.AddLogging();
        services.AddRazorTemplating();

        services.AddScoped<ICompiler, Compiler>();
    }

    private static void ConfigureLogging(ILoggingBuilder loggingBuilder)
    {
        loggingBuilder.ClearProviders();
        loggingBuilder.AddConsole();
    }
}
