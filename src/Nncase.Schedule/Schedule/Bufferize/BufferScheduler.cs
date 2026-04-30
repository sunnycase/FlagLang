// Copyright (c) SunnyCase. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Schedule.Bufferize;

public sealed record BufferScheduleResult(BufferStorage Storage, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> Buffers, long MemoryPoolStart, long MemoryPoolEnd, int Alignment);

public sealed record BufferScheduleOptions(long StartAddress = 0);

public abstract class BufferScheduler
{
    private static readonly Type[] _bufferSchedulerTypes = [
        typeof(SATBufferScheduler),
    ];

    public BufferScheduler(BufferStorage storage)
    {
        Storage = storage.WithoutAlignment();
    }

    public BufferStorage Storage { get; }

    public MemoryLocation MemoryLocation => Storage.ToLegacyMemoryLocation();

    public static BufferScheduleResult Schedule(MemoryLocation memoryLocation, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes, BufferScheduleOptions options) =>
        Schedule(BufferStorage.FromLegacy(memoryLocation), lifetimes, options);

    public static BufferScheduleResult Schedule(BufferStorage storage, IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes, BufferScheduleOptions options)
    {
        storage = storage.WithoutAlignment();
        if (storage.PhysicalLocation is PhysicalMemorySpace.Register)
        {
            throw new InvalidOperationException($"Register storage {storage} must not enter addressable buffer scheduling.");
        }

        if (UsesReusablePool(storage))
        {
            foreach (var schedulerType in _bufferSchedulerTypes)
            {
                var scheduler = (BufferScheduler)ActivatorUtilities.CreateInstance(CompileSessionScope.GetCurrentThrowIfNull(), schedulerType, storage);
                if (scheduler.TrySchedule(lifetimes, options, out var result))
                {
                    return result;
                }
            }
        }
        else if (UsesLinearPool(storage))
        {
            var scheduler = new LinearBufferScheduler(storage);
            if (scheduler.TrySchedule(lifetimes, options, out var result))
            {
                return result;
            }
        }

        throw new NotSupportedException($"Unable to schedule buffers with storage {storage}.");
    }

    public static IReadOnlyDictionary<BufferStorage, BufferScheduleResult> Schedule(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes, Func<BufferStorage, BufferScheduleOptions> options)
    {
        var result = new Dictionary<BufferStorage, BufferScheduleResult>();
        foreach (var group in lifetimes.GroupBy(x => x.Value.Buffer.Storage.WithoutAlignment()))
        {
            if (UsesReusablePool(group.Key) || UsesLinearPool(group.Key))
            {
                var lifetimeDict = group.ToDictionary(x => x.Key, x => x.Value, (IEqualityComparer<TIR.PhysicalBuffer>)ReferenceEqualityComparer.Instance);
                result.Add(group.Key, Schedule(group.Key, lifetimeDict, options(group.Key)));
            }
        }

        return result;
    }

    public bool TrySchedule(IReadOnlyDictionary<TIR.PhysicalBuffer, BufferLifetime> lifetimes, BufferScheduleOptions options, [MaybeNullWhen(false)] out BufferScheduleResult result)
    {
        long maxMemoryPoolEnd = options.StartAddress;
        int maxAlignment = 8;
        foreach (var lifetime in lifetimes.Values)
        {
            if (lifetime.Buffer.Storage.WithoutAlignment() != Storage)
            {
                throw new ArgumentException($"Storage to schedule of {lifetime.Buffer} is {lifetime.Buffer.Storage}, but expected {Storage}.");
            }

            var alignment = Math.Max(8, lifetime.Buffer.Alignment);
            maxMemoryPoolEnd = MathUtility.AlignUp(maxMemoryPoolEnd, alignment) + lifetime.Memory.Size;
            maxAlignment = Math.Max(maxAlignment, alignment);
        }

        if (TryScheduleCore(lifetimes.Values, maxMemoryPoolEnd, options, out var memoryPoolEnd))
        {
            result = new(Storage, lifetimes, options.StartAddress, memoryPoolEnd, maxAlignment);
            return true;
        }
        else
        {
            result = null;
            return false;
        }
    }

    protected abstract bool TryScheduleCore(IEnumerable<BufferLifetime> lifetimes, long maxMemoryPoolEnd, BufferScheduleOptions options, out long memoryPoolEnd);

    private static bool UsesReusablePool(BufferStorage storage) =>
        storage.Usage is BufferUsage.Temp or BufferUsage.Scratch or BufferUsage.Staging
        && storage.Scope is BufferScope.ThreadLocal or BufferScope.WarpLocal or BufferScope.BlockLocal
        && storage.PhysicalLocation is PhysicalMemorySpace.LocalAddressable or PhysicalMemorySpace.SMem;

    private static bool UsesLinearPool(BufferStorage storage) =>
        storage.Usage is BufferUsage.Output
        || (storage.Usage is BufferUsage.Const && storage.PhysicalLocation is PhysicalMemorySpace.ConstMem);
}
