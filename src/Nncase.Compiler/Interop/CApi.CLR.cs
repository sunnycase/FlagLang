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

namespace Nncase.Compiler.Interop;

public static unsafe partial class CApi
{
    [UnmanagedCallersOnly]
    private static IntPtr ArrayCreate(ArrayElementKind kind, IntPtr* elements, nuint length)
    {
        return kind switch
        {
            // ArrayElementKind.RTValue => ArrayCreateImpl<RTValue>(elements, length),
            // ArrayElementKind.Var => ArrayCreateImpl<IVar>(elements, length),
            ArrayElementKind.Object => ArrayCreateImpl<object>(elements, length),
            _ => IntPtr.Zero,
        };
    }

    private static IntPtr ArrayCreateImpl<T>(IntPtr* elements, nuint length)
    {
        var array = new T[length];
        for (nuint i = 0; i < length; i++)
        {
            array[i] = Get<T>(elements[i]);
        }

        return GCHandle.ToIntPtr(GCHandle.Alloc(array));
    }

    [UnmanagedCallersOnly]
    private static IntPtr ArrayGetItem(IntPtr arrayHandle, nuint index)
    {
        var array = Get<Array>(arrayHandle);
        return GCHandle.ToIntPtr(GCHandle.Alloc(array.GetValue((long)index)));
    }

    [UnmanagedCallersOnly]
    private static nuint ArrayGetLength(IntPtr arrayHandle)
    {
        var array = Get<Array>(arrayHandle);
        return (nuint)array.LongLength;
    }

    [UnmanagedCallersOnly]
    private static void ClrHandleDispose(IntPtr handle)
    {
        Get<IDisposable>(handle).Dispose();
    }

    [UnmanagedCallersOnly]
    private static IntPtr ClrHandleDuplicate(IntPtr handle)
    {
        if (handle != IntPtr.Zero)
        {
            var obj = GCHandle.FromIntPtr(handle).Target;
            return GCHandle.ToIntPtr(GCHandle.Alloc(obj));
        }

        return IntPtr.Zero;
    }

    [UnmanagedCallersOnly]
    private static void ClrHandleFree(IntPtr handle)
    {
        if (handle != IntPtr.Zero)
        {
            GCHandle.FromIntPtr(handle).Free();
        }
    }

    [UnmanagedCallersOnly]
    private static void LaunchDebugger()
    {
        Debugger.Launch();
        while (!Debugger.IsAttached)
        {
            Thread.Yield();
        }
    }

    [UnmanagedCallersOnly]
    private static IntPtr StreamCreate(CStreamMT* mt, IntPtr handle)
    {
        return GCHandle.ToIntPtr(GCHandle.Alloc(new CStream(mt, handle)));
    }
}
