**Commands Run**
```bash
find python/triton/_C -maxdepth 1 -type f -name 'libtriton*.so' -printf '%p\n' | sort
find . -path './.git' -prune -o -name 'libnncaseruntime.so' -printf '%p\n' | sort
find python build -type f -name '*.so' -printf '%p\n' | sort
ldd python/triton/_C/libtriton.so
ldd build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libtriton.so
readelf -d python/triton/_C/libtriton.so | rg 'NEEDED|RPATH|RUNPATH'
readelf -d build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libtriton.so | rg 'NEEDED|RPATH|RUNPATH'
nm -D --demangle python/triton/_C/libtriton.so | rg 'nncase_clr|nncase_object|nncase::clr::tensor_type'
nm -D --demangle build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libnncaseruntime.so | rg 'nncase_clr|nncase_object|nncase::clr::tensor_type'
nm --demangle build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libnncaseruntime.so | rg 'nncase_clr|nncase_object|nncase::clr::tensor_type'
LD_LIBRARY_PATH=build/cmake.linux-x86_64-cpython-3.10/Debug/lib ldd -r python/triton/_C/libtriton.so | rg 'libnncase|undefined symbol|_ZN6nncase3clr'
```

**Findings**
1. `python/triton/_C/libtriton*.so` exists: [libtriton.so](/mnt/home-nas/work/repo/FlagLang/python/triton/_C/libtriton.so). No `libnncaseruntime.so` is present beside it in `python/triton/_C`.

2. `ldd python/triton/_C/libtriton.so` reports:
```text
libnncaseruntime.so => not found
```
The build-tree library resolves it:
```text
libnncaseruntime.so => /mnt/home-nas/work/repo/FlagLang/build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libnncaseruntime.so
```

3. Dynamic tags explain the mismatch:
```text
python/triton/_C/libtriton.so RUNPATH: [$ORIGIN]
build/.../Debug/lib/libtriton.so RUNPATH: [.../Debug/lib:...]
```
So the Python artifact only searches its own directory, where the runtime is missing.

4. `libtriton.so` has undefined nncase CLR symbols:
```text
U nncase::clr::tensor_type::shape()
U nncase::clr::tensor_type::tensor_type(nncase::clr::datatype, nncase::clr::shape)
U nncase_clr_api
U nncase_clr_initialize
U nncase_clr_uninitialize
U nncase_object_release
```
`libnncaseruntime.so` exports the C wrapper symbols (`T nncase_clr_*`, `T nncase_object_*`), but the two `nncase::clr::tensor_type` C++ symbols are only local (`t`), not exported in the dynamic table. With `LD_LIBRARY_PATH` set, `ldd -r` still reports those two `nncase::clr::tensor_type` symbols unresolved.

**Recommended Next Action For AC-1**

Treat AC-1 as a native linkage/export issue: make the two `nncase::clr::tensor_type` wrapper symbols dynamically resolvable, then install/copy `libnncaseruntime.so` beside `python/triton/_C/libtriton.so` or adjust editable-install RUNPATH so `$ORIGIN` can resolve it. Recheck with `ldd -r python/triton/_C/libtriton.so` before attempting `import triton`. No files were modified.
