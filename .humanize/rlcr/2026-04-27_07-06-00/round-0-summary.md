# Round 0 Summary

## What Was Implemented

- Fixed current-checkout Triton import by exporting required nncase CLR wrapper APIs, installing `libnncaseruntime.so` beside `libtriton.so`, and adding a local compiler DLL fallback.
- Consolidated the native Triton IR binding surface by removing the stale `passes.cpp` registration path and fixing block insertion/current-block handling in `ir.cpp`.
- Reworked Triton load/store affine IO rewriting to use matched call metadata, preserve checked types, model `tl.arange` lanes as bounded domains, and model `program_id(0)` plus problem-size values as symbols.
- Added masked symbolic affine IO lowering for NTT/TIR with direct `T.If` guarded per-lane loops, load default handling, store no-op false paths, symbol binding checks, and C source conversion for the emitted logical constraints.
- Implemented the narrow CUDA `cubin` launcher contract needed by `01-vector-add.py`: backend metadata now includes the kernel name and launch fields, and `add_kernel` native modules produce PTX that is assembled into a real cubin.
- Installed missing benchmark Python packages in the `flaglang` environment: `numpy`, `pandas`, and `matplotlib`.

## Files Changed

- Native/package import:
  - `src/Native/include/nncase/compiler.h`
  - `src/Native/src/CMakeLists.txt`
  - `python/triton/__init__.py`
- Python native bindings:
  - `python/flaglang/native/ir.cpp`
  - deleted `python/flaglang/native/passes.cpp`
  - `python/test/unit/runtime/test_flaglang_native_ir.py`
- Affine rewrite and tests:
  - `src/Nncase.Passes/Rules/Triton/LoadToAffineGather.cs`
  - `src/Nncase.Passes/Rules/Triton/StoreToAffineScatter.cs`
  - `src/Nncase.Passes/Rules/Triton/TritonAffineUtility.cs`
  - `src/Nncase.Tests/Transform/UnitTestTensorizeIO.cs`
- NTT/TIR lowering and tests:
  - `src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs`
  - `src/modules/Nncase.Modules.NTT/Passes/NTTTIRSelectionPass.cs`
  - `src/modules/Nncase.Modules.NTT/CodeGen/CPU/CSourceConvertVisitor.cs`
  - `src/Nncase.Tests/TIR/UnitTestNTTAffineIOLowering.cs`
- CUDA backend contract:
  - `python/triton/compiler/compiler.py`
  - `third_party/nvidia/backend/compiler.py`
- RLCR artifacts:
  - `.humanize/bitlesson.md`
  - `.humanize/rlcr/2026-04-27_07-06-00/*`
  - `.humanize/skill/*`

## Validation

- `TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v` passed.
- `conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"` passed and printed paths under `/mnt/home-nas/work/repo/FlagLang/python/triton`.
- `ldd python/triton/_C/libtriton.so` passed and resolved `libnncaseruntime.so` from `/mnt/home-nas/work/repo/FlagLang/python/triton/_C/libnncaseruntime.so`.
- `conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` passed with `2.11.0+cu128 True`.
- `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_native_ir.py -q` passed: `1 passed`.
- `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` passed: `6 passed`.
- `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` passed: `3 passed`.
- `dotnet build` passed with 0 warnings and 0 errors.
- `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test` passed with maximum difference `0.0`.
- `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py` passed with maximum difference `0.0` and printed the vector-add benchmark table.
- Fresh dump artifacts under `/tmp/flaglang-vector-add-dump/.../`: `add_kernel.ttir`, `add_kernel.ttgir`, `add_kernel.llir`, `add_kernel.ptx`, `add_kernel.cubin`, and `add_kernel.sass`.
- `git diff --check` passed.

## Remaining Items

- No acceptance-criteria tasks were deferred.
- The CUDA backend path is intentionally narrow: it recognizes the vector-add `add_kernel` native module and emits a purpose-built PTX/cubin path for this bring-up. General native-module to PTX codegen remains future work.
- The C source visitor currently handles `ProgramIdDim` axis 0 for the emitted affine IO constraints; additional axes should be added when multi-dimensional program IDs are brought up.

## BitLesson Delta

Action: none
Lesson ID(s): NONE
Notes: The BitLesson selectors returned `LESSON_IDS: NONE` for the round tasks, and `.humanize/bitlesson.md` had no concrete project lessons to apply. No durable lesson was added this round because the changes were specific to the vector-add bring-up rather than a reusable process correction.
