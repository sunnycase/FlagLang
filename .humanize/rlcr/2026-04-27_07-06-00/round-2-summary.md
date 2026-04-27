# Round 2 Summary

## Work Completed
- Replaced the AST-derived vector-add authority with actual post-`make_ttir` native module inspection. The backend now calls `describe_vector_add()` on the lowered module in both `make_ttir` and `make_ptx`; fake `_flaglang_vector_add` / `_flaglang_validated_vector_add` attributes are not trusted.
- Added CLR/native/Python bridge APIs for real module printing and vector-add structural validation: entry lookup, text serialization, pass-manager result rebinding, and JSON descriptor generation from the lowered dataflow `Function`.
- Validated the lowered native module shape: four parameters, three scalar float32 pointers plus scalar problem size, two affine gathers, one fadd, one affine scatter, shared `program_id(0) * BLOCK_SIZE + d0` relation, problem-size constraint, pointer roles, dtype, and implicit-zero masked load defaults.
- Removed descriptor-generated fake IR dumps. Stage dumps now call the native module serializer and contain the real lowered function body with `Gather` / `Scatter` relation lines.
- Updated tests to cover fake descriptor rejection, actual native entry-name based dump checks, and real stage text instead of descriptor markers.

## Files Changed
- `src/Nncase.Compiler/Interop/CApi.cs`
- `src/Nncase.Compiler/Interop/CApi.Compile.cs`
- `src/Nncase.Compiler/Interop/CApi.IR.cs`
- `src/Native/include/nncase/compiler.h`
- `python/flaglang/native/ir.cpp`
- `python/triton/compiler/code_generator.py`
- `python/triton/compiler/compiler.py`
- `third_party/nvidia/backend/compiler.py`
- `python/test/unit/runtime/test_flaglang_cuda_backend.py`
- RLCR-generated state/review artifacts under `.humanize/rlcr/2026-04-27_07-06-00/`

## Validation
- Passed: `dotnet build` -> 0 warnings, 0 errors.
- Passed: `TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v`.
- Passed: `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q` -> 9 passed.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` -> 5 passed.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` -> 6 passed.
- Passed: import check printed `/mnt/home-nas/work/repo/FlagLang/python/triton/__init__.py` and `/mnt/home-nas/work/repo/FlagLang/python/triton/_C/libtriton.so`.
- Passed: `ldd python/triton/_C/libtriton.so | rg "libnncaseruntime|not found|libcoreclr|libhostfxr"` shows `libnncaseruntime.so` resolved from this checkout.
- Passed: CUDA dependency check printed `2.11.0+cu128 True`; `nvidia-smi` reported `NVIDIA GeForce RTX 5060 Ti, 580.126.09`.
- Passed: forced unit tutorial with fresh dumps:
  `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test`
  -> maximum difference `0.0`.
- Passed: forced full tutorial benchmark:
  `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py`
  -> maximum difference `0.0`, benchmark table printed through size `134217728.0`.
- Passed: fresh dump directory contains `add_kernel.ttir`, `add_kernel.ttgir`, `add_kernel.llir`, `add_kernel.ptx`, `add_kernel.cubin`, and `add_kernel.sass`; `rg` found real `Gather((d0)[s0, s1]` and `Scatter((d0)[s0, s1]` relation lines in all textual stages and no descriptor/object-repr markers.
- Passed: `git diff --check`.

## Remaining Items
- No unresolved implementation items from round 2.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: `.humanize/bitlesson.md` contains no entries. The required selectors for task12 and task13 each timed out after 120s and produced no selected lesson IDs, so no lessons were applied or added.

## Code Simplifier
- The `code-simplifier` plugin is not installed in this session, so no code-simplifier pass was run.

## Goal Tracker Update Request

### Requested Changes:
- Mark task12 completed/re-verified with evidence: backend PTX emission now derives the descriptor from actual post-TTIR native module inspection; fake descriptor tests reject objects without `describe_vector_add`; Python tests pass.
- Mark task13 completed/re-verified with evidence: import/linking checks passed, CUDA is available, forced unit tutorial passed with max difference `0.0`, full benchmark passed, and fresh dump artifacts contain real lowered `Gather`/`Scatter` stage text.
- Close the open issues for AST-created descriptor trust and descriptor-summary fake dumps.
- Close the CUDA validation blocker for this environment because CUDA validation is now reproducible on the current checkout.

### Justification:
The active AC-6 gaps from round 1 are addressed by binding emission and dumps to the actual lowered module rather than Python AST spelling. AC-2 and AC-6 CUDA validation also ran successfully in the current `flaglang` environment with forced recompilation and fresh dump artifacts.
