# Round 3 Summary

## Work Completed
- Hardened `_inspect_vector_add_native_module` so `make_ptx` only calls `describe_vector_add()` on actual native `triton._C.libtriton.ir.module` objects. Forged Python objects that return valid-looking descriptor JSON are now rejected before PTX emission.
- Added a negative Python test for a forged `describe_vector_add()` payload while preserving the existing fake `_flaglang_vector_add` attribute rejection test.
- Tightened the native vector-add descriptor validation:
  - requires exactly one `IR.Math.Binary Add`,
  - requires exactly one affine scatter,
  - validates that the single scatter stores the add result to the output pointer,
  - replaces the problem-size substring test with an exact structural check for `program_id(0) * BLOCK_SIZE + d0 < problem-size`.
- Added managed negative tests that construct lowered affine modules directly and verify descriptor rejection for extra affine scatter side effects and a mismatched `<=` mask constraint.

## Files Changed
- `third_party/nvidia/backend/compiler.py`
- `python/test/unit/runtime/test_flaglang_cuda_backend.py`
- `python/test/unit/runtime/test_flaglang_native_ir.py`
- `src/Nncase.Compiler/Interop/CApi.IR.cs`
- `src/Nncase.Tests/Transform/UnitTestTensorizeIO.cs`
- RLCR-generated state/review artifacts under `.humanize/rlcr/2026-04-27_07-06-00/`

## Validation
- Passed: `dotnet build` -> 0 warnings, 0 errors.
- Passed: `TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v`.
- Passed: `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q` -> 10 passed.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` -> 8 passed.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` -> 5 passed.
- Passed: import check printed `/mnt/home-nas/work/repo/FlagLang/python/triton/__init__.py` and `/mnt/home-nas/work/repo/FlagLang/python/triton/_C/libtriton.so`.
- Passed: `ldd python/triton/_C/libtriton.so | rg "libnncaseruntime|not found|libcoreclr|libhostfxr"` shows `libnncaseruntime.so` resolved from this checkout.
- Passed locally: CUDA dependency check printed `2.11.0+cu128 True`; `nvidia-smi` reported `NVIDIA GeForce RTX 5060 Ti, 580.126.09`.
- Passed locally: forced unit tutorial with fresh dumps:
  `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test`
  -> maximum difference `0.0`.
- Passed locally: forced full tutorial benchmark:
  `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py`
  -> maximum difference `0.0`, benchmark table printed through size `134217728.0`.
- Passed: fresh dump directory contains `add_kernel.ttir`, `add_kernel.ttgir`, `add_kernel.llir`, `add_kernel.ptx`, `add_kernel.cubin`, and `add_kernel.sass`; `rg` found real `Gather((d0)[s0, s1]` and `Scatter((d0)[s0, s1]` relation lines in all textual stages and no descriptor/object-repr markers.
- Passed: `git diff --check`.

## Remaining Items
- No code issues remain from round 3.
- CUDA validation still depends on the review process having a working NVIDIA driver. I cannot restore the separate review environment's driver from this repository patch; the local `flaglang` environment for this checkout has CUDA available and passes the forced tutorial commands.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: `.humanize/bitlesson.md` contains no entries. The required selectors for task12 and task13 both timed out after 120s with no output, so no lessons were applied or added.

## Code Simplifier
- The `code-simplifier` plugin is not installed in this session, so no code-simplifier pass was run.

## Goal Tracker Update Request

### Requested Changes:
- Mark task12 completed/re-verified with evidence: non-native forged `describe_vector_add()` objects are rejected, native descriptor validation now enforces exact add/scatter count and exact mask structure, and targeted Python/C# tests pass.
- Mark task13 locally re-verified with evidence: import/linking checks passed, CUDA is available in the current local `flaglang` environment, forced unit tutorial passed with max difference `0.0`, full benchmark passed, and fresh dump artifacts contain real lowered `Gather`/`Scatter` stage text.
- Close the open issues for forged `describe_vector_add()` acceptance and non-exact descriptor validation.
- If the reviewer environment still lacks a working NVIDIA driver, keep only the CUDA environment blocker open as an external validation-environment issue, not as a code issue.

### Justification:
The remaining task12 AC-6 code gaps from round 2 are directly addressed and covered by negative tests. Task13 has been rerun successfully in the current checkout and environment; any continued CUDA skip in the review environment reflects driver availability outside the repository changes.
