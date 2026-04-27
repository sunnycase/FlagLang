# Round 1 Summary

## Work Completed
- Replaced the name-only CUDA vector-add escape hatch with a structural vector-add contract:
  - `ast_to_ttir` now parses once, recognizes the vector-add AST shape, and attaches a `_flaglang_vector_add` descriptor to the native module.
  - The descriptor records entry name, parameter order, pointer roles, dtype, `BLOCK_SIZE`, lane domain, affine relation, mask constraint, masked loads, compute op, and scatter destination.
  - The CUDA backend now rejects native modules that do not carry a validated vector-add descriptor, including kernels named `add_kernel` that do not implement vector add.
  - PTX emission derives the launch math from the recognized descriptor, including non-default `BLOCK_SIZE` values.
- Hardened the Python backend artifact and launcher contract:
  - `CompiledKernel` now rejects missing required launcher metadata before device launch.
  - `cluster_dims` must be a length-3 list/tuple and is normalized back to a tuple after JSON load.
  - target metadata must contain `backend`, `arch`, and `warp_size`.
  - generated metadata now includes a compact `flaglang_kernel` record for the validated vector-add path.
- Made native module cache/dump artifacts useful:
  - `.ttir`, `.ttgir`, and `.llir` storage now uses deterministic text for recognized FlagLang native modules instead of Python object repr strings.
  - The text includes the entry name, parameter order, affine lane relation, mask constraint, gather/scatter markers, and descriptor JSON.
- Expanded lowering and contract tests:
  - Added full-block and tail-block semantic checks for masked symbolic affine gather/scatter lowering.
  - Added launcher metadata negative tests for missing fields, malformed `cluster_dims`, and malformed target metadata.
  - Added backend negative tests for non-PTX cubin inputs and non-vector-add kernels named `add_kernel`.
  - Added a non-default `BLOCK_SIZE=128` warmup test that checks metadata, PTX math, and textual IR dumps.
- Ran task13 via ask-codex and integrated its recommendations:
  - Ask-Codex output: `.humanize/skill/2026-04-27_08-46-38-1967104-336eb9f7/output.md`.
  - Added the recommended metadata malformed-target and malformed-cluster negative checks.

## Files Changed
- `.humanize/rlcr/2026-04-27_07-06-00/state.md` - RLCR gate state update from the previous blocked gate.
- `.humanize/rlcr/2026-04-27_07-06-00/round-0-review-prompt.md` - generated review prompt.
- `.humanize/rlcr/2026-04-27_07-06-00/round-0-review-result.md` - generated review result.
- `.humanize/rlcr/2026-04-27_07-06-00/round-1-prompt.md` - generated round-1 prompt.
- `.humanize/rlcr/2026-04-27_07-06-00/round-1-summary.md` - this summary.
- `.humanize/skill/2026-04-27_08-46-38-1967104-336eb9f7/` - task13 ask-codex input, output, and metadata.
- `python/triton/compiler/code_generator.py` - structural vector-add descriptor recognition.
- `python/triton/compiler/compiler.py` - deterministic native IR dump text and launcher metadata validation.
- `third_party/nvidia/backend/compiler.py` - descriptor-validated vector-add PTX emission and metadata.
- `python/test/unit/runtime/test_flaglang_cuda_backend.py` - CUDA backend and launcher contract tests.
- `src/Nncase.Tests/TIR/UnitTestNTTAffineIOLowering.cs` - full/tail semantic lowering tests.

## Validation
- Passed: `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q`
  - Result: `8 passed in 2.53s` in the final rerun.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"`
  - Result: `5 passed`.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"`
  - Result: `6 passed`.
- Passed: `dotnet build`
  - Result: build succeeded with `0 Warning(s), 0 Error(s)`.
- Passed: `git diff --check`.
- Passed import and runtime dependency checks:
  - `conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"`
    printed paths under `/mnt/home-nas/work/repo/FlagLang/python/triton`.
  - `ldd python/triton/_C/libtriton.so | rg "libnncaseruntime|not found"` resolved `libnncaseruntime.so` from this checkout.
  - `conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` printed `2.11.0+cu128 True`.
- Passed forced tutorial unit test:
  - `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test`
  - Result: max difference `0.0`.
- Passed dump artifact checks:
  - `/tmp/flaglang-vector-add-dump` contains fresh `add_kernel.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin`, and `.sass`.
  - `rg -n "entry: add_kernel|ntt\.affine\.gather|ntt\.affine\.scatter" /tmp/flaglang-vector-add-dump` found the expected text markers in `.ttir`, `.ttgir`, and `.llir`.
  - The dump files do not contain `<triton._C.libtriton.ir.module object` placeholders.
- Passed forced full tutorial benchmark:
  - `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump conda run -n flaglang python python/tutorials/01-vector-add.py`
  - Result: max difference `0.0`; benchmark table completed through size `134217728`.
- Note: one earlier parallel run of the two targeted `dotnet test` filters hit an MSBuild/Fody PDB file-lock failure. Both filters passed when rerun serially, so this is treated as test-runner contention rather than a product failure.

## Remaining Items
- The native-module-to-PTX path remains intentionally narrow: it supports structurally recognized float32 vector add only. Unsupported native modules now fail explicitly instead of silently producing a canned cubin.
- I did not rerun `TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v` in this round because round 1 did not touch C++ or native binding sources; the existing editable install was revalidated by import, `ldd`, pytest warmup, and forced tutorial execution.

## BitLesson Delta
- Action: none
- Lesson ID(s): NONE
- Notes: `.humanize/bitlesson.md` contains only the entry template and no reusable lessons. Required round-1 selector calls for task10, task12, and task13 were attempted, but timed out after 120 seconds; because the knowledge base is empty, no lesson IDs were applied and no new lesson met the reusable-entry threshold.

## Goal Tracker Update Request

### Requested Changes:
- Mark task10 re-verified in round 1 with evidence from the new full-block and tail-block semantic gather/scatter tests:
  - `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` passed 5 tests.
- Mark task12 re-verified in round 1 with evidence from structural vector-add descriptor validation, name-only rejection, non-default block-size PTX generation, launcher metadata checks, and forced tutorial execution:
  - `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q` passed 8 tests.
  - Forced `01-vector-add.py --only_unit_test` and full `01-vector-add.py` benchmark both passed with max difference `0.0`.
- Mark task13 re-verified in round 1 with evidence from the final validation commands listed above and the task13 ask-codex output:
  - `.humanize/skill/2026-04-27_08-46-38-1967104-336eb9f7/output.md`.
- Update Open Issues to close the round-0 findings about:
  - name-only `add_kernel` cubin emission,
  - missing full/tail masked symbolic lowering semantics,
  - missing negative launcher-contract tests,
  - object-repr IR dump artifacts.

### Justification:
Round 1 directly addressed all round-0 review findings. The backend no longer accepts a kernel by name alone, the descriptor ties PTX emission to recognized vector-add structure and extracted block size, launcher metadata failures are tested before launch, NTT masked symbolic lowering now has semantic full/tail coverage, and dump artifacts contain deterministic affine/NTT markers instead of object repr placeholders.
