# Goal Tracker

<!--
This file tracks the ultimate goal, acceptance criteria, and plan evolution.
It prevents goal drift by maintaining a persistent anchor across all rounds.

RULES:
- IMMUTABLE SECTION: Do not modify after initialization
- MUTABLE SECTION: Update each round, but document all changes
- Every task must be in one of: Active, Completed, or Deferred
- Deferred items require explicit justification
-->

## IMMUTABLE SECTION
<!-- Do not modify after initialization -->

### Ultimate Goal
Run `python/tutorials/01-vector-add.py` end to end from the current FlagLang checkout by aligning the Triton Python frontend, nncase native builder bindings, affine load/store rewrites, NTT/TIR masked symbolic IO lowering, CUDA `cubin` artifact/launcher contract, and required CUDA runtime dependencies.

Source plan: docs/plan/01-vector-add.md

### Acceptance Criteria
<!-- Each criterion must be independently verifiable -->
<!-- Claude must extract or define these in Round 0 -->

1. The editable native/Python Triton package imports from `/mnt/home-nas/work/repo/FlagLang/python/triton`, resolves `libnncaseruntime.so`, and rejects stale editable installs or missing runtime library paths.
2. The `flaglang` environment has CUDA-enabled PyTorch available, with `torch.cuda.is_available() == True`, so the tutorial reaches compilation rather than failing during dependency setup.
3. Python/native builder bindings expose the minimal vector-add API surface, keep Triton IR registration consistent, and have smoke coverage that fails if required builder methods disappear.
4. Triton load/store calls rewrite to typed affine gather/scatter with correct shape/type inference, lane domains, program-id and problem-size symbols, pointer-base handling, and negative coverage for invalid affine/mask cases.
5. NTT/TIR lowering supports masked symbolic affine gather/scatter semantics for full and tail blocks, including load defaults, store no-ops, and rejection of inconsistent symbol payloads.
6. The Python backend emits a real CUDA `cubin` with complete launcher metadata, forced recompilation of `01-vector-add.py --only_unit_test` produces `max error = 0.0`, and the full tutorial benchmark completes with fresh dump artifacts.

---

## MUTABLE SECTION
<!-- Update each round with justification for changes -->

### Plan Version: 3 (Updated: Round 2)

#### Plan Evolution Log
<!-- Document any changes to the plan with justification -->
| Round | Change | Reason | Impact on AC |
|-------|--------|--------|--------------|
| 0 | Initial plan | - | - |
| 1 | Accepted task10 re-verification; rejected task12/task13 completion request | Round 1 adds full/tail masked affine IO lowering tests, but the CUDA backend still trusts an AST-created descriptor rather than validating the post-`add_optimize_ttir` native/affine/NTT IR, and current review could not reproduce CUDA runtime availability | AC-5 re-verified; AC-6 remains active; AC-2/AC-6 GPU validation remains blocked in the current review environment |
| 2 | Partially accepted native module serialization changes; rejected task12/task13 completion request | Round 2 removes the AST recognizer and descriptor-generated fake dumps, but `make_ptx` still accepts any Python object that forges `describe_vector_add()` JSON instead of proving the object is a real native `ir.module`; current review also still cannot reproduce CUDA availability | AC-6 remains active; AC-2/AC-6 GPU validation remains blocked in the current review environment |

#### Active Tasks
<!-- Map each task to its target Acceptance Criterion and routing tag -->
| Task | Target AC | Status | Tag | Owner | Notes |
|------|-----------|--------|-----|-------|-------|
| task12 | AC-6 | active | coding | Claude | Rework CUDA `cubin` emission so PTX is generated only after inspecting an actual native `triton._C.libtriton.ir.module` from the post-`make_ttir` pipeline. Reject forged Python objects that merely implement `describe_vector_add()` and add a negative test for that case. Tighten native validation to enforce the exact supported shape: exactly two affine gathers, exactly one fadd, exactly one affine scatter, shared relation/constraint, pointer roles, dtype, and default semantics. |
| task13 | AC-1, AC-2, AC-4, AC-6 | active | analyze | Claude via ask-codex | Re-run final validation after task12, including forced tutorial unit test and benchmark with CUDA available, plus dump checks that prove `.ttir`, `.ttgir`, and `.llir` are actual stage output from the current checkout. |

### Completed and Verified
<!-- Only move tasks here after Codex verification -->
| AC | Task | Completed Round | Verified Round | Evidence |
|----|------|-----------------|----------------|----------|
| AC-1 | task1: Locate and record `libnncaseruntime.so` search path and `nncase::clr::*` symbol visibility state | 0 | 0 | ask-codex output `.humanize/skill/2026-04-27_07-10-43-1829840-2f688e67/output.md`; local `ldd` showed `libnncaseruntime.so => not found`; `LD_LIBRARY_PATH=build/.../Debug/lib` import exposed hidden `nncase::clr::tensor_type` symbols |
| AC-1 | task2: Fix native package/rpath/export/static-link path so `import triton` comes from this checkout | 0 | 0 | `TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v` succeeded; `conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; ..."` printed current checkout paths; `ldd python/triton/_C/libtriton.so` resolves `libnncaseruntime.so` from `python/triton/_C` |
| AC-2 | task3: Verify CUDA-enabled PyTorch and record tutorial dependency check command | 0 | 0 | ask-codex output `.humanize/skill/2026-04-27_07-17-48-1839632-4b50959f/output.md` showed missing torch; installed `torch-2.11.0+cu128`; removed PyTorch's `triton` wheel so editable FlagLang wins; `conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints `2.11.0+cu128 True` |
| AC-3 | task4: Fill vector-add builder API gaps in `python/flaglang/native/ir.cpp` | 0 | 0 | Forced tutorial unit-test now reaches C# affine rewrite; failure moved from native builder/import dependency setup to `LoadToAffineGather.GetReplace`, showing vector-add builder construction path is present |
| AC-3 | task5: Merge or remove stale `init_triton_ir` surface in `python/flaglang/native/passes.cpp` | 0 | 0 | Deleted stale `python/flaglang/native/passes.cpp`; `rg "void nncase::init_triton_ir|init_triton_ir\\(" python/flaglang/native` shows the single implementation in `ir.cpp` plus call/declaration sites |
| AC-3 | task6: Add Python/native builder smoke coverage for the vector-add API surface | 0 | 0 | Added `python/test/unit/runtime/test_flaglang_native_ir.py`; fixed insertion point end handling in `ir.cpp`; `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_native_ir.py -q` passed |
| AC-4 | task7: Fix checked type and inference behavior for affine load/store rewrites | 0 | 0 | `LoadToAffineGather` now uses the matched call shape/type; `StoreToAffineScatter` inherits call metadata and checked type; `dotnet build` passed; `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` passed; forced tutorial progressed to backend `make_cubin` receiving a native IR module |
| AC-4 | task8: Extend Triton affine domain/symbol/range modeling and relation assertions | 0 | 0 | BitLesson selector returned `NONE`; scalar problem sizes now become affine symbols while `tl.arange` lanes become bounded domains; relation assertions cover vector-add `program_id(0) * BLOCK_SIZE + d0`, `n_elements`, pointer base preservation, multi pointer base rejection, nonlinear mask rejection, and unbounded dynamic lane rejection; `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` passed 6 tests |
| AC-5 | task9: Determine whether existing TIR builder favors direct masked loops or mask peeling | 0 | 0 | BitLesson selector returned `NONE`; ask-codex output `.humanize/skill/2026-04-27_08-03-01-1899926-a73994ce/output.md` recommended direct `T.If`-guarded per-lane loops using existing `T.Serial`, `T.Load`, `T.Store`, and buffer load/store APIs |
| AC-5 | task10: Implement masked symbolic affine gather/scatter lowering | 0 | 1 | BitLesson selector returned `NONE`; `NTTAffineIOLoweringPass` now binds affine symbols, lowers constraints into per-lane `T.If`, stores gather default values on false masks, makes scatter false masks no-op, and rejects symbol payload mismatches; round 1 added full-block and tail-block semantic checks for gather/scatter and mismatch message assertions; `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` passed 5 tests in review |
| AC-6 | task11: Define CUDA `cubin` artifact/launcher contract and affected Python call path | 0 | 0 | BitLesson selector returned `NONE`; ask-codex output `.humanize/skill/2026-04-27_08-13-34-1918390-d7fc9941/output.md` defined the PTX-to-cubin stage boundary and required launcher metadata keys |

### Explicitly Deferred
<!-- Items here require strong justification -->
| Task | Original AC | Deferred Since | Justification | When to Reconsider |
|------|-------------|----------------|---------------|-------------------|

### Open Issues
<!-- Issues discovered during implementation -->
| Issue | Discovered Round | Blocking AC | Resolution Path |
|-------|-----------------|-------------|-----------------|
| CUDA backend accepts forged `describe_vector_add()` JSON from non-native Python objects before PTX emission | 2 | AC-6 | In `_inspect_vector_add_native_module`, require the source object to be an actual `triton._C.libtriton.ir.module` before calling `describe_vector_add()`. Add a negative test with a fake object that returns a fully valid descriptor JSON and assert `make_ptx` rejects it. |
| Native vector-add descriptor validation does not explicitly reject extra affine scatter calls or non-exact mask constraints | 2 | AC-6 | In `BuildVectorAddDescriptor`, count all affine scatter calls and require exactly one; validate the relation constraint structurally/exactly instead of using a substring check; add focused negative coverage for extra side effects and mismatched constraints. |
| Current review environment cannot reproduce CUDA acceptance because `torch.cuda.is_available()` is `False` and `nvidia-smi` cannot communicate with the NVIDIA driver | 1 | AC-2, AC-6 | Restore CUDA driver/runtime availability for the `flaglang` environment, then rerun the import, CUDA PyTorch, forced tutorial unit test, full benchmark, and dump validation commands from task13. |
