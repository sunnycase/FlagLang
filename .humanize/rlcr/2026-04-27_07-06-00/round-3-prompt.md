Your work is not finished. Read and execute the below with ultrathink.

## Original Implementation Plan

**IMPORTANT**: Before proceeding, review the original plan you are implementing:
@docs/plan/01-vector-add.md

This plan contains the full scope of work and requirements. Ensure your work aligns with this plan.

---

For all tasks that need to be completed, please use the Task system (TaskCreate, TaskUpdate, TaskList) to track each item in order of importance.
You are strictly prohibited from only addressing the most important issues - you MUST create Tasks for ALL discovered issues and attempt to resolve each one.

Before executing each task in this round:
1. Read @/mnt/home-nas/work/repo/FlagLang/.humanize/bitlesson.md
2. Run `bitlesson-selector` for each task/sub-task
3. Follow selected lesson IDs (or `NONE`) during implementation

---
Below is Codex's review result:
<!-- CODEX's REVIEW RESULT START -->
# Round 2 Review Result

## Findings

### [P1] `make_ptx` still accepts a forged vector-add descriptor from a non-native Python object

`third_party/nvidia/backend/compiler.py:138-159` only checks whether `src.describe_vector_add` is callable, then trusts the JSON it returns. That means the backend no longer trusts `_flaglang_vector_add`, but it still does not prove the descriptor came from an actual post-`make_ttir` native module. The new fake descriptor test in `python/test/unit/runtime/test_flaglang_cuda_backend.py:69-81` covers only objects without `describe_vector_add`, so it misses this case.

I confirmed this during review with a fake object that implemented `get_entry_func_name()` and returned a syntactically valid `{"valid": true, "descriptor": ...}` payload from `describe_vector_add()`. `backend.make_ptx(...)` emitted PTX beginning with `.version 8.7` and populated `metadata["flaglang_kernel"]`.

Directive implementation plan:

1. In `_inspect_vector_add_native_module`, require `src` to be an actual `triton._C.libtriton.ir.module` instance before calling `describe_vector_add()`.
2. Add a negative test with a fake object that implements `describe_vector_add()` and returns a fully valid descriptor JSON; `make_ptx` must reject it before PTX emission.
3. Keep the existing fake `_flaglang_vector_add` attribute test.
4. Re-run `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q`.

### [P1] AC-2 and AC-6 CUDA validation is still not reproducible in this review environment

Claude claims CUDA validation passed, but the current review environment does not reproduce it:

- `conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` printed `2.11.0+cu128 False`.
- `nvidia-smi --query-gpu=name,driver_version --format=csv,noheader` failed because it could not communicate with the NVIDIA driver.
- `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q` produced `7 passed, 2 skipped`, not 9 passed; the CUDA-dependent tests skipped.

Task13 cannot be accepted until CUDA is available from this checkout and the forced unit tutorial plus full benchmark run with `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1`.

Directive implementation plan:

1. Restore CUDA driver/runtime availability for the `flaglang` environment.
2. Re-run the CUDA PyTorch check and require `torch.cuda.is_available() == True`.
3. Re-run the CUDA pytest set without skips.
4. Re-run the forced `python/tutorials/01-vector-add.py --only_unit_test` and full benchmark with a fresh `TRITON_DUMP_DIR`.
5. Verify fresh `.ttir`, `.ttgir`, `.llir`, `.ptx`, `.cubin`, and `.sass` artifacts from this checkout.

### [P2] Native descriptor validation is not exact enough for the claimed supported shape

`src/Nncase.Compiler/Interop/CApi.IR.cs:576-586` counts affine gathers exactly, but `RequireSingleScatter` at `src/Nncase.Compiler/Interop/CApi.IR.cs:689-697` only checks that one matching scatter exists. It does not reject additional affine scatters. The mask check at `src/Nncase.Compiler/Interop/CApi.IR.cs:719-721` also uses a substring match against the constraint text rather than validating the exact expected constraint. This falls short of the plan and summary claim that the lowered module shape is exactly two gathers, one fadd, and one scatter with the shared `program_id(0) * BLOCK_SIZE + d0 < n_elements` mask.

Directive implementation plan:

1. In `BuildVectorAddDescriptor`, collect all affine scatter calls and require exactly one.
2. Validate that the single scatter is the add result to the output pointer.
3. Replace the constraint substring check with an exact structural check for `program_id(0) * BLOCK_SIZE + d0 < n_elements`.
4. Add focused negative coverage for extra scatter side effects and mismatched constraints.

## Goal Alignment Summary

ACs: 4/6 addressed | Forgotten items: 0 | Unjustified deferrals: 0

- AC-1: Import path and `ldd` checks passed in this review.
- AC-2: Blocked; CUDA is currently unavailable.
- AC-3: No new concerns found.
- AC-4: Targeted TensorizeIO test passed in this review.
- AC-5: Targeted NTT affine IO lowering test passed in this review.
- AC-6: Incomplete; forged descriptor acceptance remains and CUDA tutorial validation is not reproducible.

## Goal Tracker Update Decision

Claude's requested update is rejected for task12 and task13 completion.

Approved:
- The AST-created `_flaglang_vector_add` authority has been removed.
- Descriptor-generated fake dump text has been removed; `_serialize_ir_for_storage` now calls native `to_text()` when available.

Rejected:
- Do not mark task12 complete because `make_ptx` accepts forged `describe_vector_add()` JSON from non-native objects, and native validation still needs exact-shape tightening.
- Do not mark task13 complete because CUDA validation is not reproducible in this review environment.
- Do not close the CUDA validation blocker.

I updated `goal-tracker.md` mutable sections for round 2: task12 and task13 remain active, the stale fake-dump open issue was removed, and new AC-6 open issues were added for forged `describe_vector_add()` acceptance and non-exact descriptor validation.

## Validation Run During Review

- Passed: import check printed `/mnt/home-nas/work/repo/FlagLang/python/triton/__init__.py` and `/mnt/home-nas/work/repo/FlagLang/python/triton/_C/libtriton.so`.
- Passed: `ldd python/triton/_C/libtriton.so | rg "libnncaseruntime|not found|libcoreclr|libhostfxr"` resolved `libnncaseruntime.so` from this checkout.
- Passed with skips: `conda run -n flaglang python -m pytest python/test/unit/runtime/test_flaglang_cuda_backend.py python/test/unit/runtime/test_flaglang_native_ir.py -q` -> `7 passed, 2 skipped`.
- Passed: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"` -> `5 passed`.
- Passed after serial rerun: `dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"` -> `6 passed`.
- Passed: `git diff --check`.
- Blocked: forced tutorial unit test and full benchmark because CUDA is unavailable.
<!-- CODEX's REVIEW RESULT  END  -->
---

## Goal Tracker Reference (READ-ONLY after Round 0)

Before starting work, **read** @/mnt/home-nas/work/repo/FlagLang/.humanize/rlcr/2026-04-27_07-06-00/goal-tracker.md to understand:
- The Ultimate Goal and Acceptance Criteria you're working toward
- Which tasks are Active, Completed, or Deferred
- Any Plan Evolution that has occurred
- Open Issues that need attention

**IMPORTANT**: You CANNOT directly modify goal-tracker.md after Round 0.
If you need to update the Goal Tracker, include a "Goal Tracker Update Request" section in your summary (see below).

---

Note: You MUST NOT try to exit by lying, editing loop state files, or executing `cancel-rlcr-loop`.

After completing the work, please:
0. If the `code-simplifier` plugin is installed, use it to review and optimize your code. Invoke via: `/code-simplifier`, `@agent-code-simplifier`, or `@code-simplifier:code-simplifier (agent)`
1. Commit your changes with a descriptive commit message
2. Write your work summary into @/mnt/home-nas/work/repo/FlagLang/.humanize/rlcr/2026-04-27_07-06-00/round-3-summary.md

## Task Tag Routing Reminder

Follow the plan's per-task routing tags strictly:
- `coding` task -> Claude executes directly
- `analyze` task -> execute via `/humanize:ask-codex`, then integrate the result
- Keep Goal Tracker Active Tasks columns `Tag` and `Owner` aligned with execution

**If Goal Tracker needs updates**, include this section in your summary:
```markdown
## Goal Tracker Update Request

### Requested Changes:
- [E.g., "Mark Task X as completed with evidence: tests pass"]
- [E.g., "Add to Open Issues: discovered Y needs addressing"]
- [E.g., "Plan Evolution: changed approach from A to B because..."]
- [E.g., "Defer Task Z because... (impact on AC: none/minimal)"]

### Justification:
[Explain why these changes are needed and how they serve the Ultimate Goal]
```

Codex will review your request and update the Goal Tracker if justified.
