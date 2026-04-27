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
# Round 0 Review Result

## Findings

### [P0] AC-6 is not met: `cubin` generation ignores the lowered IR and emits a canned `add_kernel`

The current CUDA backend does not lower the nncase/NTT/native module into PTX. `make_ttgir` and `make_llir` return the module unchanged, and `make_ptx` accepts any non-string module whose entry name is exactly `add_kernel`, then emits fixed vector-add PTX from `_emit_vector_add_ptx`:

- `third_party/nvidia/backend/compiler.py:328-437`
- `third_party/nvidia/backend/compiler.py:506-515`
- `third_party/nvidia/backend/compiler.py:128-178`

This means the successful tutorial run is not an end-to-end validation of the Triton frontend, affine load/store rewrite, NTT/TIR lowering, and CUDA codegen path. A different kernel named `add_kernel`, a different operation, a different mask/default behavior, a different argument contract, or a different `BLOCK_SIZE` can be compiled to the same hard-coded x+y PTX. The PTX also hard-codes `1024` at `third_party/nvidia/backend/compiler.py:155-158` instead of deriving the block size from the compiled IR or metadata.

This directly contradicts the plan's AC-6 and Ultimate Goal, which require the Python backend and nncase/NTT codegen contract to be explicit and end-to-end. It also makes the AC-5 implementation effectively disconnected from the tutorial acceptance path.

Directive implementation plan:

1. Remove the unconditional `name == "add_kernel"` success path from `make_ptx`.
2. Add a structural vector-add native-module recognizer that inspects the compiled module after `add_optimize_ttir`. It must verify exactly the expected vector-add shape: three pointer args plus `n_elements`, two masked affine gathers, one floating add, one masked affine scatter, a single pointer base per IO op, and the shared affine relation `program_id(0) * BLOCK_SIZE + d0` guarded by `< n_elements`.
3. Extract `BLOCK_SIZE`, dtype, parameter order, mask relation, and default-value semantics from that recognized IR. Render PTX from those extracted values only after validation succeeds.
4. Reject unsupported native modules with a clear error before `ptxas`; do not silently produce a cubin for name-only matches.
5. Add CUDA tests that prove the recognizer rejects a non-vector-add `add_kernel` and that vector add works for at least the tutorial `BLOCK_SIZE=1024` plus one non-1024 block size.
6. Keep the implementation narrow to vector add, but make it structurally tied to the actual lowered IR rather than to the function name.

### [P1] AC-5 lowering tests do not cover the required full-block and tail-block semantics

`src/Nncase.Tests/TIR/UnitTestNTTAffineIOLowering.cs:23-70` only asserts that lowering creates an `IfThenElse`, a load in the then branch, a default buffer load in the else branch, and no scatter else write. It does not execute or otherwise validate full-block and tail-block behavior, nor does it prove the generated constraints bind `program_id(0)` and `n_elements` correctly for both in-bounds and masked lanes.

The plan explicitly requires full block and tail block coverage for masked symbolic affine IO. Structural checks alone can pass even if the generated address, symbol binding, or tail predicate is wrong.

Directive implementation plan:

1. Add two lowering tests for gather and scatter using the same relation shape: one full block where `program_id(0) * BLOCK_SIZE + d0 < n_elements` is true for every lane, and one tail block where only a suffix is masked off.
2. Validate the lowered TIR semantically, not just structurally. Either run the lowered function through the available TIR evaluator/codegen harness, or inspect the generated C source to assert concrete per-lane load/default/store behavior for the chosen `program_id` and `n_elements`.
3. For gather, assert false lanes read `other/defaultValue`. For scatter, assert false lanes perform no destination write.
4. Keep the existing symbol-payload mismatch test, but add an assertion that the rejected path names the symbol mismatch clearly enough to diagnose.

### [P1] AC-6 negative launcher-contract tests are missing

The plan requires failures when a backend returns a `cubin` without required metadata (`name`, `shared`, `num_warps`, `cluster_dims`, etc.) and when a non-`cubin` artifact such as `kmodel` is returned. I did not find tests covering these negative contract paths. The only effective validation is the happy-path tutorial execution.

Directive implementation plan:

1. Add a targeted Python test around `CompiledKernel`/CUDA backend metadata loading that constructs or monkeypatches a metadata group missing each required field and asserts launch initialization fails before device launch.
2. Add a backend-stage test that makes `make_cubin` receive a non-PTX/non-string artifact and asserts the current explicit `TypeError` is raised.
3. Add a test that a successful vector-add compile writes metadata containing at least `name`, `shared`, `num_warps`, `num_ctas`, `cluster_dims`, and scratch fields, and that `cluster_dims` round-trips through JSON as a tuple in `CompiledKernel`.

### [P2] Dump artifacts claimed as fresh IR are not useful IR dumps

The generated `/tmp/flaglang-vector-add-dump/.../add_kernel.ttir`, `.ttgir`, and `.llir` files contain only Python object repr strings like `<triton._C.libtriton.ir.module object at ...>`, not textual IR. The `.ptx`, `.cubin`, and `.sass` artifacts are real, but the earlier-stage dumps cannot be used to verify affine rewrite, NTT lowering, or regression triage.

This weakens the evidence for AC-6 and task13. It also hides the P0 issue above because the dump files do not show what IR was actually fed into the PTX stage.

Directive implementation plan:

1. Implement a textual serializer or `__str__`/dump method for the native module object used by the Python compiler cache/dump path.
2. Ensure `add_kernel.ttir`, `add_kernel.ttgir`, and `add_kernel.llir` contain deterministic text for the actual stage output.
3. Add a lightweight assertion in the forced tutorial validation that these files are not object repr placeholders and include the entry function name plus affine/NTT markers expected for the stage.

## Goal Alignment Summary

ACs: 4/6 fully addressed; AC-5 and AC-6 are incomplete. Forgotten items: 3. Unjustified deferrals: 1.

- AC-1: Mostly addressed by import/build checks, but the stale-editable and missing-runtime negative acceptance checks are not automated.
- AC-2: Addressed by CUDA PyTorch verification.
- AC-3: Mostly addressed for the current vector-add builder surface; the stale `passes.cpp` path is deleted and smoke coverage exists.
- AC-4: Addressed by targeted affine rewrite tests including symbol/domain and invalid-affine cases.
- AC-5: Partially addressed; masked symbolic lowering exists, but full/tail semantic coverage is missing.
- AC-6: Incomplete; the tutorial passes through a canned PTX emitter keyed by `add_kernel`, not through a validated nncase/NTT-derived codegen contract.

No "Goal Tracker Update Request" section was present in Claude's summary, so I did not modify `goal-tracker.md`. The tracker currently marks all tasks completed, but the review findings above mean task10/task12/task13 should be treated as not verified until the fixes are implemented and re-reviewed.
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
2. Write your work summary into @/mnt/home-nas/work/repo/FlagLang/.humanize/rlcr/2026-04-27_07-06-00/round-1-summary.md

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
