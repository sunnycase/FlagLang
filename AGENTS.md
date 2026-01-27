# GPT-Codex Agents Manual

FlagLang is an AI compiler initiative that rewrites Triton's GPU backend on top of nncase (https://arxiv.org/html/2512.21571v1) while layering affine reasoning, shardy-aware scheduling, and a learned cost model. This document explains how GPT-Codex agents should plan, implement, and validate code and test changes inside the shared `conda` environment `flagtree` using `dotnet build` for C# components and `pip install -e .` for C++/Python extensions.

## Mission Profile
- Translate high-level optimization goals (affine fusion, shardy partitioning, mask peeling) into concrete Triton/nncase patches.
- Maintain consistency across C++, Python, and C# entrypoints so kernels can be authored, lowered, and validated end-to-end.
- Capture telemetry for the cost model by instrumenting kernels and feeding benchmark data back into auto-schedulers.

## Agent Operating Modes
1. **Design Sweep**
  - Inspect IR / pass structure, dump AxisInfo, and outline the transform strategy.
  - Produce a TODO plan before touching files; sync with existing ADRs.
2. **Code Execution**
  - Use the prescribed toolchain (`pip install -e .`, `dotnet build`) within `conda env:flagtree`.
  - Prefer small, reviewable patches with comments explaining non-obvious affine/shardy math.
3. **Test + Benchmark**
  - Run `dotnet test` for managed components, `pytest` or targeted C++ tests for native code.
  - Log kernel timing, FLOP/s, and memory metrics whenever a schedule or cost-model path is touched.

## Workflow Template
1. **Scope Intake** – Parse issue, identify affected stacks (C#, C++, Python), list expected files.
2. **Environment Sync** – `conda activate flagtree`, ensure `pip install -e .` succeeds, then `dotnet build`.
3. **Implementation Loop**
  - Modify IR passes (affine rewrite, shardy tiling, peeling) in C++.
  - Update Python front-end APIs or scheduling scripts that feed those passes.
  - Patch C# orchestration or tests so managed tooling understands new backend behaviors.
4. **Validation Loop**
  - Run language-specific unit tests; capture outputs for regression tracking.
  - Execute sample kernels to confirm cost-model deltas and dump telemetry.
5. **Handoff Package** – Summarize architecture impact, test results, and outstanding risks in PR notes.

## Best Practices for GPT-Codex
- Maintain a shared vocabulary: affine tile specs, shardy layout descriptors, and cost-model features must match in docs and code.
- Keep comments succinct but clarify tricky math (loop peeling constraints, vector load alignment, etc.).
- When editing generated or vendored files, document why and how to re-sync upstream.
- Prefer deterministic tests; when randomness is required, seed it and explain the choice.

## Reference Commands
- Build C#/tests: `dotnet build`, `dotnet test` (run inside `src/` solutions as needed).
- Build native stack: `pip install -e .` from repo root; rerun after touching C++ or Python extensions.
- Benchmark harness: `python tools/run_bench.py --suite <name>` (add new suites when introducing kernels).

## Onboarding Checklist
1. Clone the repo and `conda activate flagtree`.
2. Run `pip install -e .` to build C++/Python extensions.
3. Run `dotnet build` and `dotnet test` to verify managed layers.
4. Execute representative kernel benchmarks to populate baseline telemetry.
5. Study current affine/shardy ADRs plus the cost-model interface before attempting a new optimization.
