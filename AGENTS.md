# GPT-Codex Agents Manual

FlagLang is an AI compiler initiative that rewrites Triton's GPU backend on top of nncase (https://arxiv.org/html/2512.21571v1) while layering affine reasoning, shardy-aware scheduling, and a learned cost model. This document explains how GPT-Codex agents should plan, implement, and validate code and test changes inside the shared `conda` environment `flaglang` using `dotnet build` for C# components and editable Python/native builds for C++/Python extensions.

## Mission Profile
- Translate high-level optimization goals (affine fusion, shardy partitioning, mask peeling) into concrete Triton/nncase patches.
- Maintain consistency across C++, Python, and C# entrypoints so kernels can be authored, lowered, and validated end-to-end.
- Capture telemetry for the cost model by instrumenting kernels and feeding benchmark data back into auto-schedulers.

## Agent Operating Modes
1. **Design Sweep**
  - Inspect IR / pass structure, dump AxisInfo, and outline the transform strategy.
  - Produce a TODO plan before touching files; sync with existing ADRs.
2. **Code Execution**
  - Use the prescribed toolchain inside `conda env:flaglang`.
  - Rebuild native/Python bindings with `TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v` from repo root unless Proton work explicitly requires enabling it.
  - Prefer small, reviewable patches with comments explaining non-obvious affine/shardy math.
3. **Test + Benchmark**
  - Run `dotnet test ... -s test.runsettings` for managed components, `pytest` or targeted C++ tests for native code.
  - Log kernel timing, FLOP/s, and memory metrics whenever a schedule or cost-model path is touched.

## Workflow Template
1. **Scope Intake** – Parse issue, identify affected stacks (C#, C++, Python), list expected files.
2. **Environment Sync** – `conda activate flaglang`, ensure the editable native build succeeds, then run `dotnet build`.
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
- ** YOU MUST ** Fail fast instead of adding fallback or workaround paths. When a problem appears, identify the root cause and fix it with a generalized, principled approach.

## Reference Commands
- Build C#: `dotnet build`.
- Test C#: `dotnet test <project-or-solution> -s test.runsettings` (always pass the runsettings file).
- Build native/Python stack: `TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v` from repo root; rerun after touching C++ or Python bindings.
- Python tutorial dependency: install CUDA-enabled `torch` in `flaglang` before running `python/tutorials/*.py`; the repository build requirements do not install it.
- Triton tutorial sanity check: verify `import triton` resolves to this repo, e.g. `python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"`.
- Force fresh Triton tutorial builds with `TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump/flaglang-vector-add-dump` so old cache artifacts do not hide backend issues and dumps stay inspectable under the repository `dump/` directory.
- NTT benchmark harness: use the generated benchmark binaries with `python ntt/test/benchmark_test/benchmark_ntt.py` after building the relevant NTT benchmark targets.

## Development Guide (Current Understanding)
- **Testing defaults**: Every `dotnet test` invocation must load `test.runsettings` (e.g., `dotnet test ... -s test.runsettings`) so CUDA backends, logging, and conditional theories pick up the expected environment knobs.
- **IR workstyle**: Keep Triton/nncase changes end-to-end—author IR samples, run them through the dataflow rewriters (via `CompilerServices.TryMatchRoot` for single-rule tests or `Rewrite` for full passes), and assert on emitted affine relations (domains, symbols, constraints) rather than just structural equality.
- **Dynamic dim safety**: When manipulating `DimVar` or other dimension carriers, route cloning through `ExprCloner`/`CompilerServices.InferenceType` to avoid invalid casts; always seed `Metadata.Range` before reusing a dim in tests.
- **Cross-stack sync**: Whenever a C# pass relies on Python/C++ helpers (e.g., TritonAffineUtility), update both the managed rule and any Python builders plus regenerate source-generator outputs when signatures shift.
- **Telemetry discipline**: For scheduling/cost-model tweaks (NTT vectorization, paged attention cache ops), capture FLOP/s + memory stats via `RunCases` harness and preserve dumps (`DumpFlags`) for regression triage.
- **Env hygiene**: Work inside `conda activate flaglang`, keep the editable native build up-to-date, and re-run `dotnet build` after touching shared contracts so source generators and analyzers stay coherent. Do not trust results from another editable install; always confirm `triton.__file__` points inside this checkout.
- **Native linking**: If `import triton` fails after a native edit, inspect `python/triton/_C/libtriton.so` and `build/cmake.linux-x86_64-cpython-3.10/Debug/lib/libnncaseruntime.so` with `ldd`, `nm -D`, or `readelf -sW` before debugging Python code. Hidden C++ wrapper symbols or missing runtime library paths can block the import before any Python backend logic runs.

## Onboarding Checklist
1. Clone the repo and `conda activate flaglang`.
2. Run `python -m pip install -r python/requirements.txt` and `python -m pip install "conan>=2.20.0"` if the environment is fresh.
3. Run `TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v` to build C++/Python extensions.
4. Run `dotnet build` and targeted `dotnet test ... -s test.runsettings` commands to verify managed layers.
5. Install CUDA-enabled `torch` before running Triton tutorials such as `python/tutorials/01-vector-add.py`.
6. Execute representative kernel benchmarks to populate baseline telemetry.
7. Study current affine/shardy ADRs plus the cost-model interface before attempting a new optimization.
