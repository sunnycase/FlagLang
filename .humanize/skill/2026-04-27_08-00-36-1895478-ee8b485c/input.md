# Ask Codex Input

## Question

In /mnt/home-nas/work/repo/FlagLang, inspect the current TIR/NTT builder and lowering code relevant to masked symbolic affine gather/scatter. Focus on src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs, src/modules/Nncase.Modules.NTT/TIR/NTT/AffineGather.cs, src/modules/Nncase.Modules.NTT/TIR/NTT/AffineScatter.cs, src/Nncase.Core/TIR/Script.cs, and the TIR codegen support for IfThenElse. Determine whether task10 should implement vector-add masked affine IO using direct per-lane masked loops or mask peeling. Return concise bullets: recommendation, exact APIs/classes to use, expected code changes, tests to add, risks/blockers.

## Configuration

- Model: gpt-5.4
- Effort: high
- Timeout: 900s
- Timestamp: 2026-04-27_08-00-36
