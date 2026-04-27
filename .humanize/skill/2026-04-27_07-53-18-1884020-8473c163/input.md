# Ask Codex Input

## Question

In /mnt/home-nas/work/repo/FlagLang, inspect the current TIR/NTT builder and lowering code relevant to masked symbolic affine gather/scatter. Focus on src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs, src/modules/Nncase.Modules.NTT/TIR/NTT/AffineGather.cs, src/modules/Nncase.Modules.NTT/TIR/NTT/AffineScatter.cs, and available TIR builders under src/Nncase.Core/IR/TIR and src/Nncase.Core/IR/F. Determine whether task10 should implement vector-add masked affine IO using direct per-lane masked loops or mask peeling. Return: recommendation, exact APIs/classes to use, expected code changes, tests to add, and any risks/blockers.

## Configuration

- Model: gpt-5.5
- Effort: xhigh
- Timeout: 1800s
- Timestamp: 2026-04-27_07-53-18
