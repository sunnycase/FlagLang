# Ask Codex Input

## Question

Task11 only. Current forced vector-add fails in python/triton/backends/nvidia/compiler.py make_cubin because src is triton._C.libtriton.ir.module, not PTX string. Inspect the Python pipeline contract from python/triton/compiler/compiler.py CompiledKernel and nvidia/compiler.py. Define the minimum CUDA cubin artifact/launcher contract for task12: what stage should emit, required metadata keys (name, shared, num_warps, cluster_dims etc.), and affected functions/files. Keep under 250 words.

## Configuration

- Model: gpt-5.4-mini
- Effort: low
- Timeout: 180s
- Timestamp: 2026-04-27_08-13-34
