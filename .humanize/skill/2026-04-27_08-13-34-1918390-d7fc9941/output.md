Task11 contract for task12:

- Stage boundary: `cubin` must consume **PTX text**, not an MLIR `ir.module`.
- So the pipeline should be: `llir -> ptx (string) -> cubin (bytes)`.
- Minimum fix is in `python/triton/backends/nvidia/compiler.py`:
  - `make_ptx(...)` must return PTX source text and populate metadata.
  - `make_cubin(...)` must write that PTX string to a temp `.ptx` file and call `ptxas`.
- `CompiledKernel` in `python/triton/compiler/compiler.py` already expects the final binary stage to be stored under the backend binary extension (`cubin`) and the launcher to use the metadata object.

Required metadata keys for launcher/binary load:
- `name`
- `shared`
- `num_warps`
- `num_ctas`
- `cluster_dims` as a 3-tuple
- `target` as `GPUTarget`
- Optional but already consumed: `tmem_size`, `global_scratch_size`, `global_scratch_align`, `profile_scratch_size`, `profile_scratch_align`

Affected functions/files:
- `python/triton/backends/nvidia/compiler.py`
  - `make_llir`
  - `make_ptx`
  - `make_cubin`
  - `add_stages`
  - `pack_metadata`
- `python/triton/compiler/compiler.py`
  - `CompiledKernel.__init__`
  - `_init_handles`
  - `compile(...)` stage materialization path
