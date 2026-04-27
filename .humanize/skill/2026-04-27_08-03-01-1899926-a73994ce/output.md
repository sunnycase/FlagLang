Task10 should use direct `T.If`-guarded per-lane loops, not mask peeling.

Reason: `NTTAffineIOLoweringPass` already lowers affine gather/scatter into canonical `T.Serial` nests, and the current implementation explicitly rejects `relation.Constraint != True` and non-empty `relation.Symbols`. That is the exact gap Task10 needs to close. Guarding each lane with `T.If` lets you preserve the existing loop structure and emit `T.Load`/`T.Store` or `T.BufferLoad`/`T.BufferStore` only on valid lanes, instead of introducing a separate peeling transform.

TIR APIs to use:
- `T.Serial` for the loop nest
- `T.If` for the mask/constraint guard
- `T.Load` / `T.Store` for raw pointer gather/scatter
- `T.BufferLoad` / `T.BufferStore` for buffer-side accesses when walking the source/dest buffers

Required support:
- `relation.Symbols`: bind them from the affine op payload and substitute them during address evaluation, as `BuildSymbolMap` already sketches.
- `relation.Constraint`: lower it to the `T.If` condition instead of rejecting it.
- Keep domain-rank validation and affine expression evaluation intact.

Key tests to add/update:
- `LoadToAffineGatherModelsVectorAddSymbolsAndLaneDomain` style coverage for symbol binding.
- New masked affine gather test: verify `T.If` guards the lane body and the relation constraint is preserved.
- New masked affine scatter test: verify guarded `T.Store` emission.
- Negative tests for unsupported dynamic/unbounded domains, like the existing `RejectsUnboundedDynamicLaneDomain`.
