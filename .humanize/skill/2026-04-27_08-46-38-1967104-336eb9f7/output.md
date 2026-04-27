No blocking implementation defect stood out in the touched diff. The remaining issues are validation gaps:

- `[P1]` The new backend contract checks are not useful unless task13 explicitly runs the new pytest file: [python/test/unit/runtime/test_flaglang_cuda_backend.py](/mnt/home-nas/work/repo/FlagLang/python/test/unit/runtime/test_flaglang_cuda_backend.py:49). That file is now the only place exercising pre-PTXAS structural rejection, launcher metadata rejection, and the non-placeholder IR text assertions.
- `[P1]` Dump verification is still only indirect. The serialization path is in [compiler.py](/mnt/home-nas/work/repo/FlagLang/python/triton/compiler/compiler.py:143) and dump writes happen at [compiler.py](/mnt/home-nas/work/repo/FlagLang/python/triton/compiler/compiler.py:357), but the current Python test only inspects `kernel.asm[...]` in memory at [test_flaglang_cuda_backend.py](/mnt/home-nas/work/repo/FlagLang/python/test/unit/runtime/test_flaglang_cuda_backend.py:86). Final verification still needs a real `TRITON_DUMP_DIR` filesystem check.
- `[P1]` The new full/tail masked lowering semantics tests exist in [UnitTestNTTAffineIOLowering.cs](/mnt/home-nas/work/repo/FlagLang/src/Nncase.Tests/TIR/UnitTestNTTAffineIOLowering.cs:45) and [UnitTestNTTAffineIOLowering.cs](/mnt/home-nas/work/repo/FlagLang/src/Nncase.Tests/TIR/UnitTestNTTAffineIOLowering.cs:87), but task13 also needs the explicit `UnitTestTensorizeIO` run from the plan. Right now that command is not implied by the changed files.
- `[P2]` Launcher metadata negative coverage is narrow. `_load_kernel_metadata` also rejects malformed `cluster_dims` and missing `target.backend/arch/warp_size` at [compiler.py](/mnt/home-nas/work/repo/FlagLang/python/triton/compiler/compiler.py:450), but the current test only covers the first missing-field failure (`shared`) at [test_flaglang_cuda_backend.py](/mnt/home-nas/work/repo/FlagLang/python/test/unit/runtime/test_flaglang_cuda_backend.py:69).

Run these before signoff:

```bash
conda run -n flaglang pytest python/test/unit/runtime/test_flaglang_cuda_backend.py -q

dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"

rm -rf /tmp/flaglang-vector-add-dump
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
  conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test

rg -n "entry: _vector_add_kernel|ntt.affine.gather|ntt.affine.scatter" /tmp/flaglang-vector-add-dump
rg -n "<triton\\._C\\.libtriton\\.ir\\.module object" /tmp/flaglang-vector-add-dump && false || true

TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
  conda run -n flaglang python python/tutorials/01-vector-add.py
```

If you want the metadata contract fully closed, add two tiny negatives for bad `cluster_dims` and bad `target` in the Python runtime test file before the final run.
