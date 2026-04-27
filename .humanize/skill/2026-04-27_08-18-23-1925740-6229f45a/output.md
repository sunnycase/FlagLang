Minimal validation set for `task13`:

1. Import + linkage check:
```bash
conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"
ldd python/triton/_C/libtriton*.so
```

2. Python native smoke:
```bash
python -m pytest -q python/test/unit/runtime/test_build.py
```

3. Targeted managed tests:
```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestNTTAffineIOLowering"
```

4. Forced tutorial unit test:
```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
  conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test
```

5. Full benchmark run:
```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
  conda run -n flaglang python python/tutorials/01-vector-add.py
```

If you want, I can also turn this into a one-line checklist with expected pass criteria.
