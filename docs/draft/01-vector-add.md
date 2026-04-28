# 01 Vector Add Bring-up Plan

本文记录在当前仓库 `/mnt/home-nas/work/repo/FlagLang` 中，基于 nncase 后端接入 Triton 并跑通 `python/tutorials/01-vector-add.py` 还需要完成的工作。

## 当前实测状态

环境已按新要求切到 `conda env: flaglang`，不再使用 `flagtree`。

已完成：

- `conda create -n flaglang python=3.10 -y`
- `conda run -n flaglang python -m pip install -r python/requirements.txt`
- `conda run -n flaglang python -m pip install "conan>=2.20.0"`
- `TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v`
- `dotnet build`

`dotnet build` 结果：通过，只有既有 StyleCop warning。

当前阻塞：

- `flaglang` 中 `import triton` 失败：`libnncaseruntime.so: cannot open shared object file`。
- 手动加 `LD_LIBRARY_PATH=build/cmake.linux-x86_64-cpython-3.10/Debug/lib` 后仍失败：`libtriton.so` 需要的 `nncase::clr::tensor_type::tensor_type(...)` 是未解析符号。
- `libnncaseruntime.so` 内对应符号是 `LOCAL`，原因是全局编译选项使用了 `-fvisibility=hidden`，而 `python/flaglang/native/ir.cpp` 的 pybind 扩展需要跨 DSO 调用这些 C++ wrapper。
- `flaglang` 还没有安装 `torch`，即使 native import 修复，tutorial 也需要 CUDA 版 PyTorch。

相关 targeted test：

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings \
  --filter "FullyQualifiedName~UnitTestTensorizeIO|FullyQualifiedName~TestAffineGatherKernelMatchesEvaluator|FullyQualifiedName~TestAffineGatherStridedKernelMatchesEvaluator|FullyQualifiedName~TestAffineScatterKernelMatchesEvaluator"
```

结果：5 个用例中 1 个通过，4 个失败。

主要失败点：

- `LoadToAffineGather.GetReplace` 读取 `load.CheckedShape`，但 `Load` 是 op，不是 call，op 本身没有 `CheckedType/CheckedShape`。
- affine scatter 进入后续 pipeline 后仍有表达式缺 checked type，`AutoDistributedRewriter` 访问 `CheckedTensorType` 时失败。

另外注意：之前在 `flagtree` 中 `01-vector-add.py --only_unit_test` 可以通过，但那次实际导入的是 `/mnt/home-nas/work/repo/flagtree/python/triton`，不是当前 FlagLang 仓库；不能作为当前仓库验收结果。

## 目标端到端路径

`01-vector-add.py` 对后端的最小需求是：

1. Python `triton.jit` 前端把 kernel AST 生成 nncase IR。
2. `tl.program_id(0)`、`tl.arange(0, BLOCK_SIZE)`、`ptr + offsets`、`offsets < n_elements`、masked `tl.load`、`x + y`、masked `tl.store` 都能表达。
3. C# pass 将 Triton load/store 转为 affine gather/scatter，并保留 mask、program id、scalar kernel 参数等语义。
4. NTT/TIR 选择和 lowering 能把 affine IO 降到可 codegen 的 buffer/loop/kernel 形态。
5. Python backend 产出 runtime 可加载 artifact，并填齐 Triton launcher 需要的 metadata。
6. `output_torch` 和 `output_triton` 最大误差为 `0.0`，强制重新编译时也通过。

## 必做项

### P0: 修复当前仓库在 `flaglang` 中的导入

- 将 `libnncaseruntime.so` 安装到 Python 包可见路径，或给 `libtriton.so` 设置可移植 rpath。
- 解决 `-fvisibility=hidden` 导致的 nncase CLR C++ wrapper 符号不可见问题。可选方案：
  - 给 `src/Native/include/nncase/compiler.h` 中被 pybind 跨库调用的类型/方法加 export 宏；
  - 对 `nncaseruntime` 目标局部关闭 hidden visibility；
  - 或把 pybind 扩展需要的 wrapper 源码静态并入 `libtriton.so`，避免跨 DSO 解析。
- 修好后用以下命令确认导入当前仓库，而不是其他 editable 安装：

```bash
conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"
```

期望路径都在 `/mnt/home-nas/work/repo/FlagLang/python/triton`。

### P0: 安装 tutorial 运行依赖

- 在 `flaglang` 中安装 CUDA 版 PyTorch。
- 验证：

```bash
conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`torch.cuda.is_available()` 必须为 `True`。

### P1: 补齐 Python nncase builder 的最小 API 面

当前 `python/flaglang/native/ir.cpp` 只暴露了部分 builder 方法。`python/triton/compiler/code_generator.py` 的通用路径还会调用更多方法；即使 vector add 很简单，也至少会经过 function/return/basic-block 相关逻辑。

vector add 最小需要覆盖：

- function/module：`create_module`、`get_or_insert_function`、`add_entry_block`、`args`、`set_arg_attr`、`finalize`、`ret`
- insertion point/block：`get_insertion_block`、`get_insertion_point`、`restore_insertion_point`、`create_block`、`set_insertion_point_to_start`、`set_insertion_point_to_end`
- types/constants：`get_int32_ty`、`get_int64_ty`、`get_float_ty`、`get_ptr_ty`、`get_block_ty`、`get_int32`、`get_int64`
- ops：`create_get_program_id`、`create_make_range`、`create_splat`、`create_add`、`create_mul`、`create_addptr`、`create_icmpSLT`、`create_masked_load`、`create_masked_store`、`create_fadd`
- diagnostics：`create_loc`、`create_name_loc`、`get_loc`、`set_loc`

同时应删除或合并 `python/flaglang/native/passes.cpp` 中过期的 `init_triton_ir` 实现，避免维护两个不一致的 native binding 版本。

### P1: 修复 Triton load/store 到 affine 的类型传播

- `LoadToAffineGather` 不应读取 `load.CheckedShape`；应从 matched `call.CheckedShape` 或 `call.CheckedTensorType.Shape` 获取 shape。
- `LoadToAffineGather` 返回 gather 后要确保 checked type 可被 inference 恢复，不要手动依赖 op 的 checked type。
- `StoreToAffineScatter` 需要确保返回表达式的 checked type 能通过 inference，且不会在 auto-distributed pass 中留下未推断类型。
- targeted tests 应先恢复：

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings \
  --filter "FullyQualifiedName~UnitTestTensorizeIO"
```

### P1: 正确表示 vector-add 的 affine 关系

vector add 的地址表达式是：

```text
offset = program_id(0) * BLOCK_SIZE + arange(0, BLOCK_SIZE)
ptr = ptr_base + offset
mask = offset < n_elements
```

当前 `TritonAffineUtility` 能识别 range 和简单指针加法，但需要明确域和符号边界：

- `arange` 产生的 lane 应作为 affine domain。
- `program_id(0)` 应作为 symbol 或调度维度，而不是普通 element domain。
- `n_elements` 是 kernel scalar 参数，应作为 symbol 进入 constraint。
- `BLOCK_SIZE` 是 constexpr，可作为常量折入 affine expr。
- 指针 base 只能有一个，且应保留为 gather/scatter 的 source/dest。

验收时不要只看结构匹配，要断言 relation：

- result 类似 `s0 * 1024 + d0`
- constraint 类似 `s0 * 1024 + d0 < s1`
- domain range 覆盖 `[0, 1023]`
- symbols 至少包含 `program_id(0)` 和 `n_elements`

### P2: 支持 masked affine IO lowering

`NTTAffineIOLoweringPass` 目前拒绝：

- `relation.Constraint != LogicalExpr.True`
- `relation.Symbols.Length != 0`

这两点都会被 vector add 触发。需要选择一种最小实现：

- 先生成带 `if mask` 的 loop：mask 为真时 load/store，否则 load 使用 `defaultValue`。
- 或做 mask peeling：整块无 mask 的 main loop + tail loop。

对 `tl.load(..., mask=mask)`：

- mask 为 false 时必须返回 `other/defaultValue`；如果 Triton 没显式传 `other`，需要按 Triton 语义处理默认值。

对 `tl.store(..., mask=mask)`：

- mask 为 false 时不能写回。

### P2: 打通 Python backend 到 nncase/NTT codegen 的 artifact contract

当前 Python `CUDABackend` 仍按 Triton 原生 launcher 假设返回 `cubin`，但当前 FlagLang 源码里的多段 lowering 还是过渡形态。需要明确 nncase 后端产物：

- 如果继续产出 `cubin`：nncase/NTT codegen 必须生成 PTX/cubin，并填 metadata：`name`、`shared`、`num_warps`、`cluster_dims` 等。
- 如果产出 nncase `kmodel` 或其他 runtime artifact：需要改 `CompiledKernel`/driver launcher，使 `kernel.run(...)` 调用 nncase runtime，而不是 CUDA driver 的 `load_binary + cuLaunchKernel`。

不要让 cache 中的旧 cubin 掩盖问题。所有端到端测试必须设置：

```bash
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump/flaglang-vector-add-dump
```

### P3: 端到端验收

建议验收命令：

```bash
conda activate flaglang
python -m pip install -r python/requirements.txt
TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v
dotnet build
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=dump/flaglang-vector-add-dump \
  python python/tutorials/01-vector-add.py --only_unit_test
```

最终通过标准：

- `import triton` 确认来自当前 FlagLang 仓库。
- 强制重新编译时 `01-vector-add.py --only_unit_test` 输出最大误差 `0.0`。
- dump 目录能看到本次生成的中间产物，而不是复用旧 cache。
- C# targeted affine tests 通过。
- 再运行完整 `python python/tutorials/01-vector-add.py`，benchmark 能正常结束并打印数据。
