# 01 Vector Add Bring-up Plan

## 目标描述

在当前 FlagLang 仓库中跑通 `python/tutorials/01-vector-add.py`，确保 Triton Python 前端、nncase IR builder、Triton load/store 到 affine IO 的重写、NTT/TIR lowering、Python backend artifact/launcher 合约，以及 CUDA 运行依赖能够端到端协同工作。

本计划的验收结果必须来自 `/mnt/home-nas/work/repo/FlagLang` 当前 checkout。任何来自其他 editable install 或旧 Triton cache 的结果都不能作为验收依据。

## 验收标准

以下标准按 TDD 思路组织，每项都包含正向测试和负向测试。涉及数值的检查中，`max error = 0.0`、`torch.cuda.is_available() == True`、targeted tests 全通过是硬性正确性门槛；benchmark 只要求能完成并打印数据，不设置性能阈值。

- AC-1: 当前仓库的 native/Python Triton 包可以在 `flaglang` 环境中导入
  - 正向测试（预期通过）:
    - 在 repo root 执行：
      ```bash
      TRITON_BUILD_PROTON=OFF conda run -n flaglang python -m pip install -e . --no-build-isolation -v
      conda run -n flaglang python -c "import triton, triton._C.libtriton as lt; print(triton.__file__); print(lt.__file__)"
      ```
      输出路径都指向 `/mnt/home-nas/work/repo/FlagLang/python/triton`。
    - `ldd python/triton/_C/libtriton*.so` 能解析 `libnncaseruntime.so`，且不会因为 `nncase::clr::*` wrapper 符号不可见而导入失败。
  - 负向测试（预期失败）:
    - 如果 `triton.__file__` 指向 `/mnt/home-nas/work/repo/flagtree` 或其他 checkout，验收脚本必须判定失败。
    - 如果清空必要 runtime library path 后 `libnncaseruntime.so` 不可定位，导入检查应失败并暴露缺失依赖。

- AC-2: tutorial 运行依赖在 `flaglang` 环境中可用
  - 正向测试（预期通过）:
    - 执行：
      ```bash
      conda run -n flaglang python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
      ```
      结果中 `torch.cuda.is_available()` 为 `True`。
    - `python/tutorials/01-vector-add.py` 的 unit-test 路径能进入编译流程，而不是在 import 或 CUDA dependency 阶段退出。
  - 负向测试（预期失败）:
    - 未安装 CUDA-enabled PyTorch 时，tutorial 验收必须失败并给出依赖缺失原因。
    - CPU-only PyTorch 报告 `torch.cuda.is_available() == False` 时不能继续作为有效验收。

- AC-3: Python nncase builder 暴露 vector add 所需的最小 API 面
  - 正向测试（预期通过）:
    - `python/triton/compiler/code_generator.py` 在处理 vector add AST 时能调用 native builder 完成 module/function/block/type/constant/op/location 的构建。
    - 覆盖以下 API 类别：`create_module`、`get_or_insert_function`、`add_entry_block`、`args`、`set_arg_attr`、`finalize`、`ret`；insertion point 和 block 管理；int/float/pointer/block 类型；scalar 常量；program id、range、splat、add/mul/addptr、integer compare、masked load/store、floating add；location 创建与设置。
    - `python/flaglang/native/ir.cpp` 与 `python/flaglang/native/passes.cpp` 不再维护两个互相漂移的 `init_triton_ir` 实现。
  - 负向测试（预期失败）:
    - 删除或隐藏 vector add 需要的任一 builder 方法后，新增的 Python/native binding test 应明确失败。
    - `passes.cpp` 中存在过期且会注册不同 enum/op surface 的 `init_triton_ir` 时，consistency test 应失败。

- AC-4: Triton load/store 可以稳定重写为 typed affine gather/scatter，并表达 vector add 的 affine 关系
  - 正向测试（预期通过）:
    - 执行：
      ```bash
      dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings --filter "FullyQualifiedName~UnitTestTensorizeIO"
      ```
      targeted affine IO tests 全部通过。
    - `LoadToAffineGather` 从 matched `call.CheckedShape` 或 `call.CheckedTensorType.Shape` 获取 shape，不读取 `IR.Triton.Load` op 的 `CheckedShape/CheckedType`。
    - `StoreToAffineScatter` 生成的表达式可以通过 `CompilerServices.InferenceType`，且不会在 `AutoDistributedRewriter` 中留下未推断的 `CheckedTensorType`。
    - vector add 地址关系断言包括：
      - result 形如 `s0 * BLOCK_SIZE + d0`
      - constraint 形如 `s0 * BLOCK_SIZE + d0 < s1`
      - `d0` 的 range 覆盖 `[0, BLOCK_SIZE - 1]`
      - symbols 至少包含 `program_id(0)` 和 `n_elements`
      - pointer base 只有一个，并保留为 gather/scatter 的 source/dest
  - 负向测试（预期失败）:
    - 多 pointer base 的地址表达式不得被错误重写为单一 gather/scatter。
    - 无法转换为 affine relation 的 mask 或 address 表达式应保持原状或明确失败，不能生成错误 relation。
    - 未设置 `Metadata.Range` 的 dynamic domain 不得悄悄通过关系断言。

- AC-5: NTT/TIR 支持 masked、symbolic affine IO 的最小 lowering
  - 正向测试（预期通过）:
    - `NTTAffineIOLoweringPass` 能处理 `relation.Constraint != LogicalExpr.True` 和 `relation.Symbols.Length > 0` 的 vector add 场景。
    - 对 masked `tl.load`，mask 为真时读取 source，mask 为假时返回 `other/defaultValue`。
    - 对 masked `tl.store`，mask 为真时写回，mask 为假时不写 dest。
    - 新增 lowering tests 覆盖 full block 和 tail block 两类输入。
  - 负向测试（预期失败）:
    - mask 为 false 时发生 store 写回，test 应失败。
    - masked load 忽略 `other/defaultValue`，test 应失败。
    - symbol payload 与 relation symbols 数量不一致时 lowering 应明确拒绝，而不是生成错误地址。

- AC-6: Python backend 与 nncase/NTT codegen 的 `cubin` artifact/launcher 合约明确且可端到端运行
  - 正向测试（预期通过）:
    - 强制重新编译运行：
      ```bash
      TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
        conda run -n flaglang python python/tutorials/01-vector-add.py --only_unit_test
      ```
      `output_torch` 和 `output_triton` 最大误差为 `0.0`。
    - dump 目录包含本次生成的中间产物，不能只命中旧 cache。
    - 完整运行：
      ```bash
      TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump \
        conda run -n flaglang python python/tutorials/01-vector-add.py
      ```
      benchmark 正常结束并打印数据。
  - 负向测试（预期失败）:
    - backend 返回 `cubin` 但缺少 `name`、`shared`、`num_warps`、`cluster_dims` 等 launcher metadata 时，launch test 应失败。
    - backend 返回 nncase `kmodel` 或其他非 `cubin` runtime artifact 时，contract test 应失败；DEC-1 已固定选择 `cubin` 路线。
    - 未设置 `TRITON_ALWAYS_COMPILE=1` 的端到端结果不得作为最终验收。

## 路径边界

### 上界（最大可接受范围）

实现一个可维护的 vector add bring-up：native packaging/import、Python builder binding、C# affine rewrite、NTT masked symbolic IO lowering、backend `cubin` artifact/launcher contract、targeted tests、强制重编译 tutorial unit test 和完整 benchmark 全部闭环。允许补充小型诊断脚本或 test helper，但不引入与 vector add 无关的大规模 backend 重构。

### 下界（最小可接受范围）

至少修复当前 FlagLang checkout 中的 `import triton`、安装/验证 CUDA PyTorch、补齐 vector add 必需 builder API、修复 affine load/store 类型传播、支持 vector add 的 masked symbolic affine IO lowering，并让 `01-vector-add.py --only_unit_test` 在强制重新编译时以 `max error = 0.0` 通过。

### 允许的选择

- 可以使用: CMake rpath/install rules、target-local visibility override、明确导出的 `nncase::clr::*` wrapper API、pybind static linkage、现有 `CompilerServices.InferenceType`、`CompilerServices.TryMatchRoot`、`Rewrite`、`dotnet test -s test.runsettings`、`TRITON_ALWAYS_COMPILE` dump 验证。
- 可以使用: masked IO 的直接 `if mask` loop lowering，或 main loop + tail loop 的 mask peeling。初始实现优先选择直接 masked loop，除非现有 TIR builder 已经更适合 peeling。
- 可以使用: 固定走 CUDA `cubin` 路线，补齐 `CompiledKernel` 和 CUDA launcher 需要的 metadata，包括 `name`、`shared`、`num_warps`、`cluster_dims` 等字段。
- 不可以使用: 依赖其他 checkout 的 editable install、复用旧 Triton cache 作为验收、手动跳过 `test.runsettings`、忽略 masked store 语义、将 pointer base 当作普通 affine domain、以结构相等替代 affine relation 断言、切换到 nncase `kmodel` 或其他非 `cubin` artifact 路线。

## 可行性提示和建议

### 概念路径

一个可行的实现顺序是先保证 Python import 和 CUDA dependency 可靠，再收敛 builder API，使 vector add AST 能生成 nncase IR。随后在 C# 中修复 load/store rewrite 的 checked type 来源，并加强 `TritonAffineUtility` 的 domain/symbol 建模。最后让 NTT lowering 接受 vector add 必然出现的 mask 和 symbols，并在 Python backend 侧落实 `cubin` artifact/launcher contract。

masked lowering 的最小语义可以按如下方式表达：

```text
for d0 in 0..BLOCK_SIZE-1:
  address = program_id(0) * BLOCK_SIZE + d0
  if address < n_elements:
    load_value = source[address]
  else:
    load_value = default_value[d0]

  result = load_x + load_y

  if address < n_elements:
    dest[address] = result
```

### 相关引用

- `python/flaglang/native/ffi.cpp` - `libtriton` pybind module 注册入口。
- `python/flaglang/native/ir.cpp` - nncase/Triton native builder binding 的主要实现。
- `python/flaglang/native/passes.cpp` - 需要删除或合并的旧 `init_triton_ir` surface。
- `python/triton/compiler/code_generator.py` - Triton AST 到 native builder 调用的 Python 入口。
- `python/triton/compiler/compiler.py` - `CompiledKernel`、cache、artifact metadata、launcher 调用路径。
- `python/triton/runtime/jit.py` - tutorial 调用 JIT 编译与 launch 的入口。
- `src/Native/include/nncase/compiler.h` - 可能需要导出给 pybind 跨 DSO 调用的 CLR wrapper 声明。
- `src/Nncase.Passes/Rules/Triton/LoadToAffineGather.cs` - load 到 affine gather 的重写规则。
- `src/Nncase.Passes/Rules/Triton/StoreToAffineScatter.cs` - store 到 affine scatter 的重写规则。
- `src/Nncase.Passes/Rules/Triton/TritonAffineUtility.cs` - Triton address/mask 到 affine relation 的建模工具。
- `src/Nncase.Tests/Transform/UnitTestTensorizeIO.cs` - targeted affine IO rewrite tests。
- `src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs` - affine gather/scatter 到 TIR loop/buffer op 的 lowering。
- `src/modules/Nncase.Modules.NTT/TIR/NTT/AffineGather.cs` 和 `src/modules/Nncase.Modules.NTT/TIR/NTT/AffineScatter.cs` - NTT affine IO op 定义。
- `src/modules/Nncase.Modules.NTT/Targets/NTTTarget.cs` - NTT pass pipeline 注册位置。
- `python/tutorials/01-vector-add.py` - 最终端到端验收入口。

## 依赖和顺序

### 里程碑

1. Native import 与环境闭环
   - 修复 `libnncaseruntime.so` 可定位性。
   - 修复 `-fvisibility=hidden` 导致的 `nncase::clr::*` wrapper 符号不可见问题。
   - 在 `flaglang` 中安装并验证 CUDA-enabled PyTorch。
   - 确认 `import triton` 来自当前 FlagLang checkout。

2. Python builder 最小 surface
   - 对照 `code_generator.py` 的 vector add 路径补齐 native builder 方法。
   - 合并或删除过期 `init_triton_ir` 实现。
   - 增加 binding consistency 和 vector add AST smoke test。

3. C# affine rewrite 正确性
   - 修复 `LoadToAffineGather` 的 shape/type 来源。
   - 修复 `StoreToAffineScatter` 的 inference 和 downstream checked type。
   - 将 `program_id(0)`、`n_elements` 建模为 symbols，将 `arange` lane 建模为 domain。
   - 用 relation assertions 替代单纯结构断言。

4. NTT masked symbolic IO lowering
   - 让 `NTTAffineIOLoweringPass` 支持 relation symbols 和 mask constraint。
   - 按 Triton 语义处理 masked load default value 与 masked store no-op。
   - 增加 full block 和 tail block tests。

5. Backend `cubin` artifact/launcher contract 与端到端验收
   - 实现 CUDA `cubin` artifact contract。
   - 填齐 CUDA launcher metadata。
   - 强制重编译运行 `01-vector-add.py --only_unit_test`。
   - 运行完整 tutorial benchmark 并保存 dump 信息。

组件依赖顺序为：import/package 修复先于 Python builder smoke test；builder smoke test 先于 tutorial 编译；affine rewrite 先于 NTT lowering；NTT lowering 先于 backend `cubin` artifact/launcher 端到端验收。

## 任务拆分

每个任务只包含一个 routing tag：`coding` 表示实现任务，`analyze` 表示需要 Codex 调研或验证。

| Task ID | 描述 | 目标 AC | Tag (`coding`/`analyze`) | 依赖 |
|---------|-------------|-----------|----------------------------|------------|
| task1 | 定位并记录 `libnncaseruntime.so` 搜索路径和 `nncase::clr::*` symbol visibility 的实际链接状态 | AC-1 | analyze | - |
| task2 | 修复 native package/rpath/export/static-link 方案，使 `import triton` 来自当前 checkout | AC-1 | coding | task1 |
| task3 | 安装或验证 CUDA-enabled PyTorch，并记录 tutorial 依赖检查命令 | AC-2 | analyze | task2 |
| task4 | 对照 vector add AST 调用链补齐 `python/flaglang/native/ir.cpp` builder API | AC-3 | coding | task2 |
| task5 | 合并或删除 `python/flaglang/native/passes.cpp` 中过期 `init_triton_ir` surface | AC-3 | coding | task4 |
| task6 | 增加 Python/native binding smoke test，覆盖 vector add 最小 API 面 | AC-3 | coding | task4 |
| task7 | 修复 `LoadToAffineGather` 和 `StoreToAffineScatter` 的 checked type 与 inference 行为 | AC-4 | coding | task4 |
| task8 | 扩展 `TritonAffineUtility` 的 domain/symbol/range 建模，并增加 relation assertions | AC-4 | coding | task7 |
| task9 | 调研现有 TIR builder 是否更适合直接 masked loop 或 mask peeling | AC-5 | analyze | task8 |
| task10 | 实现 masked symbolic affine gather/scatter lowering | AC-5 | coding | task9 |
| task11 | 明确 Python backend `cubin` artifact/launcher contract，并列出需要修改的 call path | AC-6 | analyze | task10 |
| task12 | 实现 `cubin` contract 并补齐 CUDA launcher metadata | AC-6 | coding | task11 |
| task13 | 执行强制重编译 tutorial unit test、完整 benchmark、targeted C# tests，并汇总 dump/日志 | AC-1, AC-2, AC-4, AC-6 | analyze | task12 |

## Claude-Codex 协商结论

### 共识

- 验收必须基于当前 FlagLang checkout，不能复用 `flagtree` 的历史结果。
- `LoadToAffineGather` 读取 op 的 checked type 是错误方向，应从 matched call 或 inference 结果获取。
- vector add 必然需要 symbols 和 mask constraint，因此 `NTTAffineIOLoweringPass` 当前的 hard reject 需要放宽。
- artifact/launcher contract 必须明确，不能同时假设 Triton 原生 `cubin` launch 和 nncase runtime artifact。

### 已解决分歧

- Native import 修复方式：可选方案包括 rpath/install、export macro、target-local visibility override、static linkage。计划不固定唯一实现，但要求结果能通过 import、`ldd`、symbol visibility 验证；优先采用最小、可维护、局部影响的改动。
- Masked IO 策略：直接 masked loop 和 mask peeling 都可行。初始推荐直接 masked loop，因为它覆盖 full block 和 tail block 语义，且实现范围较小；后续性能优化可再引入 peeling。
- Python backend artifact contract：用户已选择 CUDA `cubin` 路线。后续实现不再并行推进 nncase `kmodel` runtime launcher 路线。

### 收敛状态

- 最终状态: `converged`

## 用户决策记录

- DEC-1: Python backend artifact contract
  - Claude 立场: 优先继续产出 `cubin`，补齐 `CompiledKernel` 需要的 metadata，尽量少改 Triton launcher。
  - Codex 立场: 如果 nncase/NTT 近期更稳定地产出 `kmodel`，应改造 `CompiledKernel`/driver path 走 nncase runtime，避免伪装成 CUDA binary。
  - 取舍摘要: `cubin` 路径对 Triton 原生 launcher 改动小，但要求 codegen 真的生成 CUDA binary；`kmodel` 路径更贴近 nncase runtime，但会触碰 Python runtime launch contract。
  - 用户决策: 选择 `cubin` 路线。
  - 决策状态: `RESOLVED`

## 实现说明

### 代码风格要求

- 实现代码和代码注释中不要包含 `AC-`、`Milestone`、`Step`、`Phase` 等计划文档术语。
- 代码命名使用领域语义，例如 `maskedAffineLoad`、`programIdSymbol`、`BuildSymbolMap`，不要使用计划编号命名。
- 复杂 affine、mask、symbol lowering 逻辑可以添加简短注释，解释数学约束或 Triton 语义。
- 所有 `dotnet test` 命令必须带 `-s test.runsettings`。
- C++/Python native 变更后重新运行：
  ```bash
  TRITON_BUILD_PROTON=OFF python -m pip install -e . --no-build-isolation -v
  ```
- 端到端 tutorial 验收必须设置：
  ```bash
  TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=/tmp/flaglang-vector-add-dump
  ```
