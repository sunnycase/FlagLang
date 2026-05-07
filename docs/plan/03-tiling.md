# 03 Tiling Plan

## 目标描述

建立 FlagLang CUDA backend 的 tiling、buffer storage、distribution layout 和 codegen 内存层级模型，让中间数据可以被明确分配到 register、smem、tmem 或 gmem，避免 transient intermediate 被默认物化到 gmem 或 CUDA `.local` memory。

本计划的第一阶段不追求完整 matmul/tensor-core pipeline，而是先修正当前 buffer/distribution 表达能力：把 buffer 的 usage、visibility/scope、physical location 拆成独立维度；把 `DistributedType` 从简单 `SBP.S(axis)` 扩展为可表达 Triton blocked layout 的 ownership map；把 owner 本地的 register fragment、smem swizzle、bank layout 等放入独立 `StorageLayout`。随后在主 `CompileAsync()` pipeline 中启用 tiling，支持 direct `IR.Affine.Gather/Scatter` 路线的 register-resident elementwise tile 和 block-local smem tile。

性能指标中，vector-add 16M 的 `356 GB/s` 是 optimized non-diagnostic build 的硬性回归护栏，对应当前 direct distributed baseline `375.21 GB/s` 的 5% 退化限制。smem 收益不强制用 vector add 证明，必须用有 block 内复用的 microkernel 证明中间数据没有走 gmem/local memory。

## 验收标准

以下标准按 TDD 思路组织。每项都包含正向测试和负向测试，且所有失败路径必须 fail fast，不能静默退回 implicit thread-local temp storage、non-tiled pipeline、gmem 或 local memory。

- AC-1: Buffer storage model 拆分 usage、visibility/scope 和 physical location
  - 正向测试（预期通过）:
    - 新增 `BufferStorage` 或等价结构，至少包含 `Usage`、`Visibility/Scope`、`PhysicalLocation/MemorySpace`、`Alignment`，并能被 `IR.Buffers.Uninitialized`、`TIR.PhysicalBuffer`、`TIR.Buffer`、bufferize schedule result 和 codegen visitor 消费。
    - `Usage` 能独立表达 `Input`、`Output`、`Const/Rdata`、`Temp`、`Scratch`、`Staging`、`Accumulator`。
    - `Visibility/Scope` 能独立表达 `ThreadLocal`、`WarpLocal`、`BlockLocal`、`ClusterLocal`、`Grid/Device`。
    - GPU physical location 能表达 `Register`、`SMem`、`TMem`、`GMem`、`LocalAddressable`；CPU physical location 至少表达 `Register`、`Stack/Heap/DRAM`，L1/L2 作为 cache policy 或 residency hint。
    - dump 中每个 tiled intermediate 都能打印 `Usage`、`Visibility/Scope`、`PhysicalLocation`。
  - 负向测试（预期失败）:
    - 新增 code 把 usage、scope 和 physical location 拼回同一个 enum value 时，static/verifier test 失败。
    - `GpuRegister` buffer 被建成 addressable `MemSpan` 或可动态取地址 buffer 时，verifier 失败。
    - transient intermediate 没有明确 storage decision 却被创建成 implicit thread-local temp storage 时，pipeline test 失败。

- AC-2: `DistributedType` 支持双向 `DistributionLayout` ownership map
  - 正向测试（预期通过）:
    - `src/Nncase.Core/DistributedType.cs` 或等价位置新增 `DistributionLayout`，包含 `GlobalToOwnerLocal`、`OwnerLocalToGlobal`、`LocalShape`、`ValidPredicate`。
    - 现有 `SBP.S/B/P/R` 能 lowering 成 `DistributionLayout`，保持 simple split、broadcast、partial、replicate 的兼容语义。
    - 新增 Triton blocked layout preset，至少包含 `sizePerThread`、`threadsPerWarp`、`warpsPerCTA`、`order`、`ctasPerCGA`、`ctaSplitNum`、`ctaOrder`。
    - fixture 覆盖 contiguous-per-thread ownership 和 strided-per-thread ownership，dump 中能看到不同的 `GlobalToOwnerLocal` 和 `OwnerLocalToGlobal`。
  - 负向测试（预期失败）:
    - `DistributionLayout` 缺少 inverse map 时，verifier 失败。
    - owner 坐标超出 `Placement.Hierarchy` 范围时，verifier 失败。
    - 新增 code 只靠 `SBP.S(axis)` 表达 Triton blocked/strided ownership 时，layout capability test 失败。
    - unsupported named permutation、xor swizzle 或 MMA fragment layout 没有 forward/inverse/domain/codegen lowering 时，必须 fail fast。

- AC-3: `StorageLayout` 只描述 owner 本地物理表示
  - 正向测试（预期通过）:
    - 新增 `StorageLayout` 或等价结构，描述 local logical index 到 physical offset、register fragment coordinate、smem swizzled address、bank layout、alignment 的映射。
    - `StorageLayout` 的 logical domain 等于 `DistributionLayout` 推导出的 local shard domain，或有显式 tile-local view map 解释差异。
    - smem bank swizzle 出现在 `StorageLayout`，不改变 `DistributionLayout` 的 owner 语义。
    - register fragment layout 出现在 `StorageLayout` 或更具体的 register-fragment representation 中，不伪装成普通 addressable buffer stride。
  - 负向测试（预期失败）:
    - `StorageLayout` 改变 thread/lane ownership 时，verifier 失败。
    - `StorageLayout` local domain 和 `DistributionLayout` local shard domain 不一致且无 view map 时，verifier 失败。
    - producer/consumer 的 `DistributionLayout` 不一致但没有显式 redistribute/reshard 时，pipeline test 失败。

- AC-4: 主编译流程启用 tiling 阶段
  - 正向测试（预期通过）:
    - `Compiler.CompileAsync()` 中存在 tiling 阶段，位于 `AutoDistributedPass` 之后、`TIRPass` 之前，或有等价位置并能看见 distributed affine dataflow。
    - tiling pass 能消费当前 Triton route 的 `IR.Affine.Gather/Scatter`、`DistributionLayout`、`StorageLayout`、`ProgramIdDim`、`ThreadId`、mask constraint 和 local shard shape。
    - `DumpFlags.Tiling` 在 forced compile 下输出真实 tiling dump。
    - targeted command 可运行并覆盖新增 tests：
      ```bash
      dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings \
        --filter "FullyQualifiedName~Tiling|FullyQualifiedName~AffineGather|FullyQualifiedName~AffineScatter"
      ```
  - 负向测试（预期失败）:
    - 主 pipeline 跳过 tiling 阶段但 tiling-eligible kernel 仍成功时，pipeline test 失败。
    - tiling pass 遇到 unsupported shape、layout、mask 或 relation 后静默进入 non-tiled success path 时，negative test 失败。
    - dump 中没有 tiling 阶段输出时，forced compile 验收失败。

- AC-5: Tiling decision 同时产生 tile shape、ownership、storage、lifetime 和 capacity 结果
  - 正向测试（预期通过）:
    - 每个 tiled intermediate 的 dump 包含 tile shape、producer/consumer tile relation、`DistributionLayout`、`StorageLayout`、`BufferStorage`、lifetime、byte size。
    - thread-private elementwise tile 被决策为 `Usage=Temp`、`Scope=ThreadLocal`、`Location=GpuRegister` 或等价表达。
    - block-reused tile 被决策为 `Usage=Temp/Scratch/Staging`、`Scope=BlockLocal`、`Location=GpuSMem` 或等价表达。
    - smem/register budget 来自 target options 或 CUDA device 属性，并参与 capacity check。
    - smem slot 复用由 lifetime analysis 证明，schedule dump 能看到复用区间。
  - 负向测试（预期失败）:
    - tile bytes 超过 register/smem budget 时，编译失败并报告 op、tile shape、requested bytes、budget bytes。
    - dynamic bound、alias、producer/consumer ownership 或 synchronization 无法证明时，编译失败并报告根因。
    - tiling decision 缺少 storage 或 lifetime 却继续进入 bufferization 时，verifier 失败。

- AC-6: `NTTAffineIOLoweringPass` 支持 tile-aware lowering
  - 正向测试（预期通过）:
    - `Gather -> Binary -> Scatter` 的 simple chain lower 成 tile-local load/compute/store。
    - TIR 后不再生成完整 local shard 大小的中间 `Data` buffer。
    - mask constraint 进入 tile 内 guard 或 tail peeling；masked load 的 default value 和 masked store 的 no-write 语义正确。
    - distributed gather/scatter 的 global offset、local shard offset、thread/lane owner offset 组合正确。
  - 负向测试（预期失败）:
    - lowering 丢弃 mask constraint 时，correctness test 失败。
    - lowering 为了处理 mask 或 distribution 而回退到完整 shard temporary buffer 时，structural test 失败。
    - direct distributed gather/scatter 被重新替换成 `TensorLoad/TensorStore` boxing path 时，pipeline test 失败。

- AC-7: Register-resident elementwise tile 不产生 address-taken local memory
  - 正向测试（预期通过）:
    - vector add 的 elementwise intermediate dump 为 `Usage=Temp, Scope=ThreadLocal, Location=GpuRegister` 或等价表达。
    - codegen 对 register tile 生成 SSA/scalar/register fragment，不生成对应的 `thread_local_data` tensor buffer。
    - `CodeGen/cuda/thread_main.cu` 中 `flaglang_thread_local_data_storage` 不因 register-eligible elementwise intermediate 增长。
    - PTX/SASS 中不出现与该 intermediate 对应的 `.local` load/store。
  - 负向测试（预期失败）:
    - register-eligible intermediate 被创建成 address-taken byte array 时，register tile test 失败。
    - `.local` 访问由 IR/codegen 主动 address-taken local array 导致时，PTX check 失败。
    - 如果 CUDA compiler 因真实 register pressure spill，需要 dump register count/occupancy 并作为性能诊断，不能被当作 IR register path 验收通过。

- AC-8: Block-local smem tile 有可验证 shared-memory 证据
  - 正向测试（预期通过）:
    - 新增一个有 block 内复用的最小 CUDA microkernel，例如两个 consumer 复用同一 `AffineGather` tile，或小型 tiled transpose/stencil。
    - Tiling 后该 intermediate dump 为 `Usage=Temp/Scratch/Staging, Scope=BlockLocal, Location=GpuSMem` 或等价表达。
    - CUDA source/PTX/SASS 中能看到 shared-memory 证据：`__shared__`、已证明位于 shared memory 的 block-local runtime buffer，或 `.shared` 指令。
    - 使用 block-local storage 的 kernel 有必要同步点，且同步点由 data dependency 解释。
    - smem microkernel 的 global memory transaction 或 runtime 指标优于 non-tiled 版本，至少 dump/PTX 证明中间数据没有走 gmem/local mem。
  - 负向测试（预期失败）:
    - smem tile 没有 `.shared` 或等价 shared-memory 证据时，codegen artifact test 失败。
    - block-local producer/consumer 缺少必要同步时，verifier 或 runtime correctness test 失败。
    - smem bank swizzle 被放进 `DistributionLayout` 并改变 owner 语义时，layout verifier 失败。

- AC-9: Codegen 按 composition map 和 storage 三元模型分派
  - 正向测试（预期通过）:
    - Codegen 的访问生成遵循组合映射：
      ```text
      global logical index
        -> DistributionLayout: owner coordinates + local logical index
        -> tile/schedule map: tile coordinate + inner coordinate
        -> StorageLayout: physical offset / register fragment / smem swizzled address
      ```
    - usage 决定 input/output/const ABI，scope 决定参数实例数、同步和 sharing domain，physical location 决定 CUDA storage class 或 SSA/register 表达形态。
    - input/output/persistent global data 使用 gmem ABI；transient register/smem intermediate 不走 gmem ABI。
  - 负向测试（预期失败）:
    - Codegen 继续用 `Rdata/Data/BlockLocalData` 等组合 enum 直接推导所有行为时，static review test 失败。
    - scope 与 location 不一致，例如 `Scope=BlockLocal` 但 codegen 为每 thread 分配独立 private local array 时，artifact test 失败。
    - `Location=GpuRegister` 却生成 addressable span 或 pointer ABI 时，codegen verifier 失败。

- AC-10: 端到端 vector add 正确性、性能和 dump 门禁
  - 正向测试（预期通过）:
    - 强制重新编译运行：
      ```bash
      TRITON_ALWAYS_COMPILE=1 \
      TRITON_KERNEL_DUMP=1 \
      TRITON_DUMP_DIR=dump/tiling-vector-add \
      python python/tutorials/01-vector-add.py --only_unit_test
      ```
    - 输出最大误差为 `0.0` 或严格等价的 float32 elementwise add 结果。
    - 16M vector add optimized non-diagnostic build 不低于约 `356 GB/s`，除非 dump 明确证明本次使用额外诊断或非优化编译选项。
    - `triton.__file__` 和 `triton._C.libtriton` 都来自当前 `/mnt/home-nas/work/repo/FlagLang` checkout。
    - dump 保存在 `dump/tiling-*` 下，不写 `/tmp`。
    - 需要访问 GPU 的 humanize/Codex review、benchmark 和端到端验收必须禁用 sandbox，并显式设置：
      ```bash
      HUMANIZE_CODEX_BYPASS_SANDBOX=true
      ```
  - 负向测试（预期失败）:
    - 命中旧 Triton cache、未设置 `TRITON_ALWAYS_COMPILE=1`、或导入其他 checkout 时，端到端验收失败。
    - review 或 benchmark 在 sandbox 中运行导致 GPU 不可见时，结果不能作为有效验收。
    - vector add 通过手写 shortcut、旧 non-tiled path 或 hidden fallback 成功时，验收失败。
    - 16M 性能低于护栏且没有诊断解释时，性能门禁失败。

- AC-11: Fail-fast diagnostics 覆盖所有 unsupported 路径
  - 正向测试（预期通过）:
    - unsupported `DistributionLayout`、缺失 inverse map、layout domain 不匹配、producer/consumer ownership 无法证明时，错误信息包含 layout 名称、op、shape、owner domain 和失败原因。
    - smem/register capacity 失败时，错误信息包含 requested bytes、budget bytes、tile shape、scope、location。
    - affine relation、mask constraint 或 dynamic shape 无法 tile 时，错误信息指向具体 relation 或 dimension。
  - 负向测试（预期失败）:
    - 任何 unsupported case 自动退回 implicit thread-local temp storage、gmem、local mem、旧 `AutoTilePass` 成功路径或 non-tiled pipeline 时，negative test 失败。
    - error message 只有通用 `NotSupportedException` 且没有可定位上下文时，diagnostic quality test 失败。

## 路径边界

### 上界（最大可接受范围）

完整实现 buffer/distribution/storage model 改造，启用主 pipeline tiling，支持 direct `IR.Affine.Gather/Scatter` 的 register-resident elementwise tile 和 block-local smem tile，完成 verifier、bufferization、codegen、PTX/SASS artifact 检查、vector add 端到端性能门禁和一个 smem 复用 microkernel。允许迁移现有 `DistributedType`、`TIR.Buffer`、`PhysicalBuffer`、bufferize scheduler、NTT CUDA codegen 和相关 tests。

### 下界（最小可接受范围）

最小可合并实现必须包含：

- `BufferStorage` 或等价 annotation，拆分 usage、scope、physical location。
- `DistributionLayout` 和 `StorageLayout` 的最小 IR 类型、dump 和 verifier。
- `SBP.S/B/P/R` 到 `DistributionLayout` 的兼容 lowering。
- 一维 Triton/vector-add 风格 ownership 的 contiguous 和 strided fixtures。
- 主 pipeline 运行 tiling 阶段并对 unsupported case fail fast。
- vector add register path 不再为 elementwise intermediate 分配 address-taken local buffer。
- 一个 block-local smem microkernel 证明 smem path 可生成可验证 artifact。

### 允许的选择

- 可以使用: 现有 `AutoTilePass`、`GraphTiler`、`TileGraph` 和 OR-Tools 代码作为实现参考，但必须支持当前 direct `IR.Affine.Gather/Scatter` route，不能只支持旧 `IR.Affine.Grid`。
- 可以使用: 受限 index map 表达 affine、floordiv、mod、power-of-two permutation、xor swizzle 和 named map primitive。
- 可以使用: named map primitive 表达 Triton blocked layout、MMA fragment layout、non-affine swizzle，但每个 primitive 必须有 forward、inverse、domain、cost features 和 codegen lowering。
- 可以使用: 先用确定性 tile policy，后续再接 cost model/autotune。
- 不可以使用: 手写 vector-add 特例、Python backend PTX shortcut、hidden non-tiled fallback、旧 cache、`/tmp` dump、缺失 `test.runsettings` 的 managed test。
- 不可以使用: 将 smem bank swizzle 放进 `DistributionLayout` 并改变 owner 语义。
- 不可以使用: 将 register path 表达成普通 addressable `MemSpan` 后依赖 CUDA 编译器消除 local memory。
- 不要求: 本阶段完整支持 tensor core、async copy、TMA、warp specialization、跨 kernel global cache、完整 tmem codegen 或全自动 cost model search。

## 可行性提示和建议

### 概念路径

建议先建立三层映射，而不是直接改 codegen：

```text
global logical index
  -> DistributionLayout
     owner coordinates + local logical index
  -> tile/schedule map
     tile coordinate + inner coordinate
  -> StorageLayout
     register fragment / smem address / gmem address
```

`DistributedType` 的职责是 ownership，不应编码 stride、bank swizzle 或 register fragment。`StorageLayout` 的职责是 owner-local representation，不应改变 thread/warp/block ownership。`BufferStorage` 的职责是表达 usage、scope 和 physical memory space，不应承担 sharding 或 local layout 语义。

对于 register，优先设计为 SSA/scalar/register fragment，不作为 addressable `TIR.Buffer`。对于 smem，必须建模 block scope、capacity、bank layout、lifetime 和 synchronization。对于 CUDA `.local` memory，应作为 address-taken local array 或 register spill 的诊断结果，而不是 tiling 主动选择的目标 location。

### 相关引用

- `src/Nncase.Core/DistributedType.cs` - 当前 `SBP`、`Placement`、`DistributedType` 定义，新增 `DistributionLayout` 的主要入口。
- `src/Nncase.Core/IR/Buffers/Uninitialized.cs` - buffer storage 入口，必须直接消费 `BufferStorage`。
- `src/Nncase.Core/IR/Buffers/Functional.cs` - buffer construction API，需要新增 storage-aware overload。
- `src/Nncase.Core/TIR/PhysicalBuffer.cs` - physical buffer 必须只持有 `BufferStorage`，不得恢复组合 memory-location API。
- `src/Nncase.Core/TIR/Buffer.cs` - TIR buffer 持有 `MemSpan`、shape、stride、`DistributedType`，需接入 storage/layout。
- `src/Nncase.Core/TIR/Script.cs` - `T.CreateBuffer` / `AttachBuffer` API，需要 storage-aware construction。
- `src/Nncase.Diagnostics/Diagnostics/ScriptPrintVisitor.cs` - dump 新模型的关键位置。
- `src/Nncase.Evaluator/Buffers/Uninitialized.cs` - distributed/uninitialized type inference 需兼容新 layout。
- `src/Nncase.Core/Utilities/DistributedUtility.cs` - simple SBP 到 ownership map 的兼容 lowering 参考点。
- `src/Nncase.Passes/Distributed/AutoDistributed.cs` - 当前 distributed type 传播，需要避免只生成 simple `SBP.S`。
- `src/Nncase.Compiler/Compiler.cs` - `CompileAsync` 主 pipeline，启用 tiling 阶段的位置。
- `src/Nncase.Schedule/Transforms/AutoTilePass.cs` - 旧 tiling pass，可作为结构参考。
- `src/Nncase.Schedule/Schedule/GraphTiler.cs` - 旧 graph tiler 和 cost/schedule 逻辑。
- `src/Nncase.Schedule/Schedule/TileGraph/PrimGraphSolveResult.cs` - tile storage decision 和 addressable physical buffer 生成点。
- `src/Nncase.Schedule/Transforms/TIRSelectionPass.cs` - implicit thread-local temp storage 的中间 buffer 分配点，必须由显式 storage decision 驱动。
- `src/Nncase.Schedule/Schedule/Bufferize/BufferizeVisitor.cs` - buffer lifetime、pool size、data/warp/block local pool 分配。
- `src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs` - tile-aware affine IO lowering 的主要落点。
- `src/modules/Nncase.Modules.NTT/CodeGen/CPU/KernelCSourceConvertVisitor.cs` - TIR buffer 到 CUDA/NTT source 的主要 visitor。
- `src/modules/Nncase.Modules.NTT/CodeGen/CPU/DeviceCSourceConvertVisitor.cs` - thread-local addressable storage 到 runtime address pool 的路径。
- `src/modules/Nncase.Modules.NTT/CodeGen/CPU/Templates/thread_main.cpp.cshtml` - CUDA entry 中 local data pool 生成点。
- `ntt/src/cuda_runtime.cu` - CUDA runtime thread/warp/block local data pool 与 shared context。
- `python/tutorials/01-vector-add.py` - register path 的端到端验收用例。

## 依赖和顺序

### 里程碑

1. 建立 storage/distribution/layout 基础模型
   - 新增 `BufferStorage` 或等价 annotation，拆分 usage、scope、physical location。
   - 新增 `DistributionLayout` 和 `StorageLayout` 类型。
   - 将 `SBP.S/B/P/R` lowering 到 `DistributionLayout`，保持当前 simple distributed tests 兼容。
   - 增加 dump 和 verifier，先防止新模型退化成另一个大 enum。

2. 接入主 pipeline 和 direct affine IO tiling
   - 在 `CompileAsync()` 中启用 tiling 阶段。
   - 让 tiling pass 消费 direct `IR.Affine.Gather/Scatter`，而不是只消费旧 `IR.Affine.Grid`。
   - 先支持一维 elementwise tile，unsupported case 全部 fail fast。
   - 增加 tiling dump 和 capacity/lifetime diagnostics。

3. 改造 TIR selection 和 bufferization
   - 阻断 `TIRSelectionPass` 对 tiling-eligible intermediate 默认创建 implicit thread-local temp storage。
   - 让 bufferization 按 `BufferStorage + StorageLayout + Lifetime` 分配 addressable storage。
   - 对 register path 保持 SSA/scalar/register fragment，不进入 addressable buffer pool。
   - 对 smem path 分配 block-local storage，并做 lifetime slot reuse。

4. Tile-aware affine IO lowering
   - 改造 `NTTAffineIOLoweringPass` 使用 tile-local domain、owner/local offset 和 storage layout。
   - 确保 mask/tail 语义正确。
   - 保持 direct distributed gather/scatter，不引入 boxing/TensorLoad fallback。

5. Register-resident vector add
   - vector add elementwise intermediate 走 register/SSA path。
   - `thread_local_data_storage` 不因 elementwise intermediate 增长。
   - forced compile 通过，PTX/SASS 无对应 `.local` memory。
   - 16M 性能满足 `>= 356 GB/s` 护栏。

6. Block-local smem microkernel
   - 增加 block 内复用 microkernel。
   - 生成 `Usage=Temp/Scratch/Staging, Scope=BlockLocal, Location=GpuSMem`。
   - 验证 `.shared` 或等价 shared-memory artifact、同步点和 runtime correctness。
   - 记录 global transaction 或 runtime 指标，与 non-tiled 版本对比。

7. Cost model 和 telemetry hook
   - 将 tile shape、ownership、storage、bytes、reuse count、estimated traffic、register/smem usage 写入 dump/telemetry。
   - 先不做全自动 search，但为后续 cost model/autotune 保留结构化输入。

## 实现备注

- 代码中不应出现 `AC-*`、`M1/M2` 等 plan 术语；这些只用于文档和验收跟踪。
- 所有 `dotnet test` 命令必须带 `-s test.runsettings`。
- 所有 tutorial/test dump 必须写到仓库 `dump/` 目录，不写 `/tmp`。
- 需要访问 GPU 的 review、benchmark 和端到端验收不能使用 sandbox；humanize/Codex review 必须设置 `HUMANIZE_CODEX_BYPASS_SANDBOX=true`。
- 遇到 unsupported layout、capacity、mask、dynamic shape、ownership 或 codegen path 时必须 fail fast，并输出可定位根因。
- 不要在 codegen 里用 pattern workaround 修性能问题；先让 IR、layout、storage 和 verifier 表达正确，再做 lowering 和优化。
