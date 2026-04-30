# 03 Tiling Requirements

本文记录下一阶段 enable tiling 的目标边界和验收方案。核心目标是让中间数据按 tile 进入 CUDA 内存层级：可复用的 block tile 放到 smem，可 thread-private 的小 tile 留在寄存器/SSA 值中，避免把 transient intermediate 物化到 gmem 或 `.local` memory。

## 当前状态

当前 CUDA pipeline 已经能把 Triton load/store 转为 `IR.Affine.Gather/Scatter`，经过 `AutoDistributedPass` 后直接进入 `TIRPass`，再由 `NTTAffineIOLoweringPass` 降成 buffer loop 和 NTT CUDA codegen。vector add 已能通过，并且 large-size 性能接近 `torch.add_out` 基线：

- 1M: FlagLang `0.049152 ms`, `256.00 GB/s`; torch `0.036864 ms`, `341.33 GB/s`
- 16M: FlagLang `0.536576 ms`, `375.21 GB/s`; torch `0.518144 ms`, `388.55 GB/s`

但这条路径还没有真正的 tiling 和内存层级选择：

- `src/Nncase.Compiler/Compiler.cs` 中 `AutoTilingPass` 仍在主 `CompileAsync` 中注释掉，普通 Python/Triton 编译不会经过 tiler。
- `src/Nncase.Schedule/Transforms/TIRSelectionPass.cs` 的 `CreateOutputBuffer` 默认把非 entry 中间结果建成 `MemoryLocation.Data`；非 entry caller 追加输出 buffer 时也固定用 `IR.F.Buffer.Uninitialized(..., MemoryLocation.Data, ...)`。
- `src/modules/Nncase.Modules.NTT/Passes/NTTAffineIOLoweringPass.cs` 对 `AffineGather/Scatter` 只继承已有 buffer 的 memory location；当 source 不是 readable buffer 时，`RequireReadableBuffer` 也会新建 `MemoryLocation.Data` buffer。
- 当前 `MemoryLocation` 把三类事实混在一起：usage 包含 `Rdata` / `Data` / `Input` / `Output`，visibility 包含 `ThreadLocal` / `WarpLocal` / `BlockLocal`，physical location 又用 `Cache` / `Data` 等名字隐式表达。这会让 tiling 和 bufferization 无法正确决策，也让 codegen 难以区分 register、smem、tmem、gmem。
- 当前 `DistributedType` 的 `SBP.S(axis)` 主要表达均分式 split，不能覆盖 Triton blocked layout 中常见的 strided per-thread ownership、lane/warp permutation、CTA order，也不能和 smem bank swizzle、register fragment layout 清晰分工。
- 当前 CUDA dump 中可见 `flaglang_thread_local_data_storage[...]`，这种 address-taken per-thread byte array 很容易在 PTX 中表现为 `.local` load/store；它不是寄存器级表达。
- `MemoryLocation.BlockLocalData` / `WarpLocalData` / `Cache` 已存在，runtime/codegen 也有 `block_local_data`、`warp_local_data`、`tar::get_cache_address<level>()` 等入口，但当前 pipeline 没有根据 tile/lifetime/reuse 做自动选择。
- `AutoTilePass` 和 `GraphTiler` 已有旧实现：它能基于 `IR.Affine.Grid` 求解 tile、生成 `MemoryLocation.Cache` physical buffer，并有 `DumpFlags.Tiling`。问题是它没有接到当前 Triton direct `Affine.Gather/Scatter` 路线，且 `MemoryLocation.Cache` 到 CUDA smem/register 的代码生成语义还未作为验收闭环固定下来。

## 目标

### G0: 拆分 buffer usage、visibility 和 physical location

Buffer IR 必须把三个维度拆开表达，不能继续用一个 `MemoryLocation` 同时编码语义用途、可见范围和物理存储位置：

- Usage: buffer 的语义用途，例如 `Input`、`Output`、`Rdata/Const`、`Temp`、`Scratch`。usage 只回答“这块数据为什么存在”，不回答放在哪、谁可见。
- Visibility: buffer 的可见范围，例如 `ThreadLocal`、`WarpLocal`、`BlockLocal`、`Grid/Global`。visibility 决定别名边界、同步需求和 distributed/thread mapping。
- Physical location: buffer 的真实存储层级。CPU 至少区分 `Register`、`L1`、`L2`、`GMem/DRAM`；GPU 至少区分 `Register`、`SMem`、`TMem`、`GMem`。`.local` memory 应作为 register spill 或 address-taken local array 的诊断结果，而不是 tiling 主动选择的目标位置。

目标接口可以是 `BufferStorage { Usage, Visibility, Location, Scope/Hierarchy, Alignment }` 或等价结构。`IR.Buffers.Uninitialized`、`TIR.PhysicalBuffer`、`TIR.Buffer`、bufferize schedule result 和 codegen visitor 都需要消费同一个三元模型。

基本合法性矩阵：

- GPU `Register`: visibility 必须是 thread-local，通常不应有 addressable `MemSpan`；优先表示为 SSA/scalar/register fragment。
- GPU `SMem`: visibility 是 block-local 或明确的 warp/block scope，需要容量检查和同步边界。
- GPU `TMem`: 预留给 tensor-core/tensor-memory 路线；本阶段可以只建模和 fail fast，不要求完整 codegen。
- GPU `GMem`: 用于 input/output/persistent global data；不允许作为 transient intermediate 的默认 fallback。
- CPU `Register`: 用于 scalar/thread-private temporary；CPU `L1/L2` 多数情况下是 cache policy 或 allocation hint，不应伪装成 GPU smem。

### G1: 扩展 DistributedType 为 ownership map

`DistributedType` 不应只依赖 `NdSBP` 的简单均分语义。目标是引入 `DistributionLayout`，让 `SBP.S/B/P/R` 成为简单 case 的 sugar：

```text
DistributedType:
  TensorType
  Placement
  DistributionLayout

DistributionLayout:
  GlobalToOwnerLocal: global logical index -> owner coordinates + local logical index
  OwnerLocalToGlobal: owner coordinates + local logical index -> global logical index
  LocalShape
  ValidPredicate
```

核心边界：

- `DistributedType / DistributionLayout` 只回答“哪个 execution owner 拥有哪个 logical element”。
- `StorageLayout / BufferLayout` 只回答“owner 本地的 logical element 如何物理表达”，例如 contiguous/vectorized stride、register fragment、smem swizzle、bank layout、alignment。
- 改变 thread/lane ownership 的 swizzle 属于 `DistributionLayout`；只改变 smem 地址和 bank 排布的 swizzle 属于 `StorageLayout`。
- `S(axis)`、broadcast、partial、replicate 都应 lowering 成 `DistributionLayout`，便于后续 pass 只消费统一 map。

Triton blocked layout 应作为 `DistributionLayout` preset，而不是散落在 codegen 特例中：

```text
TritonBlockedLayout:
  sizePerThread
  threadsPerWarp
  warpsPerCTA
  order
  ctasPerCGA
  ctaSplitNum
  ctaOrder
```

它展开后形成：

```text
global index
  -> CTA tile
  -> warp id
  -> lane id
  -> element id inside thread
```

Map 表达能力需要受限但足够通用：affine、floordiv、mod、常见 power-of-two permutation、xor swizzle、MMA fragment layout 等。非 affine permutation 不要伪装成 affine，可建成 named map primitive；每个 primitive 必须提供 forward、inverse、domain、cost features 和 codegen lowering。无法证明 inverse 或合法域时 fail fast。

### G2: 在主编译流程中启用 tile 规划

- `Compiler.CompileAsync` 必须在 `AutoDistributedPass` 之后、`TIRPass` 之前运行 tiling 规划，或在等价位置插入能看见 distributed affine dataflow 的 tiling pass。
- Tiling 输入必须是纯数据依赖 DAG；不得因为历史 `Sequential` 或隐藏 side-effect 阻断 egraph/dataflow 优化。
- Pass 需要支持当前 Triton 路线的 `IR.Affine.Gather/Scatter`、`DistributionLayout`、`StorageLayout`、`ProgramIdDim`、`ThreadId`、mask constraint 和 local shard shape。

### G3: 中间数据显式绑定内存层级

- 对 block 内多线程复用的数据，tiling 结果必须生成 block scope tile buffer，并映射到 CUDA smem。
- 对 thread-private、小型、无 address-escape 的临时 tile，tiling 结果必须保留为 SSA/scalar 或 register-fragment 形式；不能把它表达成 address-taken `thread_local_data` byte array 后期待 CUDA 编译器自动救回来。
- 对需要跨 warp 或跨 block 的数据，必须显式说明其 scope 和同步边界；不能静默退回 `MemoryLocation.Data`。
- `MemoryLocation.Data` 只允许作为明确的 global/runtime data pool 语义，不允许作为 tiling 失败后的兜底中间缓存。

### G4: 建立 tile、lifetime、capacity 的统一决策

- Tiling pass 要产出每个 intermediate 的 tile shape、producer/consumer tile relation、memory scope、lifetime 和 byte size。
- Tiling 决策必须同时给出 usage、visibility 和 physical location；例如同样是 temp，thread-private elementwise tile 应走 register，block-reused tile 应走 smem，跨 kernel persistent 数据才允许走 gmem。
- Tiling 决策必须组合 `DistributionLayout` 和 `StorageLayout`：先决定 tile 的 owner/local logical domain，再决定 owner 本地的 register/smem/tmem/gmem 表达。
- smem/register 预算必须来自 target options 或 CUDA device 属性，并进入 capacity check。
- lifetime 分析要能复用 smem slot，但复用必须由 schedule/lifetime 证明，不能依赖命名或 pass 顺序偶然成立。
- 当 tile shape、dynamic bound、alias、capacity 或 synchronization 无法证明时，编译必须 fail fast，输出定位信息。

### G5: Affine IO lowering 变成 tile-aware

- `NTTAffineIOLoweringPass` 需要按 tile-local domain lower `AffineGather/Scatter`，而不是总是对完整 local shard 建 buffer loop。
- `Gather -> compute -> Scatter` 的简单链路应能生成每 tile 的 load/compute/store，不产生完整 `[BLOCK_SIZE]` 中间 Data buffer。
- mask constraint 必须进入 tile 内 guard 或 tail peeling；不能因为 mask 存在而退回全量临时 buffer。
- distributed gather/scatter 的 global offset、local shard offset、thread split offset 必须继续正确组合。

### G6: CUDA codegen 产物可验证

- smem tile 路径的 CUDA source/PTX 必须有可审查证据：`__shared__`、block-local runtime buffer 被证明位于 shared memory，或 PTX/SASS 中出现 `.shared` 访问。
- register tile 路径不能生成 address-taken local byte array；PTX 中不应出现与该 intermediate 对应的 `.local` load/store。
- Codegen 必须按三元模型分派：usage 决定 const/input/output ABI，visibility 决定参数/同步/scope，physical location 决定 CUDA storage class 或表达形态。不能用 `Rdata/Data/BlockLocalData` 这种组合枚举直接推导所有行为。
- Codegen 必须按组合映射生成访问：

```text
global logical index
  -> DistributionLayout: owner coordinates + local logical index
  -> tile/schedule map: tile coordinate + inner coordinate
  -> StorageLayout: physical offset / register fragment / smem swizzled address
```

- 需要把 tiling dump、schedule dump、CUDA source、PTX/SASS 全部保存在 repo `dump/` 下，方便定位。

## 非目标

- 不在本阶段实现完整 Triton matmul/tensor core/async copy/TMA/warp specialization。
- 不用手写 vector-add 特例、手写 PTX、或者 Python backend pattern shortcut 来证明 tiling。
- 不要求替代 CUDA 编译器的寄存器分配器；本阶段只要求 compiler IR/codegen 不主动把 register-eligible intermediate 变成 address-taken local memory。
- 不要求一次性完成全自动 cost model 搜索；可以先用确定性 tile policy，但 IR 和 dump 必须保留 cost model 所需特征。
- 不做跨 kernel、跨 block 的全局缓存优化，也不引入新的 collective 语义。
- 不允许 fallback/workaround：tiling 失败不能静默走 `MemoryLocation.Data`、gmem、local mem 或旧 non-tiled pipeline。

## 现有代码与目标差距

| Area | Existing | Gap | Required change |
| --- | --- | --- | --- |
| Buffer model | `MemoryLocation` 混合 usage、visibility、physical location | tiling/bufferization/codegen 无法独立决策 register、smem、gmem 和同步范围 | 引入 `Usage x Visibility x Location` 三元模型，并迁移 IR/TIR/bufferize/codegen |
| Distribution model | `NdSBP.S(axis)` 主要表达均分 split | 无法表达 strided ownership、Triton blocked layout、lane/warp permutation | 引入双向 `DistributionLayout`，让 SBP 成为 sugar |
| Storage layout | shape/stride 和 distributed local shard 混在 buffer/type 里 | 无法表达 smem swizzle、bank layout、register fragment、tmem layout | 新增 local-domain `StorageLayout`，只描述 owner 本地物理表示 |
| Pass pipeline | `AutoDistributedPass` 默认运行，`AutoTilingPass` 在 `CompileAsync` 中注释 | 主 Python/Triton 路线完全跳过 tiler | 启用 tiling pass，并保证它消费当前 distributed affine DAG |
| Tiling IR 输入 | 旧 `AutoTilePass` 主要处理 `IR.Affine.Grid` | 当前 Triton path 是 direct `IR.Affine.Gather/Scatter` | 增加 direct affine IO tiling，或在不丢语义的前提下统一到 tileable grid/dataflow |
| Buffer allocation | `TIRSelectionPass` 默认 `MemoryLocation.Data` | 中间结果先被物化，再谈优化，容易进入 thread-local/local memory | 在 TIR selection 前确定 tile memory scope，或让 selection 接收 schedule annotation |
| Affine IO lowering | lowering 按现有 buffer shape/load-store loop 生成代码 | 没有 tile-local buffer、tile loop、smem/register placement | 让 lowering 使用 tile domain、tile offset 和 chosen memory scope |
| Memory hierarchy | enum 已有 `WarpLocalData`、`BlockLocalData`、`Cache` | 自动选择缺失；`Cache` 到 CUDA smem/register 语义未验收 | 固定 CUDA scope 映射，并把 capacity/lifetime 接到 bufferize/codegen |
| Register path | 小中间结果通常仍是 buffer/tensor view | address-taken byte array 可能变成 `.local` | 对 thread-private tile 做 scalarization/SSA/register fragment lowering |
| Smem path | runtime 有 `block_local_data` 和同步插入 | normal vector-add dump 中 block pool 为 1 byte，未用于 tile caching | 生成真实 block tile storage，并验证 shared-memory PTX |
| Diagnostics | 现有代码多处默认 `Data` 或通用 buffer | 失败原因会被隐藏成慢路径 | 对 unsupported tiling shape/scope/capacity 直接抛 diagnostic |

## 验收标准

### A0: 静态防回归

- 主 `CompileAsync` 中存在 tiling 阶段，且位于 `AutoDistributedPass` 之后、`TIRPass` 之前或有明确等价理由。
- 对当前 Triton vector-add pipeline，dump 中出现 tiling pass 输出；没有命中旧 cache。
- 对 tiling-eligible intermediate，不允许出现无诊断的 `MemoryLocation.Data` fallback。
- 新 buffer API 或 annotation 明确暴露 usage、visibility、physical location；新增 code 不允许继续引入把这三个维度拼到同一个 enum value 的路径。
- `DistributedType` 新增或等价暴露 `DistributionLayout`；新增 code 不允许只靠 `SBP.S(axis)` 表达 Triton blocked/strided ownership。

### A1: IR/TIR 验收

- 新增 targeted test 覆盖 `Gather -> Binary -> Scatter`：
  - AutoDist 后仍保留 distributed gather/scatter 语义。
  - Tiling 后出现 tile domain/tile offsets/tile memory scope、`DistributionLayout` 和 `StorageLayout`。
  - 每个 tile intermediate 都能 dump 出 `Usage`、`Visibility`、`Location`。
  - TIR 后不再产生完整 local shard 大小的中间 Data buffer。
- 新增 capacity negative test：当 tile bytes 超过 register/smem budget 时编译失败，并报告 op、tile shape、requested bytes、budget bytes。
- 新增 unsupported-shape negative test：dynamic shape 或 non-affine relation 无法证明时 fail fast，不能走 non-tiled 成功路径。

Suggested command:

```bash
dotnet test src/Nncase.Tests/Nncase.Tests.csproj -s test.runsettings \
  --filter "FullyQualifiedName~Tiling|FullyQualifiedName~AffineGather|FullyQualifiedName~AffineScatter"
```

### A2: Distribution/Layout verifier 验收

- 对每个 `DistributionLayout`，verifier 检查 `OwnerLocalToGlobal(GlobalToOwnerLocal(i)) == i`，并检查 owner 坐标落在 `Placement` hierarchy 范围内。
- `ValidPredicate` 必须覆盖 tail/mask，不能用越界 local index 表示 padding。
- `StorageLayout` 的 logical domain 必须等于 `DistributionLayout` 推导出的 local shard domain，或有显式 tile-local view map 解释差异。
- producer/consumer 的 `DistributionLayout` 不一致时，必须显式插入 redistribute/reshard；不能让 codegen 暗中换布局。
- 新增 Triton blocked layout fixture：至少覆盖 contiguous-per-thread 和 strided-per-thread 两种 ownership，dump 中能看到不同的 `GlobalToOwnerLocal`/`OwnerLocalToGlobal`。

### A3: Register tile 验收

- vector add 的 elementwise intermediate dump 为 `Usage=Temp, Visibility=ThreadLocal, Location=GpuRegister` 或等价表达，并以 SSA/scalar/register fragment lowering，不生成对应的 `thread_local_data` tensor buffer。
- `CodeGen/cuda/thread_main.cu` 中 `flaglang_thread_local_data_storage` 不因 register-eligible elementwise intermediate 增长。
- PTX/SASS 检查不应出现与该 intermediate 对应的 `.local` load/store；如 CUDA 编译器因真实寄存器压力 spill，需要通过 register count/occupancy 解释，而不是 IR 主动 address-taken 导致。

Suggested run:

```bash
TRITON_ALWAYS_COMPILE=1 \
TRITON_KERNEL_DUMP=1 \
TRITON_DUMP_DIR=dump/tiling-vector-add \
python python/tutorials/01-vector-add.py --only_unit_test
```

### A4: Smem tile 验收

- 构造一个有 block 内复用的最小 CUDA test，例如两个 consumer 复用同一 `AffineGather` tile，或小型 tiled transpose/stencil。
- Tiling 后该 intermediate dump 为 `Usage=Temp/Scratch, Visibility=BlockLocal, Location=GpuSMem` 或等价表达，而不是每 thread 重复 global load 或落到 local memory。
- smem bank swizzle 必须出现在 `StorageLayout`，不能改变 `DistributionLayout` 的 owner 语义。
- CUDA source/PTX/SASS 中能看到 shared-memory 证据：`__shared__`、已证明位于 shared memory 的 block-local runtime buffer，或 `.shared` 指令。
- 对使用 block-local storage 的 kernel，必要同步点必须可见且由数据依赖解释。

### A5: 端到端正确性与性能门禁

- `python/tutorials/01-vector-add.py --only_unit_test` 强制重编译通过，最大误差为 `0.0`。
- vector-add 16M 性能不应比当前 direct distributed baseline 退化超过 5%，即不低于约 `356 GB/s`，除非 dump 证明本次开启了额外诊断/非优化编译选项。
- 新增 smem 复用 microkernel 的 global memory transaction 或 runtime 指标应优于 non-tiled 版本；至少 dump/PTX 必须证明中间数据没有走 gmem/local mem。
- 所有 dump 保存在 `dump/tiling-*` 下，不能写 `/tmp`。
- 需要访问 GPU 的 review、benchmark 和端到端验收不能运行在 sandbox 中；humanize/Codex review 必须显式设置 `HUMANIZE_CODEX_BYPASS_SANDBOX=true`，否则 GPU 不可见，结果无效。

### A6: Fail-fast 验收

- 禁用 smem/register tiling 所需的 scope mapping 后，tiling-eligible kernel 必须编译失败，不能自动回退到 `MemoryLocation.Data`。
- 遇到 unsupported `DistributionLayout`、缺失 inverse map、layout domain 不匹配、或无法证明 producer/consumer ownership 时必须 fail fast。
- 破坏 smem capacity 配置或设置过小 budget 时，必须报 capacity diagnostic。
- 破坏 affine relation 或 mask constraint 时，必须报 relation/tileability diagnostic。
- 任何 fallback path 都需要删除或改成显式 error；测试要覆盖这些错误路径。

## 分阶段建议

### M1: 建立 buffer/distribution/layout 模型

- 引入 `BufferStorage` 或等价 annotation，拆分 usage、visibility、physical location。
- 引入 `DistributionLayout` 和 `StorageLayout`，让 `SBP.S/B/P/R` lowering 到统一 ownership map。
- 先实现 verifier 和 dump，防止新模型退化成另一个大 enum。

### M2: 接通 tiling 阶段和诊断

- 让主 pipeline 运行 tiling pass，并对当前 direct affine IO path 产生 dump。
- 暂时只接受可证明的一维 elementwise tile；其他 case fail fast。
- 增加 negative tests 防止静默 `Data` fallback。

### M3: Register-resident elementwise tile

- 对 thread-private elementwise chain 做 tile-local SSA/scalar lowering。
- vector add 不再为 elementwise intermediate 分配 `thread_local_data`。
- 用 PTX/SASS 检查 `.local` 回归。

### M4: Block smem tile

- 支持 block scope tile buffer、lifetime scheduling 和 synchronization。
- 增加 block reuse microkernel，并验证 `.shared` 或等价 smem 证据。
- 接入 smem capacity 和 occupancy 约束。

### M5: Cost model/autotune hook

- 把 tile shape、bytes、reuse count、estimated traffic、register/smem usage 写入 dump/telemetry。
- 后续让 cost model 在有限候选上选择 tile policy，但仍保持所有 unsupported case fail fast。

## 风险与门禁

- 风险：旧 `AutoTilePass` 依赖 `IR.Affine.Grid`，直接开启可能对当前 `Affine.Gather/Scatter` 无效。
  - 门禁：先用 dump 证明 tiling 输入包含当前 Triton kernel 的真实 dataflow。
- 风险：`DistributionLayout` 和 `StorageLayout` 边界不清，会重新混淆 ownership 与 physical representation。
  - 门禁：verifier 必须检查 owner/local map 和 storage local domain，producer/consumer layout 不一致时必须显式 reshard。
- 风险：`MemoryLocation.Cache` 的 CUDA codegen 路径和当前 Triton cubin path 不一致。
  - 门禁：smem 验收必须看 CUDA source/PTX/SASS，不只看 IR memory enum。
- 风险：寄存器是否实际分配由 CUDA 编译器决定。
  - 门禁：IR/codegen 不能生成 address-taken local buffer；PTX `.local` 只能作为真实 spill 诊断处理。
- 风险：tiling 可能降低 vector add 性能，因为 vector add 本身没有 block reuse。
  - 门禁：vector add 主要验证 register path 和不退化，smem 收益用有复用的 microkernel 验证。
