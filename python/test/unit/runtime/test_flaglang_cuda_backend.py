import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import triton
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget, Language
from triton.backends.nvidia import compiler as nvidia_compiler
from triton.compiler import compiler as triton_compiler
from triton.compiler.compiler import CompiledKernel
from triton.runtime.jit import MockTensor


@triton.jit
def _vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x - y
    tl.store(output_ptr + offsets, output, mask=mask)


def _cuda_backend():
    return backends["nvidia"].compiler(GPUTarget("cuda", 80, 32))


def _torch_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for FlagLang CUDA backend warmup tests")
    return torch


def _valid_vector_add_descriptor():
    return {
        "kind": "flaglang.vector_add",
        "version": 1,
        "ir_source": "post_ttir_native_module",
        "entry_name": "primfunc_0",
        "parameter_order": ["param_0", "param_1", "param_2", "param_3"],
        "pointers": ["param_0", "param_1", "param_2"],
        "n_elements_arg": "param_3",
        "block_size": 128,
        "dtype": "float32",
        "element_size": 4,
        "program_id_axis": 0,
        "relation": "s0 * 128 + d0",
        "constraint": "s0 * 128 + d0 < s1",
        "loads": [
            {"source": "param_0", "default": "implicit_zero"},
            {"source": "param_1", "default": "implicit_zero"},
        ],
        "compute": "fadd",
        "store": {"dest": "param_2"},
    }


def _fake_cubin_with_symbol(symbol=b"native_entry"):
    header = bytearray(64)
    header[0:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[6] = 1
    header[7] = 0x41
    return bytes(header) + b"\0.symtab\0" + symbol + b"\0"


def _runtime_argument_order():
    return ["x_ptr", "y_ptr", "output_ptr", "n_elements"]


def _runtime_argument_types():
    return ["*fp32", "*fp32", "*fp32", "i32"]


def _native_arg_type(arg):
    return "i32" if arg.endswith("_3") else "*f32"


def _raw_arg_type(arg):
    return "int32_t" if arg.endswith("_3") else "float*"


def _native_metadata(name="native_entry", argument_order=None, raw_argument_order=None, imported_argument_order=None):
    if argument_order is None:
        argument_order = ["param_0", "param_1", "param_2", "param_3"]
    if raw_argument_order is None:
        raw_argument_order = [f"id_{arg}" for arg in argument_order]
    if imported_argument_order is None:
        imported_argument_order = ["param_0", "param_1", "param_2", "param_3"]
    return {
        "name": name,
        "cuda_compiler": "nvcc",
        "cuda_arch": "sm_80",
        "flaglang_abi": {
            "entry": name,
            "wrapped_entry": "primfunc_0",
            "argument_count": len(argument_order),
            "imported_argument_order": imported_argument_order,
            "imported_argument_types": [_native_arg_type(arg) for arg in imported_argument_order],
            "argument_order": argument_order,
            "argument_types": [_native_arg_type(arg) for arg in argument_order],
            "raw_argument_order": raw_argument_order,
            "raw_argument_types": [_raw_arg_type(arg) for arg in argument_order],
            "runtime_argument_count": len(_runtime_argument_order()),
            "runtime_argument_order": _runtime_argument_order(),
            "runtime_argument_types": _runtime_argument_types(),
            "wrapper": "triton_raw_args_to_thread_main",
        },
    }


def _runtime_metadata():
    return {
        "runtime_argument_count": len(_runtime_argument_order()),
        "runtime_argument_order": _runtime_argument_order(),
        "runtime_argument_types": _runtime_argument_types(),
    }


def _native_entry_source(name="native_entry", raw_argument_order=None, raw_argument_types=None):
    if raw_argument_order is None:
        raw_argument_order = ["id_param_0", "id_param_1", "id_param_2", "id_param_3"]
    if raw_argument_types is None:
        raw_argument_types = ["float*", "float*", "float*", "int32_t"]
    params = ",\n".join(f"{ty} {arg}" for ty, arg in zip(raw_argument_types, raw_argument_order))
    return f'extern "C" __global__ __attribute__((used)) void {name}({params}) {{}}\n'


def _compile_real_cubin(tmp_path, symbol="native_entry"):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("nvcc is required to generate real cubin fixtures")
    source = tmp_path / f"{symbol}.cu"
    cubin = tmp_path / f"{symbol}.cubin"
    source.write_text(f'extern "C" __global__ void {symbol}() {{}}\n')
    result = subprocess.run(
        [nvcc, "--cubin", "-arch=sm_80", str(source), "-o", str(cubin)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        pytest.skip(f"nvcc could not generate cubin fixture: {result.stderr or result.stdout}")
    return cubin.read_bytes()


def _require_cuobjdump():
    try:
        nvidia_compiler.knobs.nvidia.cuobjdump.path
    except RuntimeError as exc:
        pytest.skip(f"cuobjdump is required for native cubin validation tests: {exc}")


def test_make_cubin_rejects_non_ptx_artifact():
    backend = _cuda_backend()
    options = backend.parse_options({})

    with pytest.raises(TypeError, match="expected PTX text or native CUDA compilation result"):
        backend.make_cubin(object(), {}, options, 80)


def test_make_cubin_populates_metadata_for_ptx_text(monkeypatch):
    backend = _cuda_backend()
    options = backend.parse_options({})
    metadata = {"name": "stale_entry", **_runtime_metadata()}
    cubin = _fake_cubin_with_symbol(b"ptx_entry")
    ptx = """
.version 8.0
.target sm_80
.address_size 64

.visible .entry ptx_entry(
    .param .u64 ptx_entry_param_0
)
{
    ret;
}
"""

    def fake_run(cmd, check, close_fds, stderr):
        assert check is True
        assert close_fds is False
        Path(cmd[cmd.index("-o") + 1]).write_bytes(cubin)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(nvidia_compiler, "get_ptxas", lambda: SimpleNamespace(path="ptxas"))
    monkeypatch.setattr(nvidia_compiler.subprocess, "run", fake_run)

    assert backend.make_cubin(ptx, metadata, options, 80) == cubin
    assert metadata["name"] == "ptx_entry"
    assert metadata["shared"] == 0
    assert metadata["tmem_size"] == 0
    assert metadata["global_scratch_size"] == 0
    assert metadata["global_scratch_align"] == 1
    assert metadata["profile_scratch_size"] == 0
    assert metadata["profile_scratch_align"] == 1
    assert metadata["num_warps"] == options.num_warps
    assert metadata["num_ctas"] == options.num_ctas
    assert metadata["cluster_dims"] == (1, 1, 1)


def test_make_cubin_rejects_ptx_text_without_single_entry():
    backend = _cuda_backend()
    options = backend.parse_options({})

    with pytest.raises(ValueError, match="exactly one launchable \\.entry symbol"):
        backend.make_cubin(".visible .func helper() { ret; }", {}, options, 80)


def test_direct_vector_add_ptx_emitter_is_not_available():
    assert not hasattr(nvidia_compiler, "_emit_vector_add_ptx")
    source = Path(nvidia_compiler.__file__).read_text()
    assert "mad.lo.u32" not in source
    assert "flaglang_kernel" not in source


def test_name_only_add_kernel_is_rejected_before_ptxas():
    backend = _cuda_backend()
    options = backend.parse_options({})

    class NamedOnlyModule:
        def get_entry_func_name(self):
            return "add_kernel"

    with pytest.raises(TypeError, match="actual post-TTIR native .*ir\\.module"):
        backend.make_ptx(NamedOnlyModule(), {}, options, 80)


def test_fake_vector_add_descriptor_is_rejected_before_ptxas():
    backend = _cuda_backend()
    options = backend.parse_options({})

    class FakeDescriptorModule:
        _flaglang_vector_add = {"kind": "flaglang.vector_add", "version": 1}
        _flaglang_validated_vector_add = {"kind": "flaglang.vector_add", "version": 1}

        def get_entry_func_name(self):
            return "add_kernel"

    with pytest.raises(TypeError, match="actual post-TTIR native .*ir\\.module"):
        backend.make_ptx(FakeDescriptorModule(), {}, options, 80)


def test_forged_describe_vector_add_json_is_rejected_before_ptxas():
    backend = _cuda_backend()
    options = backend.parse_options({})

    class ForgedDescribeModule:
        def get_entry_func_name(self):
            return "primfunc_0"

        def describe_vector_add(self):
            return json.dumps({"valid": True, "descriptor": _valid_vector_add_descriptor()})

    with pytest.raises(TypeError, match="actual post-TTIR native ir\\.module"):
        backend.make_ptx(ForgedDescribeModule(), {}, options, 80)


def test_native_module_without_compile_helper_fails_closed(monkeypatch):
    backend = _cuda_backend()
    options = backend.parse_options({})

    class FakeNativeModule:
        def get_entry_func_name(self):
            return "primfunc_0"

    monkeypatch.setattr(nvidia_compiler.ir, "module", FakeNativeModule, raising=False)
    monkeypatch.delattr(nvidia_compiler.ir, "compile_to_cubin", raising=False)

    with pytest.raises(RuntimeError, match="native CUDA compile helper is unavailable"):
        backend.make_ptx(FakeNativeModule(), {}, options, 80)


def test_native_compile_helper_receives_parsed_capability_and_options(monkeypatch):
    backend = _cuda_backend()
    options = backend.parse_options({})
    captured = {}
    monkeypatch.setattr(nvidia_compiler, "_validate_native_cubin", lambda cubin, entry_name: None)

    class FakeNativeModule:
        def get_entry_func_name(self):
            return "primfunc_0"

    def fake_compile_to_cubin(src, helper_options):
        captured["src"] = src
        captured["options"] = helper_options
        return {
            "cubin": b"validated-cubin",
            "metadata": _native_metadata("block_entry"),
            "asm": {"ntt_cu": _native_entry_source("block_entry")},
            "compiler_log": "nvcc --gpu-architecture=sm_90a",
        }

    monkeypatch.setattr(nvidia_compiler.ir, "module", FakeNativeModule, raising=False)
    monkeypatch.setattr(nvidia_compiler.ir, "compile_to_cubin", fake_compile_to_cubin, raising=False)

    result = backend.make_ptx(FakeNativeModule(), {}, options, 90)

    assert isinstance(result, nvidia_compiler.NativeCudaCompilation)
    assert captured["src"].get_entry_func_name() == "primfunc_0"
    assert captured["options"]["entry_name"] == "primfunc_0"
    assert captured["options"]["capability"] == 90
    assert captured["options"]["arch"] == "sm_90a"
    assert captured["options"]["threads_per_cta"] == options.num_warps * options.warp_size
    assert captured["options"]["binary_ext"] == "cubin"
    assert captured["options"]["cuobjdump"] == nvidia_compiler.knobs.nvidia.cuobjdump.path
    assert captured["options"]["enable_auto_dist"] is False
    assert "ntt_cu" in captured["options"]["stage_names"]
    assert "compiler_log" in captured["options"]["stage_names"]


def test_native_compile_result_cache_artifacts_uses_truthful_stage_names():
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata=_native_metadata("block_entry"),
        asm={
            "triton_tir": "triton module text",
            "nncase_ir": "nncase module text",
            "after_compile": "compiled module text",
            "tir": "tir module text",
            "ntt_cu": "__global__ void block_entry() {}",
        },
        compiler_log="nvcc --gpu-architecture=sm_80",
    )

    artifacts = result.cache_artifacts()
    assert "ptx" not in artifacts
    assert artifacts["triton_tir"] == "triton module text"
    assert artifacts["compiler_log"] == "nvcc --gpu-architecture=sm_80"
    assert result.suppress_stage_file is True


def test_native_compile_result_supplies_cubin_and_metadata(monkeypatch):
    backend = _cuda_backend()
    options = backend.parse_options({})
    metadata = {"name": "placeholder", **_runtime_metadata()}
    cubin = b"validated-cubin"
    monkeypatch.setattr(nvidia_compiler, "_validate_native_cubin", lambda cubin, entry_name: None)

    class FakeNativeModule:
        pass

    def fake_compile(src, metadata, opt, capability):
        assert isinstance(src, FakeNativeModule)
        assert capability == 80
        return nvidia_compiler.NativeCudaCompilation(
            cubin=cubin,
            metadata=_native_metadata(),
            asm={"ntt_cu": _native_entry_source()},
            compiler_log="nvcc --gpu-architecture=sm_80",
        )

    monkeypatch.setattr(nvidia_compiler, "_compile_native_module_to_cubin", fake_compile)

    result = backend.make_ptx(FakeNativeModule(), metadata, options, 80)
    assert isinstance(result, nvidia_compiler.NativeCudaCompilation)

    cubin = backend.make_cubin(result, metadata, options, 80)
    assert cubin == b"validated-cubin"
    assert metadata["name"] == "native_entry"
    assert metadata["shared"] == 0
    assert metadata["num_warps"] == 4
    assert metadata["cluster_dims"] == (1, 1, 1)
    assert metadata["flaglang_pipeline"]["kind"] == "native_cuda"
    assert metadata["flaglang_pipeline"]["artifact"] == "cubin"
    assert "flaglang_kernel" not in metadata


def test_native_compile_result_rejects_malformed_cubin():
    with pytest.raises(ValueError, match="not a CUDA ELF cubin"):
        nvidia_compiler._normalize_native_cuda_compilation({
            "cubin": b"not-a-cubin",
            "metadata": _native_metadata(),
            "asm": {},
        })


def test_native_compile_result_rejects_elf_like_non_cubin():
    _require_cuobjdump()

    with pytest.raises(ValueError, match="cuobjdump rejected"):
        nvidia_compiler._normalize_native_cuda_compilation({
            "cubin": _fake_cubin_with_symbol(),
            "metadata": _native_metadata(),
            "asm": {},
        })


def test_native_compile_result_accepts_real_cubin_with_entry(tmp_path):
    _require_cuobjdump()
    cubin = _compile_real_cubin(tmp_path, "native_entry")

    result = nvidia_compiler._normalize_native_cuda_compilation({
        "cubin": cubin,
        "metadata": _native_metadata(),
        "asm": {"ntt_cu": _native_entry_source()},
    })

    assert result.cubin == cubin


def test_native_compile_result_rejects_real_cubin_missing_entry(tmp_path):
    _require_cuobjdump()
    cubin = _compile_real_cubin(tmp_path, "other_entry")

    with pytest.raises(ValueError, match="does not contain entry symbol"):
        nvidia_compiler._normalize_native_cuda_compilation({
            "cubin": cubin,
            "metadata": _native_metadata("native_entry"),
            "asm": {"ntt_cu": _native_entry_source()},
        })


def test_native_compile_result_rejects_corrupted_real_cubin(tmp_path):
    _require_cuobjdump()
    cubin = _compile_real_cubin(tmp_path, "native_entry")

    with pytest.raises(ValueError, match="cuobjdump rejected"):
        nvidia_compiler._normalize_native_cuda_compilation({
            "cubin": cubin[:80],
            "metadata": _native_metadata(),
            "asm": {"ntt_cu": _native_entry_source()},
        })


def test_native_compile_metadata_requires_abi_contract():
    metadata = {}
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata={"name": "native_entry"},
        asm={"ntt_cu": _native_entry_source()},
    )

    with pytest.raises(KeyError, match="flaglang_abi"):
        nvidia_compiler._apply_native_cuda_metadata(metadata, result, _cuda_backend().parse_options({}))


def test_native_compile_metadata_rejects_abi_entry_mismatch():
    metadata = _runtime_metadata()
    bad_metadata = _native_metadata("native_entry")
    bad_metadata["flaglang_abi"]["entry"] = "other_entry"
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata=bad_metadata,
        asm={"ntt_cu": _native_entry_source()},
    )

    with pytest.raises(ValueError, match="ABI entry"):
        nvidia_compiler._apply_native_cuda_metadata(metadata, result, _cuda_backend().parse_options({}))


def test_native_compile_metadata_accepts_matching_runtime_abi(monkeypatch):
    metadata = _runtime_metadata()
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata=_native_metadata(),
        asm={"ntt_cu": _native_entry_source()},
    )

    monkeypatch.setattr(nvidia_compiler, "_validate_native_cubin", lambda cubin, entry_name: None)
    cubin = _cuda_backend().make_cubin(result, metadata, _cuda_backend().parse_options({}), 80)

    assert cubin == b"validated-cubin"
    assert metadata["flaglang_abi"]["argument_order"] == ["param_0", "param_1", "param_2", "param_3"]


@pytest.mark.parametrize(
    ("argument_order", "match"),
    [
        (["param_3", "param_2", "param_1", "param_0"], "imported_argument_order"),
        (["param_0", "param_1", "param_2"], "imported_argument_order"),
        (["param_0", "param_1", "param_2", "param_3", "param_4"], "imported_argument_order"),
    ],
)
def test_native_compile_metadata_rejects_abi_argument_order_mismatch(argument_order, match):
    metadata = _runtime_metadata()
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata=_native_metadata(argument_order=argument_order),
        asm={"ntt_cu": _native_entry_source()},
    )

    with pytest.raises(ValueError, match=match):
        nvidia_compiler._apply_native_cuda_metadata(metadata, result, _cuda_backend().parse_options({}))


def test_native_compile_metadata_rejects_self_consistent_reordered_wrapper_abi():
    metadata = _runtime_metadata()
    argument_order = ["param_3", "param_2", "param_1", "param_0"]
    raw_argument_order = ["id_param_3", "id_param_2", "id_param_1", "id_param_0"]
    raw_argument_types = [_raw_arg_type(arg) for arg in argument_order]
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=b"validated-cubin",
        metadata=_native_metadata(argument_order=argument_order, raw_argument_order=raw_argument_order),
        asm={"ntt_cu": _native_entry_source(raw_argument_order=raw_argument_order, raw_argument_types=raw_argument_types)},
    )

    with pytest.raises(ValueError, match="imported_argument_order"):
        nvidia_compiler._apply_native_cuda_metadata(metadata, result, _cuda_backend().parse_options({}))


def test_add_stages_routes_parsed_capability_to_native_compile(monkeypatch):
    backend = _cuda_backend()
    options = backend.parse_options({"arch": "sm90"})
    stages = {}
    captured = {}

    class FakeNativeModule:
        context = object()

        def get_entry_func_name(self):
            return "primfunc_0"

    native_module = FakeNativeModule()

    def fake_make_ttir(mod, metadata, opt, capability):
        captured["ttir_capability"] = capability
        return nvidia_compiler.NativeCudaIRStage(native_module)

    def fake_compile(src, metadata, opt, capability):
        captured["ptx_capability"] = capability
        return nvidia_compiler.NativeCudaCompilation(
            cubin=b"validated-cubin",
            metadata=_native_metadata(),
            asm={"ntt_cu": _native_entry_source()},
        )

    monkeypatch.setattr(backend, "make_ttir", fake_make_ttir)
    monkeypatch.setattr(nvidia_compiler, "_compile_native_module_to_cubin", fake_compile)
    monkeypatch.setattr(nvidia_compiler, "_validate_native_cubin", lambda cubin, entry_name: None)

    backend.add_stages(stages, options, Language.TRITON)
    stage = stages["ttir"](native_module, {})
    stage = stages["ttgir"](stage, {})
    stage = stages["llir"](stage, {})
    stage = stages["ptx"](stage, {})

    assert captured == {"ttir_capability": 90, "ptx_capability": 90}
    assert isinstance(stage, nvidia_compiler.NativeCudaCompilation)
    assert stage.suppress_stage_file is True


def test_compiled_kernel_rejects_missing_launcher_metadata(tmp_path):
    metadata_path = tmp_path / "kernel.json"
    cubin_path = tmp_path / "kernel.cubin"
    metadata = {
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "name": "kernel",
        "num_warps": 4,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
    }
    metadata_path.write_text(json.dumps(metadata))
    cubin_path.write_bytes(b"not-a-real-cubin")

    with pytest.raises(KeyError, match="shared"):
        CompiledKernel(object(), {"kernel.json": metadata_path, "kernel.cubin": cubin_path}, "hash")


def test_compiled_kernel_rejects_malformed_cluster_dims(tmp_path):
    metadata_path = tmp_path / "kernel.json"
    cubin_path = tmp_path / "kernel.cubin"
    metadata = {
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "name": "kernel",
        "shared": 0,
        "num_warps": 4,
        "num_ctas": 1,
        "cluster_dims": [1, 1],
        "tmem_size": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
    }
    metadata_path.write_text(json.dumps(metadata))
    cubin_path.write_bytes(b"not-a-real-cubin")

    with pytest.raises(ValueError, match="cluster_dims"):
        CompiledKernel(object(), {"kernel.json": metadata_path, "kernel.cubin": cubin_path}, "hash")


def test_compiled_kernel_rejects_malformed_target_metadata(tmp_path):
    metadata_path = tmp_path / "kernel.json"
    cubin_path = tmp_path / "kernel.cubin"
    metadata = {
        "target": {"backend": "cuda", "arch": 80},
        "name": "kernel",
        "shared": 0,
        "num_warps": 4,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "tmem_size": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
    }
    metadata_path.write_text(json.dumps(metadata))
    cubin_path.write_bytes(b"not-a-real-cubin")

    with pytest.raises(KeyError, match="target"):
        CompiledKernel(object(), {"kernel.json": metadata_path, "kernel.cubin": cubin_path}, "hash")


def test_compiled_kernel_does_not_require_cuda_metadata_for_other_backends(tmp_path, monkeypatch):
    metadata_path = tmp_path / "kernel.json"
    hsaco_path = tmp_path / "kernel.hsaco"
    metadata = {
        "target": {"backend": "hip", "arch": "gfx90a", "warp_size": 64},
        "name": "kernel",
        "shared": 0,
        "num_warps": 4,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
    }
    metadata_path.write_text(json.dumps(metadata))
    hsaco_path.write_bytes(b"not-a-real-hsaco")

    class FakeBackend:
        binary_ext = "hsaco"

        def pack_metadata(self, metadata):
            assert metadata.target.backend == "hip"
            assert not hasattr(metadata, "tmem_size")
            assert not hasattr(metadata, "global_scratch_size")
            return ("packed", )

    monkeypatch.setattr(triton_compiler, "make_backend", lambda target: FakeBackend())
    kernel = CompiledKernel(object(), {"kernel.json": metadata_path, "kernel.hsaco": hsaco_path}, "hash")

    assert kernel.metadata.target.backend == "hip"
    assert kernel.packed_metadata == ("packed", )


def test_compiled_kernel_surfaces_cuda_driver_load_binary_failure(tmp_path, monkeypatch):
    metadata_path = tmp_path / "kernel.json"
    cubin_path = tmp_path / "kernel.cubin"
    metadata = {
        "target": {"backend": "cuda", "arch": 80, "warp_size": 32},
        "name": "kernel",
        "shared": 0,
        "num_warps": 4,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "tmem_size": 0,
        "global_scratch_size": 0,
        "global_scratch_align": 1,
        "profile_scratch_size": 0,
        "profile_scratch_align": 1,
    }
    metadata_path.write_text(json.dumps(metadata))
    cubin_path.write_bytes(b"malformed-cubin")

    class FailingUtils:
        def get_device_properties(self, device):
            return {"max_shared_mem": 1 << 20}

        def load_binary(self, name, kernel, shared, device):
            raise RuntimeError("driver rejected cubin")

    class FailingDriver:
        utils = FailingUtils()

        def get_current_device(self):
            return 0

        def launcher_cls(self, src, metadata):
            return lambda *args, **kwargs: None

    monkeypatch.setattr(triton_compiler.driver, "_active", FailingDriver())
    kernel = CompiledKernel(object(), {"kernel.json": metadata_path, "kernel.cubin": cubin_path}, "hash")

    with pytest.raises(RuntimeError, match="driver rejected cubin"):
        kernel._init_handles()


def test_vector_add_cannot_fallback_to_direct_ptx_when_helper_is_missing(monkeypatch):
    torch = _torch_cuda()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setattr(nvidia_compiler.ir, "compile_to_cubin", None, raising=False)

    with pytest.raises(RuntimeError, match="native CUDA compile helper is unavailable"):
        _vector_add_kernel.warmup(
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            128,
            BLOCK_SIZE=128,
            grid=(1, ),
            num_warps=4,
        )


def test_non_vector_add_named_add_kernel_is_rejected(monkeypatch):
    torch = _torch_cuda()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setattr(nvidia_compiler.ir, "compile_to_cubin", None, raising=False)

    with pytest.raises(RuntimeError, match="native CUDA compile helper is unavailable"):
        add_kernel.warmup(
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            128,
            BLOCK_SIZE=128,
            grid=(1, ),
            num_warps=4,
        )


def _dump_file(dump_dir, suffix):
    matches = sorted(Path(dump_dir).rglob(f"*.{suffix}"))
    assert matches, f"missing dump artifact '*.{suffix}' under {dump_dir}"
    return matches[-1]


def _dump_text(dump_dir, suffix):
    return _dump_file(dump_dir, suffix).read_text()


def test_native_cuda_vector_add_forced_compile_dump_regression(tmp_path, monkeypatch):
    torch = _torch_cuda()
    dump_dir = tmp_path / "native-dump"
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")
    monkeypatch.setenv("TRITON_KERNEL_DUMP", "1")
    monkeypatch.setenv("TRITON_DUMP_DIR", str(dump_dir))

    n_elements = 4096
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size), )

    def run_once():
        _vector_add_kernel.device_caches.clear()
        x = torch.arange(n_elements, device="cuda", dtype=torch.float32)
        y = torch.arange(n_elements, device="cuda", dtype=torch.float32) * 2
        output = torch.empty_like(x)
        _vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size, num_warps=4)
        torch.cuda.synchronize()
        assert torch.max(torch.abs(output - (x + y))).item() == 0.0

    run_once()
    first_cubin = _dump_file(dump_dir, "cubin")
    first_cubin_mtime = first_cubin.stat().st_mtime_ns

    required_passes = [
        "NativeCudaImportPass",
        "TargetIndependentPass",
        "TargetIndependentQuantPass",
        "TargetDependentPass",
        "QuantizePass",
        "AutoVectorizePass",
        "AutoPackingPass",
        "AutoDistributedPass",
        "TIRPass",
        "TargetDependentBeforeCodeGen",
    ]
    pass_dump_text = _dump_text(dump_dir, "pass_dumps")
    for pass_name in required_passes:
        assert pass_name in pass_dump_text

    native_stage_files = [
        path for path in Path(dump_dir).rglob("*")
        if path.is_file() and "CodeGen" not in path.parts
    ]
    for forbidden_suffix in (".ttir", ".ttgir", ".llir", ".ptx"):
        assert not [path for path in native_stage_files if path.name.endswith(forbidden_suffix)]

    triton_tir = _dump_text(dump_dir, "triton_tir")
    nncase_ir = _dump_text(dump_dir, "nncase_ir")
    after_compile = _dump_text(dump_dir, "after_compile")
    tir = _dump_text(dump_dir, "tir")
    ntt_cu = _dump_text(dump_dir, "ntt_cu")
    compiler_cmd = _dump_text(dump_dir, "compiler_cmd")
    cubin = _dump_file(dump_dir, "cubin").read_bytes()

    for lowered_stage in (nncase_ir, after_compile, tir):
        assert "Triton.Load" not in lowered_stage
        assert "Triton.Store" not in lowered_stage
        assert "IR.Triton.Load" not in lowered_stage
        assert "IR.Triton.Store" not in lowered_stage

    assert triton_tir != nncase_ir
    assert "__global__" in ntt_cu
    assert "flaglang_native_entry" in ntt_cu
    assert "nvcc" in compiler_cmd or "clang" in compiler_cmd
    assert cubin.startswith(b"\x7fELF")

    time.sleep(0.1)
    run_once()
    second_cubin_mtime = _dump_file(dump_dir, "cubin").stat().st_mtime_ns
    assert second_cubin_mtime > first_cubin_mtime
