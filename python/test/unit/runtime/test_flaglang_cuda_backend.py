import json
from pathlib import Path

import pytest

import triton
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget, Language
from triton.backends.nvidia import compiler as nvidia_compiler
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


def _native_metadata(name="native_entry", argument_order=None):
    if argument_order is None:
        argument_order = ["param_0", "param_1", "param_2", "param_3"]
    return {
        "name": name,
        "cuda_compiler": "nvcc",
        "cuda_arch": "sm_80",
        "flaglang_abi": {
            "entry": name,
            "wrapped_entry": "primfunc_0",
            "argument_order": argument_order,
            "wrapper": "triton_raw_args_to_thread_main",
        },
    }


def test_make_cubin_rejects_non_ptx_artifact():
    backend = _cuda_backend()
    options = backend.parse_options({})

    with pytest.raises(TypeError, match="expected PTX text or native CUDA compilation result"):
        backend.make_cubin(object(), {}, options, 80)


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

    class FakeNativeModule:
        def get_entry_func_name(self):
            return "primfunc_0"

    def fake_compile_to_cubin(src, helper_options):
        captured["src"] = src
        captured["options"] = helper_options
        return {
            "cubin": _fake_cubin_with_symbol(b"block_entry"),
            "metadata": _native_metadata("block_entry"),
            "asm": {"ntt_cu": "__global__ void block_entry() {}"},
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
    assert captured["options"]["enable_auto_dist"] is False
    assert "ntt_cu" in captured["options"]["stage_names"]
    assert "compiler_log" in captured["options"]["stage_names"]


def test_native_compile_result_cache_artifacts_uses_truthful_stage_names():
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=_fake_cubin_with_symbol(b"block_entry"),
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
    metadata = {"name": "placeholder"}

    class FakeNativeModule:
        pass

    def fake_compile(src, metadata, opt, capability):
        assert isinstance(src, FakeNativeModule)
        assert capability == 80
        return nvidia_compiler.NativeCudaCompilation(
            cubin=_fake_cubin_with_symbol(),
            metadata=_native_metadata(),
            asm={"ntt_cu": "__global__ void native_entry() {}"},
            compiler_log="nvcc --gpu-architecture=sm_80",
        )

    monkeypatch.setattr(nvidia_compiler, "_compile_native_module_to_cubin", fake_compile)

    result = backend.make_ptx(FakeNativeModule(), metadata, options, 80)
    assert isinstance(result, nvidia_compiler.NativeCudaCompilation)

    cubin = backend.make_cubin(result, metadata, options, 80)
    assert cubin == _fake_cubin_with_symbol()
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


def test_native_compile_metadata_requires_abi_contract():
    metadata = {}
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=_fake_cubin_with_symbol(),
        metadata={"name": "native_entry"},
        asm={},
    )

    with pytest.raises(KeyError, match="flaglang_abi"):
        nvidia_compiler._apply_native_cuda_metadata(metadata, result, _cuda_backend().parse_options({}))


def test_native_compile_metadata_rejects_abi_entry_mismatch():
    metadata = {}
    bad_metadata = _native_metadata("native_entry")
    bad_metadata["flaglang_abi"]["entry"] = "other_entry"
    result = nvidia_compiler.NativeCudaCompilation(
        cubin=_fake_cubin_with_symbol(),
        metadata=bad_metadata,
        asm={},
    )

    with pytest.raises(ValueError, match="ABI entry"):
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
            cubin=_fake_cubin_with_symbol(),
            metadata=_native_metadata(),
            asm={},
        )

    monkeypatch.setattr(backend, "make_ttir", fake_make_ttir)
    monkeypatch.setattr(nvidia_compiler, "_compile_native_module_to_cubin", fake_compile)

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
