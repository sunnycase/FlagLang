import json
from pathlib import Path

import pytest

import triton
import triton.language as tl
from triton.backends import backends
from triton.backends.compiler import GPUTarget
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


def test_make_cubin_rejects_non_ptx_artifact():
    backend = _cuda_backend()
    options = backend.parse_options({})

    with pytest.raises(TypeError, match="expected PTX text"):
        backend.make_cubin(object(), {}, options, 80)


def test_name_only_add_kernel_is_rejected_before_ptxas():
    backend = _cuda_backend()
    options = backend.parse_options({})

    class NamedOnlyModule:
        def get_entry_func_name(self):
            return "add_kernel"

    with pytest.raises(TypeError, match="actual post-TTIR native .*inspection"):
        backend.make_ptx(NamedOnlyModule(), {}, options, 80)


def test_fake_vector_add_descriptor_is_rejected_before_ptxas():
    backend = _cuda_backend()
    options = backend.parse_options({})

    class FakeDescriptorModule:
        _flaglang_vector_add = {"kind": "flaglang.vector_add", "version": 1}
        _flaglang_validated_vector_add = {"kind": "flaglang.vector_add", "version": 1}

        def get_entry_func_name(self):
            return "add_kernel"

    with pytest.raises(TypeError, match="actual post-TTIR native .*inspection"):
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


def test_vector_add_compile_metadata_and_text_dumps_for_non_default_block_size(monkeypatch):
    torch = _torch_cuda()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    kernel = _vector_add_kernel.warmup(
        MockTensor(torch.float32),
        MockTensor(torch.float32),
        MockTensor(torch.float32),
        128,
        BLOCK_SIZE=128,
        grid=(1, ),
        num_warps=4,
    )

    required_fields = [
        "name",
        "shared",
        "num_warps",
        "num_ctas",
        "cluster_dims",
        "tmem_size",
        "global_scratch_size",
        "global_scratch_align",
        "profile_scratch_size",
        "profile_scratch_align",
    ]
    for field in required_fields:
        assert hasattr(kernel.metadata, field)
    assert isinstance(kernel.metadata.cluster_dims, tuple)
    assert kernel.metadata.cluster_dims == (1, 1, 1)
    assert hasattr(kernel.metadata, "flaglang_kernel")
    assert kernel.metadata.flaglang_kernel["block_size"] == 128
    assert kernel.metadata.flaglang_kernel["entry_name"]
    assert "mad.lo.u32 %r5, %r2, 128, %r3;" in kernel.asm["ptx"]

    for stage in ("ttir", "ttgir", "llir"):
        text = kernel.asm[stage]
        assert "<triton._C.libtriton.ir.module object" not in text
        assert kernel.metadata.flaglang_kernel["entry_name"] in text
        assert "Gather((d0)[s0, s1]" in text
        assert "Scatter((d0)[s0, s1]" in text
        assert "descriptor_json" not in text


def test_non_vector_add_named_add_kernel_is_rejected(monkeypatch):
    torch = _torch_cuda()
    monkeypatch.setenv("TRITON_ALWAYS_COMPILE", "1")

    with pytest.raises(TypeError, match="Unsupported native module|descriptor"):
        add_kernel.warmup(
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            MockTensor(torch.float32),
            128,
            BLOCK_SIZE=128,
            grid=(1, ),
            num_warps=4,
        )
