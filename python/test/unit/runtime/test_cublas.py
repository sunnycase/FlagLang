from pathlib import Path

import pytest
import torch
from triton._internal_testing import is_cuda


def _cublas_instance_header():
    return Path(__file__).resolve().parents[4] / "third_party" / "nvidia" / "include" / "cublas_instance.h"


def test_cublaslt_wrapper_loads_cublaslt_library():
    source = _cublas_instance_header().read_text()

    assert 'static constexpr const char *name = "libcublasLt.so";' in source
    assert 'static constexpr const char *name = "libcublas.so";' not in source


def test_cublaslt_gemm_describes_c_operand_as_float16():
    source = _cublas_instance_header().read_text()

    assert "gemm_impl(n, m, k, B, A, 0, C, dtype, dtype, 1.0f, 0.0f);" in source
    assert "gemm_impl(n, m, k, B, A, C, D, dtype, CUDA_R_16F, alpha, beta);" in source
    assert "auto c_dtype = dtype == CUDA_R_8F_E4M3 ? CUDA_R_16F : dtype;" not in source


@pytest.mark.parametrize("m, n, k", [(16, 16, 16), (32, 16, 16), (16, 32, 16), (16, 16, 32)])
@pytest.mark.parametrize("dtype_str", ["float8_e4m3fn", "float16"])
def test_cublas(m, n, k, dtype_str, device):
    dtype = getattr(torch, dtype_str)
    if not is_cuda():
        pytest.skip("test_cublas is only supported on CUDA")
    if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("fp8 is only supported on CUDA with cc >= 90")

    from triton._C.libtriton import nvidia

    torch.manual_seed(123)
    workspace_size = 32 * 1024 * 1024

    def limited_rand(elements, shape):
        total_elems = torch.prod(torch.tensor(shape)).item()
        indices = torch.randint(0, len(elements), (total_elems, ), device=device)
        return elements[indices].view(shape)

    elements = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32, device=device)
    a = limited_rand(elements, (m, k)).to(dtype)
    b = limited_rand(elements, (k, n)).to(dtype)
    c = torch.zeros((m, n), dtype=dtype, device=device)

    b = b.T.contiguous()

    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)

    cublas = nvidia.cublas.CublasLt(workspace)
    cublas.matmul(a, b, c)

    ref = torch.matmul(a.to(torch.float16), b.to(torch.float16).T)

    assert torch.allclose(c.to(torch.float16), ref, atol=2.0)
