import itertools
import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.jit import MockTensor, compute_cache_key


def test_compute_cache_key_normalizes_unhashable_constexprs():
    cache = {}
    options = {"num_warps": 4}
    specialization_a = [("constexpr", {"shape": [1, 2, 3], "flags": {"b": True, "a": False}})]
    specialization_b = [("constexpr", {"flags": {"a": False, "b": True}, "shape": [1, 2, 3]})]

    key_a = compute_cache_key(cache, specialization_a, options)
    key_b = compute_cache_key(cache, specialization_b, options)

    assert key_a == key_b
    assert len(cache) == 1


def test_mock_tensor_stride_matches_contiguous_layout():
    assert MockTensor(torch.float32, shape=[]).stride() == ()
    assert MockTensor(torch.float32, shape=[4]).stride() == (1, )
    assert MockTensor(torch.float32, shape=[2, 3]).stride() == (3, 1)
    assert MockTensor(torch.float32, shape=[2, 3, 4]).stride() == (12, 4, 1)
    assert MockTensor(torch.float32, shape=[2, 3, 4, 5]).stride() == (60, 20, 5, 1)


def test_pre_call_hooks(device):

    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    class MyTensor(torch.Tensor):
        pass

    def my_hook(*args, **kwargs):
        for arg in itertools.chain(args, kwargs.values()):
            if isinstance(arg, MyTensor):
                raise Exception("MyTensor is not allowed")

    add_kernel.add_pre_run_hook(my_hook)

    x = torch.randn(4, device=device)
    y = MyTensor(x)
    out = torch.zeros_like(x)
    with pytest.raises(Exception):
        add_kernel[(4, )](x, y, out, 4, 4)
