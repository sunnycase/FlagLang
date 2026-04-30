import itertools
import pytest
import torch

import triton
import triton.language as tl
from triton.runtime.jit import (
    MockTensor,
    create_function_from_signature,
    compute_cache_key,
    create_specialize_impl,
)

_CLOSURE_SHADOW_VALUE = 7


class _FakeBackend:

    def get_arg_specialization(self, _arg, _kind, **_kwargs):
        return None

    def parse_options(self, _kwargs):

        class Options:
            pass

        return Options()

    def parse_attr(self, attr):
        return attr


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


def test_tuple_specialization_preserves_opt_out_flags():
    calls = []

    def specialize_extra(arg, kind, **kwargs):
        calls.append((arg, kind, kwargs))
        return (arg, kind, kwargs["align"])

    specialize_impl = create_specialize_impl(specialize_extra)

    assert specialize_impl((1, 2), specialize_value=False) == (("i32", "i32"), (None, None))
    assert calls == []

    assert specialize_impl((16, 32), align=False) == (
        ("i32", "i32"),
        ((16, "int", False), (32, "int", False)),
    )
    assert [call[2]["align"] for call in calls] == [False, False]


def test_dependency_finder_resolves_nonlocals_before_globals():

    def make_kernel():
        _CLOSURE_SHADOW_VALUE = 11

        @triton.jit
        def kernel():
            return _CLOSURE_SHADOW_VALUE

        return kernel

    kernel = make_kernel()
    _ = kernel.cache_key

    used_values = {name: value for (name, _), (value, _) in kernel.used_global_vals.items()}
    assert used_values["_CLOSURE_SHADOW_VALUE"] == 11


def test_none_non_constexpr_argument_remains_runtime_null_pointer():

    @triton.jit
    def kernel(ptr, block: tl.constexpr):
        return ptr

    backend = _FakeBackend()
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)

    bound_args, specialization, options = binder(None, block=8)
    _options, signature, constexprs, attrs = kernel._pack_args(backend, {"block": 8}, bound_args, specialization,
                                                               options)
    runtime_args = tuple(arg for arg, spec in zip(bound_args.values(), specialization) if spec[0] != "constexpr")

    assert specialization == [("*i8", None), ("constexpr", 8)]
    assert signature == {"ptr": "*i8", "block": "constexpr"}
    assert constexprs == {(1, ): 8}
    assert attrs == {}
    assert runtime_args == (None, )


def test_generated_binder_internal_names_do_not_collide_with_kernel_args():

    @triton.jit
    def kernel(options, params, specialize_impl, block: tl.constexpr):
        return options + params + specialize_impl + block

    backend = _FakeBackend()
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)

    bound_args, specialization, options = binder(1, 2, 3, block=4, num_warps=8)

    assert bound_args == {"options": 1, "params": 2, "specialize_impl": 3, "block": 4}
    assert specialization[-1] == ("constexpr", 4)
    assert options == {"num_warps": 8}


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
