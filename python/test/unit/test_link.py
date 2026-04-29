import sys

import pytest
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

try:
    from triton._C.libtriton import llvm
except ImportError:
    pytest.skip(
        "test_link exercises the legacy Triton LLVM/NVVM lowering path; "
        "FlagLang native libtriton does not export libtriton.llvm.",
        allow_module_level=True,
    )


@triton.jit(noinline=True)
def add_one(x_ptr, SQRT: tl.constexpr) -> None:
    x = tl.load(x_ptr)
    if SQRT:
        x = libdevice.sqrt(x)
    tl.store(x_ptr, x + 1.0)


@triton.jit
def add_one_indirect(x_ptr, SQRT: tl.constexpr) -> None:
    add_one(x_ptr, SQRT)


@pytest.mark.parametrize("use_libdevice", (False, True))
@pytest.mark.parametrize("kernel", (add_one, add_one_indirect))
def test_link_extern_libs(use_libdevice, kernel, device):
    if device != "cuda" or not torch.cuda.is_available():
        pytest.skip("test_link requires a CUDA runner with a legacy libtriton.llvm binding")

    link_called: bool = False

    def callback(frame, event, arg):
        nonlocal link_called
        if event == "c_call" and arg is llvm.link_extern_libs:
            link_called = True

    x = torch.ones((1, ), device=device)
    prior_callback = sys.getprofile()
    try:
        sys.setprofile(callback)
        with (compilation := triton.knobs.compilation).scope():
            compilation.always_compile = True
            kernel[(1, )](x, SQRT=use_libdevice)
    finally:
        sys.setprofile(prior_callback)

    assert (link_called == use_libdevice)
