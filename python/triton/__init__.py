"""isort:skip_file"""
__version__ = '3.5.1'

# ---------------------------------------
# Note: import order is significant here.

# initialize dotnet
import os


def _require_compiler_file(path, source):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nncase.Compiler.dll from {source} does not exist: {path}")
    return path


def _resolve_compiler_path(libtriton):
    compiler_path = os.getenv("NNCASE_COMPILER")
    if compiler_path:
        return _require_compiler_file(os.path.abspath(compiler_path), "NNCASE_COMPILER")

    extension_dir = os.path.dirname(libtriton.__file__)
    compiler_path = os.path.join(extension_dir, "nncase", "Nncase.Compiler.dll")
    return _require_compiler_file(compiler_path, "the Python package")


def _initialize_dotnet():
    import triton._C.libtriton as libtriton
    compiler_path = _resolve_compiler_path(libtriton)
    libtriton.hosting.initialize(compiler_path)


_initialize_dotnet()

# submodules
from .runtime import (
    autotune,
    Config,
    heuristics,
    JITFunction,
    KernelInterface,
    reinterpret,
    TensorWrapper,
    OutOfResources,
    InterpreterError,
    MockTensor,
)
from .runtime.jit import constexpr_function, jit
from .runtime._async_compile import AsyncCompileMode, FutureKernel
from .compiler import compile, CompilationError
from .errors import TritonError
from .runtime._allocation import set_allocator

from . import language
from . import testing
from . import tools

must_use_result = language.core.must_use_result

__all__ = [
    "AsyncCompileMode",
    "autotune",
    "cdiv",
    "CompilationError",
    "compile",
    "Config",
    "constexpr_function",
    "FutureKernel",
    "heuristics",
    "InterpreterError",
    "jit",
    "JITFunction",
    "KernelInterface",
    "language",
    "MockTensor",
    "must_use_result",
    "next_power_of_2",
    "OutOfResources",
    "reinterpret",
    "runtime",
    "set_allocator",
    "TensorWrapper",
    "TritonError",
    "testing",
    "tools",
]

# -------------------------------------
# misc. utilities that  don't fit well
# into any specific module
# -------------------------------------


@constexpr_function
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@constexpr_function
def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n
