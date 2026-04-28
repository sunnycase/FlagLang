import functools
import os
import inspect
import shutil
import subprocess
import tempfile

import triton
from triton.compiler import ASTSource, make_backend
from triton.backends.compiler import GPUTarget
from triton.runtime.jit import create_function_from_signature
from triton._C.libtriton import ir

# ===-----------------------------------------------------------------------===#
# filecheck_test
# ===-----------------------------------------------------------------------===#

# Stub target for testing the frontend.
stub_target = GPUTarget("cuda", 100, 32)

_FILECHECK_ENV_VARS = ("FILECHECK_PATH", "LLVM_FILECHECK")


def _resolve_filecheck_path():
    for env_var in _FILECHECK_ENV_VARS:
        configured = os.environ.get(env_var)
        if not configured:
            continue
        path = os.path.abspath(os.path.expanduser(configured))
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{env_var} points to missing FileCheck binary: {path}")
        return path

    candidates = []
    llvm_syspath = os.environ.get("LLVM_SYSPATH")
    if llvm_syspath:
        candidates.append(os.path.join(llvm_syspath, "bin", "FileCheck"))
    llvm_library_dir = os.environ.get("LLVM_LIBRARY_DIR")
    if llvm_library_dir:
        candidates.append(os.path.abspath(os.path.join(llvm_library_dir, os.pardir, "bin", "FileCheck")))

    path = shutil.which("FileCheck")
    if path:
        return path

    llvm_config = shutil.which("llvm-config")
    if llvm_config:
        bindir = subprocess.check_output([llvm_config, "--bindir"], text=True).strip()
        if not bindir:
            raise RuntimeError(f"{llvm_config} --bindir returned an empty path")
        candidates.append(os.path.join(bindir, "FileCheck"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    checked = ", ".join(candidates) if candidates else "no LLVM candidate paths"
    raise FileNotFoundError("Unable to locate FileCheck. Set FILECHECK_PATH or LLVM_FILECHECK, "
                            f"or install FileCheck on PATH. Checked {checked}.")


class MatchError(ValueError):

    def __init__(self, message, module_str):
        super().__init__(message)
        self.module_str = module_str

    def __str__(self):
        return f"{super().__str__()}\n{self.module_str}"


def run_filecheck(name, module_str, check_template):
    with tempfile.TemporaryDirectory() as tempdir:
        temp_module = os.path.join(tempdir, "module")
        with open(temp_module, "w") as temp:
            temp.write(module_str)

        temp_expected = os.path.join(tempdir, "expected")
        with open(temp_expected, "w") as temp:
            temp.write(check_template)

        try:
            subprocess.check_output(
                [_resolve_filecheck_path(), temp_expected, "--input-file", temp_module, "--dump-input-context=50"],
                stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as error:
            decoded = error.output.decode('unicode_escape')
            raise ValueError(decoded)


def run_parser(kernel_fn, args=(), kwargs={}, target=stub_target):
    if "sanitize_overflow" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["sanitize_overflow"] = False
    backend = make_backend(target)
    binder = create_function_from_signature(
        kernel_fn.signature,
        kernel_fn.params,
        backend,
    )

    bound_args, specialization, options = binder(*args, **kwargs)
    options, signature, constexprs, attrs = kernel_fn._pack_args(backend, kwargs, bound_args, specialization, options)
    if kernel_fn.is_gluon():
        from triton.experimental.gluon._runtime import GluonASTSource

        src = GluonASTSource(kernel_fn, signature, constexprs, attrs)
    else:
        src = ASTSource(kernel_fn, signature, constexprs, attrs)

    context = ir.context()
    ir.load_dialects(context)
    backend.load_dialects(context)

    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    assert module.verify()
    return module


def run_filecheck_test(kernel_fn):
    assert isinstance(kernel_fn, triton.runtime.JITFunction)
    check_template = inspect.getsource(kernel_fn.fn)
    if check_template is None:
        raise ValueError("kernel function must have a docstring with FileCheck template")
    mlir_module = run_parser(kernel_fn)

    run_filecheck("placeholder", mlir_module.str_nodebug(), check_template)


def filecheck_test(fn):

    @functools.wraps(fn)
    def test_fn():
        run_filecheck_test(fn)

    return test_fn
