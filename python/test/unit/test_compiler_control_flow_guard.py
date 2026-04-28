import ast
from types import SimpleNamespace

import pytest

import triton
import triton.language as tl
from triton._C.libtriton import ir
from triton.backends.compiler import GPUTarget
from triton.compiler.code_generator import (
    CodeGenerator,
    _NATIVE_CONTROL_FLOW_BUILDER_METHODS,
    _missing_builder_methods,
)
from triton.compiler.errors import UnsupportedLanguageConstruct


@triton.jit
def _dynamic_if_kernel(x, n):
    pid = tl.program_id(0)
    if pid < n:
        tl.store(x + pid, 0.0)


def test_missing_builder_methods_reports_incomplete_native_surface():
    class Builder:
        def get_insertion_point(self):
            return object()

        def restore_insertion_point(self, _point):
            return None

    missing = _missing_builder_methods(Builder(), _NATIVE_CONTROL_FLOW_BUILDER_METHODS)

    assert "get_insertion_point" not in missing
    assert "restore_insertion_point" not in missing
    assert "create_block" in missing
    assert "create_if_op" in missing


def test_dynamic_while_fails_before_attribute_error_on_incomplete_builder():
    source = "while n:\n    n = n - 1\n"
    node = ast.parse(source).body[0]
    generator = object.__new__(CodeGenerator)
    generator.builder = object()
    generator.jit_fn = SimpleNamespace(src=source)

    with pytest.raises(UnsupportedLanguageConstruct, match="dynamic Triton while.*missing required methods"):
        CodeGenerator.visit_While(generator, node)


def test_dynamic_if_compile_fails_with_explicit_native_control_flow_error():
    source = triton.compiler.ASTSource(
        fn=_dynamic_if_kernel,
        signature={"x": "*fp32", "n": "i32"},
        constexprs={},
    )

    with pytest.raises(UnsupportedLanguageConstruct, match="dynamic Triton if.*create_block"):
        triton.compile(source, target=GPUTarget("cuda", 80, 32))


def test_native_builder_exposes_insertion_point_snapshot():
    builder = ir.builder(ir.context())
    module = builder.create_module()
    fn_type = builder.get_function_ty([], [])
    fn = builder.get_or_insert_function(module, "entry", fn_type, "public", False)
    entry = fn.add_entry_block()

    builder.set_insertion_point_to_start(entry)
    point = builder.get_insertion_point()
    builder.ret([])
    builder.restore_insertion_point(point)

    assert builder.get_insertion_block() is not None
