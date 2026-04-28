import triton
import triton.language as tl


@triton.jit
def _compat_kernel(a, b, n: tl.constexpr):
    return


@triton.jit
def _bool_compat_kernel(a, flag):
    return


def test_attrs_descriptor_is_public_and_normalizes_for_ast_source():
    attrs = triton.compiler.AttrsDescriptor(divisible_by_16=(0, ), equal_to_1=(2, ), divisible_by_8=(1, ))

    src = triton.compiler.ASTSource(
        fn=_compat_kernel,
        signature={0: "*fp32", 1: "i32"},
        constants={2: 32},
        attrs=attrs,
    )

    assert src.signature == {"a": "*fp32", "b": "i32"}
    assert src.constants == {(2, ): 32}
    assert src.attrs == {
        (0, ): [["tt.divisibility", 16]],
        (1, ): [["tt.divisibility", 8]],
    }
    assert triton.compiler.AttrsDescriptor.from_dict(attrs.to_dict()).hash() == attrs.hash()


def test_ast_source_accepts_legacy_string_signature_and_constants_alias():
    src = triton.compiler.ASTSource(
        fn=_compat_kernel,
        signature="*fp32, i32",
        constants={"n": 64},
    )

    assert src.signature == {"a": "*fp32", "b": "i32"}
    assert src.constants == {(2, ): 64}


def test_attrs_descriptor_equal_to_1_specializes_constants():
    src = triton.compiler.ASTSource(
        fn=_compat_kernel,
        signature={0: "*fp32", 1: "i32"},
        attrs=triton.compiler.AttrsDescriptor(equal_to_1=(1, 2)),
    )

    assert src.constants == {(1, ): 1, (2, ): 1}


def test_attrs_descriptor_equal_to_1_preserves_explicit_constants():
    src = triton.compiler.ASTSource(
        fn=_compat_kernel,
        signature={0: "*fp32", 1: "i32"},
        constants={2: 32},
        attrs=triton.compiler.AttrsDescriptor(equal_to_1=(2, )),
    )

    assert src.constants == {(2, ): 32}


def test_attrs_descriptor_equal_to_1_uses_true_for_bool_signature():
    src = triton.compiler.ASTSource(
        fn=_bool_compat_kernel,
        signature={0: "*fp32", 1: "i1"},
        attrs=triton.compiler.AttrsDescriptor(equal_to_1=(1, )),
    )

    assert src.constants == {(1, ): True}
