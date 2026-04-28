import numpy as np

from triton._C.libtriton import RuntimeTensor, ir


def test_libtriton_exports_interpreter_submodule():
    import triton.runtime.interpreter as runtime_interpreter
    from triton._C.libtriton import interpreter

    assert runtime_interpreter._interpreter is interpreter
    assert hasattr(interpreter, "load")
    assert hasattr(interpreter, "atomic_rmw")


def test_native_ir_builder_float_constant_surface():
    builder = ir.builder(ir.context())

    fp16 = builder.get_fp16(1.5)
    fp32 = builder.get_fp32(2.5)
    fp64 = builder.get_fp64(3.5)

    assert "1.5" in fp16.to_text()
    assert "2.5" in fp32.to_text()
    assert "3.5" in fp64.to_text()


def test_runtime_tensor_to_numpy_returns_owned_array():
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    tensor = RuntimeTensor.from_numpy(source)

    result = tensor.to_numpy()

    assert result.flags.owndata
    np.testing.assert_array_equal(result, source)


def test_native_ir_module_function_lookup_surface():
    builder = ir.builder(ir.context())
    module = builder.create_module()
    fn_type = builder.get_function_ty([], [])
    fn = builder.get_or_insert_function(module, "helper", fn_type, "private", False)

    assert not module.has_function("helper")

    module.push_back(fn)

    assert module.has_function("helper")
    assert module.get_function("helper").get_num_args() == 0


def _build_vector_add_module():
    session = ir.compile_session(ir.target("cuda"), ir.compile_options())
    builder = ir.builder(session)

    module = builder.create_module()
    f32 = builder.get_float_ty()
    i32 = builder.get_int32_ty()
    ptr_f32 = builder.get_ptr_ty(f32)
    block_i32 = builder.get_block_ty(i32, [8])
    block_f32 = builder.get_block_ty(f32, [8])
    fn_ty = builder.get_function_ty([ptr_f32, ptr_f32, ptr_f32, i32], [])
    fn = builder.get_or_insert_function(module, "add_kernel", fn_ty, "public", False)

    entry = fn.add_entry_block()
    builder.set_insertion_point_to_end(entry)
    x_arg, y_arg, output_arg, n_elements = [fn.args(i) for i in range(4)]
    fn.set_arg_attr(0, "tt.divisibility", 16)
    builder.set_loc(builder.create_loc("vector_add.py", 1, 1))

    pid = builder.create_get_program_id(0)
    block_size = builder.get_int32(8)
    block_start = builder.create_mul(pid, block_size)
    lanes = builder.create_make_range(block_i32, 0, 8)
    offsets = builder.create_add(builder.create_splat(block_i32, block_start), lanes)
    mask = builder.create_icmpSLT(offsets, builder.create_splat(block_i32, n_elements))

    x_ptrs = builder.create_addptr(x_arg, offsets)
    y_ptrs = builder.create_addptr(y_arg, offsets)
    output_ptrs = builder.create_addptr(output_arg, offsets)
    default_value = builder.create_splat(block_f32, builder.get_int32(0))
    x = builder.create_masked_load(
        x_ptrs,
        mask,
        default_value,
        ir.CACHE_MODIFIER.NONE,
        ir.EVICTION_POLICY.NORMAL,
        False,
        None,
    )
    y = builder.create_masked_load(
        y_ptrs,
        mask,
        default_value,
        ir.CACHE_MODIFIER.NONE,
        ir.EVICTION_POLICY.NORMAL,
        False,
        None,
    )
    output = builder.create_fadd(x, y)
    builder.create_masked_store(
        output_ptrs,
        output,
        mask,
        ir.CACHE_MODIFIER.NONE,
        ir.EVICTION_POLICY.NORMAL,
    )
    builder.ret([])

    fn.finalize()
    module.push_back(fn)
    return session, module


def test_native_ir_builder_vector_add_surface():
    _, module = _build_vector_add_module()

    assert module.verify_with_diagnostics()


def test_native_ir_builder_unmasked_store_surface():
    session = ir.compile_session(ir.target("cuda"), ir.compile_options())
    builder = ir.builder(session)

    module = builder.create_module()
    f32 = builder.get_float_ty()
    i32 = builder.get_int32_ty()
    ptr_f32 = builder.get_ptr_ty(f32)
    block_i32 = builder.get_block_ty(i32, [8])
    block_f32 = builder.get_block_ty(f32, [8])
    fn_ty = builder.get_function_ty([ptr_f32], [])
    fn = builder.get_or_insert_function(module, "store_kernel", fn_ty, "public", False)

    entry = fn.add_entry_block()
    builder.set_insertion_point_to_end(entry)
    output_arg = fn.args(0)
    offsets = builder.create_make_range(block_i32, 0, 8)
    output_ptrs = builder.create_addptr(output_arg, offsets)
    value = builder.create_splat(block_f32, builder.get_fp32(1.0))
    builder.create_store(
        output_ptrs,
        value,
        ir.CACHE_MODIFIER.NONE,
        ir.EVICTION_POLICY.NORMAL,
    )
    builder.ret([])

    fn.finalize()
    module.push_back(fn)

    assert module.verify_with_diagnostics()
