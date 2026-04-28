from triton.backends.compiler import BaseBackend, GPUTarget, Language
#from triton._C.libtriton import ir, passes, llvm, nvidia
#from triton._C.libtriton import tle
from triton._C.libtriton import ir
from triton import knobs
from triton.runtime.errors import PTXASError

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import re
import tempfile
import signal
import os
import subprocess
from pathlib import Path


def min_dot_size(target: GPUTarget):

    def check_dot_compatibility(lhs_type, rhs_type) -> Tuple[int, int, int]:  # [m, n, k]
        lhs_bitwidth = lhs_type.scalar.primitive_bitwidth
        rhs_bitwidth = rhs_type.scalar.primitive_bitwidth
        assert lhs_bitwidth == rhs_bitwidth, "lhs and rhs bitwidth must be the same"
        # For small M/N the input we can still use tensorcores with padding.
        if lhs_bitwidth == 8:
            return (1, 1, 32)
        else:
            return (1, 1, 16)

    return check_dot_compatibility


def get_ptxas() -> knobs.NvidiaTool:
    return knobs.nvidia.ptxas


@functools.lru_cache()
def get_ptxas_version():
    mock_ver = knobs.nvidia.mock_ptx_version
    if mock_ver is not None:
        return mock_ver  # This is not really a version of ptxas, but it is good enough for testing
    version = subprocess.check_output([get_ptxas().path, "--version"]).decode("utf-8")
    return version


@functools.lru_cache()
def ptx_get_version(cuda_version) -> int:
    '''
    Get the highest PTX version supported by the current CUDA driver.
    '''
    assert isinstance(cuda_version, str)
    major, minor = map(int, cuda_version.split('.'))
    if major == 12:
        if minor < 6:
            return 80 + minor
        else:
            return 80 + minor - 1
    if major == 11:
        return 70 + minor
    if major == 10:
        return 63 + minor

    if major >= 13:
        base_ptx = 90
        return base_ptx + (major - 13) * 10 + minor

    raise RuntimeError("Triton only support CUDA 10.0 or higher, but got CUDA version: " + cuda_version)


def get_ptx_version_from_options(options, arch: int):
    ptx_version = options.ptx_version
    if ptx_version is None:
        cuda_version = get_ptxas().version
        ptx_version = ptx_get_version(cuda_version)
    return ptx_version


@functools.lru_cache()
def get_features(options, arch: int):
    ptx_version = get_ptx_version_from_options(options, arch)

    # PTX 8.6 is the max version supported by llvm c1188642.
    #
    # To check if a newer PTX version is supported, increase this value
    # and run a test.  If it's not supported, LLVM will print a warning
    # like "+ptx8.4 is not a recognized feature for this target".
    llvm_ptx_version = min(86, ptx_version)
    features = f'+ptx{llvm_ptx_version}'
    return features


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def sm_arch_from_capability(capability: int):
    # TODO: Handle non-"a" sms
    suffix = "a" if capability >= 90 else ""
    return f"sm_{capability}{suffix}"


def _module_entry_name(src):
    get_name = getattr(src, "get_entry_func_name", None)
    if get_name is None:
        return None
    name = get_name()
    return name[1:] if isinstance(name, str) and name.startswith("@") else name


def _initialize_cuda_kernel_metadata(metadata, name, opt):
    metadata["name"] = name
    metadata["shared"] = 0
    metadata["num_warps"] = opt.num_warps
    metadata["num_ctas"] = opt.num_ctas
    metadata["cluster_dims"] = tuple(opt.cluster_dims or (1, 1, 1))
    metadata["tmem_size"] = 0
    metadata["global_scratch_size"] = 0
    metadata["global_scratch_align"] = 1
    metadata["profile_scratch_size"] = 0
    metadata["profile_scratch_align"] = 1


def _extract_ptx_entry_name(src):
    names = re.findall(r"^\s*(?:\.(?:visible|extern)\s+)?\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(", src,
                       flags=re.MULTILINE)
    if len(names) != 1:
        raise ValueError(f"PTX text must contain exactly one launchable .entry symbol, found {len(names)}.")
    return names[0]


@dataclass(frozen=True)
class NativeCudaIRStage:
    module: Any
    suppress_stage_file: bool = True

    def __str__(self):
        return str(self.module)


@dataclass(frozen=True)
class NativeCudaCompilation:
    cubin: bytes
    metadata: Dict[str, Any]
    asm: Dict[str, Any]
    compiler_log: str = ""
    suppress_stage_file: bool = True

    def __str__(self):
        for key in ("ntt_cu", "cuda_source", "compiler_log"):
            value = self.asm.get(key)
            if value:
                return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        if self.compiler_log:
            return self.compiler_log
        return "<FlagLang native CUDA compilation result: cubin bytes are stored in the cubin stage>"

    def cache_artifacts(self):
        artifacts = dict(self.asm)
        if self.compiler_log:
            artifacts.setdefault("compiler_log", self.compiler_log)
        return artifacts


def _require_native_ir_module(src):
    src = _unwrap_native_cuda_stage(src)
    mlir_source_type = getattr(ir, "mlir_source_module", None)
    if mlir_source_type is not None and isinstance(src, mlir_source_type):
        raise NotImplementedError(
            "FlagLang native CUDA backend can parse .ttir/.ttgir MLIR files for metadata, "
            "but it cannot lower external MLIR source text to a native nncase ir.module. "
            "Compile Triton AST kernels through the native frontend so lowering starts from ir.module.")

    module_type = getattr(ir, "module", None)
    if module_type is None or not isinstance(src, module_type):
        entry = _module_entry_name(src)
        raise TypeError("Unsupported native module for CUDA cubin emission: expected actual post-TTIR native "
                        f"ir.module, got entry {entry!r} of type {type(src).__name__}.")


def _unwrap_native_cuda_stage(src):
    return src.module if isinstance(src, NativeCudaIRStage) else src


def _native_stage_text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _validate_native_cubin(cubin: bytes, entry_name: str):
    if (len(cubin) < 64 or cubin[0:4] != b"\x7fELF" or cubin[4] != 2 or cubin[5] != 1):
        raise ValueError("Native CUDA compile helper returned bytes that are not a CUDA ELF cubin artifact.")

    cuobjdump = knobs.nvidia.cuobjdump.path

    cubin_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".cubin") as fbin:
            fbin.write(cubin)
            cubin_path = fbin.name
        result = subprocess.run(
            [cuobjdump, "--dump-elf", cubin_path],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        if cubin_path is not None and os.path.exists(cubin_path):
            os.remove(cubin_path)

    if result.returncode != 0:
        diagnostics = (result.stderr or result.stdout or "").strip()
        raise ValueError(f"cuobjdump rejected native CUDA cubin artifact: {diagnostics}")

    output = result.stdout + result.stderr
    if entry_name not in output:
        raise ValueError(f"Native CUDA cubin artifact does not contain entry symbol '{entry_name}'.")


def _split_top_level_commas(text: str):
    parts = []
    start = 0
    paren_depth = 0
    bracket_depth = 0
    angle_depth = 0
    for idx, char in enumerate(text):
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "<":
            angle_depth += 1
        elif char == ">":
            angle_depth -= 1
        elif char == "," and paren_depth == 0 and bracket_depth == 0 and angle_depth == 0:
            parts.append(text[start:idx].strip())
            start = idx + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _canonical_cpp_type(type_text: str):
    normalized = re.sub(r"\s+", " ", type_text.strip())
    normalized = normalized.replace(" *", "*").replace("* ", "*")
    return normalized


def _parse_cuda_entry_parameters(source: str, entry_name: str):
    pattern = re.compile(r"\bvoid\s+" + re.escape(entry_name) + r"\s*\((.*?)\)\s*\{", re.DOTALL)
    match = pattern.search(source)
    if match is None:
        raise ValueError(f"Native CUDA source does not define entry '{entry_name}'.")

    parameters = match.group(1).strip()
    if not parameters or parameters == "void":
        return []

    result = []
    for parameter in _split_top_level_commas(parameters):
        parameter = parameter.split("=", 1)[0].strip()
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*$", parameter)
        if name_match is None:
            raise ValueError(f"Unable to parse native CUDA entry parameter: {parameter!r}.")
        result.append({
            "name": name_match.group(1),
            "type": _canonical_cpp_type(parameter[:name_match.start()].strip()),
        })
    return result


def _require_string_list(value, field_name: str):
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"Native CUDA ABI {field_name} must be a list of non-empty strings.")
    return value


def _validate_native_cuda_abi(metadata, native_metadata, compilation: NativeCudaCompilation, name: str):
    abi = native_metadata.get("flaglang_abi")
    if not isinstance(abi, dict):
        raise KeyError("Native CUDA compile helper metadata must include flaglang_abi contract.")
    if abi.get("entry") != name:
        raise ValueError("Native CUDA ABI entry must match kernel metadata name.")

    argument_order = _require_string_list(abi.get("argument_order"), "argument_order")
    if len(argument_order) != len(set(argument_order)):
        raise ValueError("Native CUDA ABI argument_order must contain unique entries.")

    argument_count = abi.get("argument_count")
    if not isinstance(argument_count, int) or argument_count != len(argument_order):
        raise ValueError("Native CUDA ABI argument_count must match argument_order length.")

    imported_argument_order = _require_string_list(abi.get("imported_argument_order"), "imported_argument_order")
    if imported_argument_order != argument_order:
        raise ValueError("Native CUDA ABI argument_order must match imported_argument_order.")

    argument_types = _require_string_list(abi.get("argument_types"), "argument_types")
    if len(argument_types) != argument_count:
        raise ValueError("Native CUDA ABI argument_types must match argument_count.")

    imported_argument_types = _require_string_list(abi.get("imported_argument_types"), "imported_argument_types")
    if imported_argument_types != argument_types:
        raise ValueError("Native CUDA ABI argument_types must match imported_argument_types.")

    raw_argument_order = _require_string_list(abi.get("raw_argument_order"), "raw_argument_order")
    if len(raw_argument_order) != argument_count or len(raw_argument_order) != len(set(raw_argument_order)):
        raise ValueError("Native CUDA ABI raw_argument_order must be unique and match argument_count.")

    raw_argument_types = _require_string_list(abi.get("raw_argument_types"), "raw_argument_types")
    if len(raw_argument_types) != argument_count:
        raise ValueError("Native CUDA ABI raw_argument_types must match argument_count.")

    runtime_argument_order = _require_string_list(abi.get("runtime_argument_order"), "runtime_argument_order")
    runtime_argument_types = _require_string_list(abi.get("runtime_argument_types"), "runtime_argument_types")
    runtime_argument_count = abi.get("runtime_argument_count")
    if runtime_argument_count != len(runtime_argument_order) or runtime_argument_count != len(runtime_argument_types):
        raise ValueError("Native CUDA ABI runtime argument metadata must have matching count, order, and types.")
    if runtime_argument_count != argument_count:
        raise ValueError("Native CUDA ABI runtime_argument_count must match argument_count.")

    expected_runtime_count = metadata.get("runtime_argument_count")
    if expected_runtime_count is not None and runtime_argument_count != expected_runtime_count:
        raise ValueError("Native CUDA ABI runtime_argument_count does not match Python launcher signature.")
    expected_runtime_order = metadata.get("runtime_argument_order")
    if expected_runtime_order is not None and runtime_argument_order != list(expected_runtime_order):
        raise ValueError("Native CUDA ABI runtime_argument_order does not match Python launcher signature.")
    expected_runtime_types = metadata.get("runtime_argument_types")
    if expected_runtime_types is not None and runtime_argument_types != [str(item) for item in expected_runtime_types]:
        raise ValueError("Native CUDA ABI runtime_argument_types does not match Python launcher signature.")

    source = compilation.asm.get("ntt_cu") or compilation.asm.get("cuda_source")
    if source is None:
        raise KeyError("Native CUDA compile helper asm must include ntt_cu source for ABI validation.")
    source_parameters = _parse_cuda_entry_parameters(_native_stage_text(source), name)
    source_order = [parameter["name"] for parameter in source_parameters]
    source_types = [parameter["type"] for parameter in source_parameters]
    if source_order != raw_argument_order:
        raise ValueError("Native CUDA ABI raw_argument_order does not match generated CUDA entry signature.")
    if source_types != [_canonical_cpp_type(item) for item in raw_argument_types]:
        raise ValueError("Native CUDA ABI raw_argument_types does not match generated CUDA entry signature.")


def _normalize_native_cuda_compilation(result) -> NativeCudaCompilation:
    if isinstance(result, NativeCudaCompilation):
        return result
    if not isinstance(result, dict):
        raise TypeError(f"Native CUDA compile helper returned {type(result).__name__}, expected dict.")

    cubin = result.get("cubin")
    if isinstance(cubin, memoryview):
        cubin = cubin.tobytes()
    elif isinstance(cubin, bytearray):
        cubin = bytes(cubin)
    if not isinstance(cubin, bytes) or not cubin:
        raise TypeError("Native CUDA compile helper must return non-empty cubin bytes.")

    metadata = result.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("Native CUDA compile helper metadata must be a dict.")
    entry_name = metadata.get("name")
    if not isinstance(entry_name, str) or not entry_name:
        raise KeyError("Native CUDA compile helper metadata must include kernel entry name.")
    _validate_native_cubin(cubin, entry_name)

    asm = result.get("asm", result.get("stages", {}))
    if not isinstance(asm, dict):
        raise TypeError("Native CUDA compile helper asm/stages must be a dict.")

    compiler_log = result.get("compiler_log", "")
    if compiler_log is None:
        compiler_log = ""
    if not isinstance(compiler_log, str):
        raise TypeError("Native CUDA compile helper compiler_log must be a string.")

    return NativeCudaCompilation(cubin=cubin, metadata=metadata, asm=asm, compiler_log=compiler_log)


def _native_compile_options(src, metadata, opt, capability):
    src = _unwrap_native_cuda_stage(src)
    cluster_dims = tuple(opt.cluster_dims or (1, 1, 1))
    return {
        "entry_name":
        _module_entry_name(src),
        "arch":
        sm_arch_from_capability(capability),
        "capability":
        capability,
        "num_warps":
        opt.num_warps,
        "num_ctas":
        opt.num_ctas,
        "cluster_dims":
        cluster_dims,
        "warp_size":
        opt.warp_size,
        "threads_per_warp":
        opt.warp_size,
        "threads_per_cta":
        opt.num_warps * opt.warp_size,
        "binary_ext":
        "cubin",
        "required_metadata": [
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
        ],
        "stage_names": ["triton_tir", "nncase_ir", "after_compile", "tir", "ntt_cu", "compiler_log", "cubin"],
        "dump_dir":
        os.environ.get("TRITON_DUMP_DIR"),
        "cuobjdump":
        knobs.nvidia.cuobjdump.path,
        "enable_auto_dist":
        False,
        "runtime_argument_count":
        metadata.get("runtime_argument_count", 0),
        "runtime_argument_order":
        list(metadata.get("runtime_argument_order", [])),
        "runtime_argument_types": [str(item) for item in metadata.get("runtime_argument_types", [])],
    }


def _compile_native_module_to_cubin(src, metadata, opt, capability) -> NativeCudaCompilation:
    src = _unwrap_native_cuda_stage(src)
    _require_native_ir_module(src)
    compile_to_cubin = getattr(ir, "compile_to_cubin", None)
    if not callable(compile_to_cubin):
        raise RuntimeError(
            "FlagLang native CUDA compile helper is unavailable; refusing to emit handwritten PTX shortcut.")

    result = compile_to_cubin(src, _native_compile_options(src, metadata, opt, capability))
    return _normalize_native_cuda_compilation(result)


def _apply_native_cuda_metadata(metadata, compilation: NativeCudaCompilation, opt):
    native_metadata = dict(compilation.metadata)
    name = native_metadata.get("name") or metadata.get("name")
    if not name:
        raise KeyError("Native CUDA compile helper metadata must include kernel entry name.")
    _validate_native_cuda_abi(metadata, native_metadata, compilation, name)

    _initialize_cuda_kernel_metadata(metadata, name, opt)
    metadata.update(native_metadata)
    metadata["flaglang_pipeline"] = {
        "kind": "native_cuda",
        "artifact": "cubin",
        "compiler_log": bool(compilation.compiler_log),
        "stages": sorted(compilation.asm.keys()),
    }


@dataclass(frozen=True)
class CUDAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 3
    warp_size: int = 32
    # maxnreg corresponds to the ptx parameter .maxnreg, which controls the
    # maximum number of 32-bit registers used by one thread.
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    ptx_version: int = None
    ptx_options: str = None
    ir_override: Optional[str] = None  # filename of a user-defined IR (*.{ttir|ttgir|llir|ptx})
    enable_fp_fusion: bool = True
    launch_cooperative_grid: bool = False
    launch_pdl: bool = False
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4b15")
    deprecated_fp8_dot_operand_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: bool = None
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = 'cuda'
    sanitize_overflow: bool = True
    arch: str = None
    instrumentation_mode: str = ""

    def __post_init__(self):
        default_libdir = Path(__file__).parent / 'lib'
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get('libdevice', None):
            extern_libs['libdevice'] = knobs.nvidia.libdevice_path or str(default_libdir / 'libdevice.10.bc')

        object.__setattr__(self, 'extern_libs', tuple(extern_libs.items()))
        assert self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0, \
               "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


class CUDABackend(BaseBackend):
    instrumentation = None

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == 'cuda'

    def _parse_arch(self, arch):
        pattern = r"^sm(\d+)$"
        match = re.fullmatch(pattern, arch)
        if not match:
            raise ValueError(f"TRITON_OVERRIDE_ARCH must have the form {pattern}")
        return int(match.group(1))

    def get_target_name(self, options) -> str:
        capability = self._parse_arch(options.arch)
        return f"cuda:{capability}"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.binary_ext = "cubin"

    def make_context(self, options: object):
        target = ir.target("cuda")
        options = ir.compile_options()
        return ir.compile_session(target, options)

    def parse_options(self, opts) -> Any:
        args = {'arch': knobs.runtime.override_arch or f"sm{self.target.arch}"}
        args.update({k: opts[k] for k in CUDAOptions.__dataclass_fields__.keys() if k in opts if opts[k] is not None})
        capability = int(self._parse_arch(args["arch"]))

        if args.get("num_ctas", 1) > 1 and capability < 90:
            raise ValueError((f"num_ctas > 1 requires NVIDIA SM90+ (Hopper). "
                              f"Current target is sm_{capability}. This configuration will fail. "
                              f"Please set num_ctas=1 or target an SM90+ GPU."))

        if "supported_fp8_dtypes" not in args:
            supported_fp8_dtypes = set(CUDAOptions.supported_fp8_dtypes)
            if capability >= 89:
                supported_fp8_dtypes.add("fp8e4nv")
            args["supported_fp8_dtypes"] = tuple(sorted(supported_fp8_dtypes))

        if "deprecated_fp8_dot_operand_dtypes" not in args:
            if capability >= 90:
                args["deprecated_fp8_dot_operand_dtypes"] = ("fp8e4b15", )

        if "enable_fp_fusion" not in args:
            args["enable_fp_fusion"] = knobs.language.default_fp_fusion

        args["max_num_imprecise_acc_default"] = 2**30 if capability == 90 else 0

        return CUDAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self, options):
        import triton.language.extra.cuda as cuda
        capability = int(self._parse_arch(options.arch))
        codegen_fns = {
            "convert_custom_types":
            cuda.convert_custom_float8_sm80 if capability >= 80 else cuda.convert_custom_float8_sm70, "min_dot_size":
            min_dot_size(self.target)
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.cuda import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, context):
        if not isinstance(context, ir.compile_session):
            raise TypeError(f"CUDA backend requires a native compile_session context, got {type(context).__name__}")
        # Native FlagLang contexts are backed by the managed compiler host; its
        # application parts are loaded when the host is initialized.
        # if CUDABackend.instrumentation:
        #     CUDABackend.instrumentation.load_dialects(context)

    @staticmethod
    def make_ttir(mod, metadata, opt, capability):
        pm = ir.pass_manager(mod.context, "ttir")
        pm.add_optimize_ttir(capability)
        # passes.common.add_inliner(pm)
        # passes.ttir.add_rewrite_tensor_pointer(pm)
        # if capability // 10 < 9:
        #     passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
        # passes.common.add_canonicalizer(pm)
        # passes.ttir.add_combine(pm)
        # passes.ttir.add_reorder_broadcast(pm)
        # passes.common.add_cse(pm)
        # passes.common.add_symbol_dce(pm)
        # passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        if isinstance(mod, getattr(ir, "module", ())):
            return NativeCudaIRStage(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability):
        if isinstance(mod, NativeCudaIRStage):
            return mod
        # Set maxnreg on all kernels, if it was provided.
        # if opt.maxnreg is not None:
        #     mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

        # cluster_info = nvidia.ClusterInfo()
        # if opt.cluster_dims is not None:
        #     cluster_info.clusterDimX = opt.cluster_dims[0]
        #     cluster_info.clusterDimY = opt.cluster_dims[1]
        #     cluster_info.clusterDimZ = opt.cluster_dims[2]
        # pm = ir.pass_manager(mod.context)
        # dump_enabled = pm.enable_debug()
        # passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
        # # flagtree tle raw
        # tle.raw_passes.add_tle_convert_arg_to_memdesc(pm)
        # # optimize TTGIR
        # passes.ttgpuir.add_coalesce(pm)
        # passes.ttgpuir.add_process_shared_memory_hint(pm)  # flagtree hints
        # if capability // 10 >= 8:
        #     passes.ttgpuir.add_f32_dot_tc(pm)
        # # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
        # nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
        # passes.ttgpuir.add_remove_layout_conversions(pm)
        # passes.ttgpuir.add_optimize_thread_locality(pm)
        # tle.passes.add_early_assign_memory_space(pm)
        # passes.ttgpuir.add_accelerate_matmul(pm)
        # passes.ttgpuir.add_remove_layout_conversions(pm)
        # passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        # nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
        # passes.ttir.add_loop_aware_cse(pm)
        # if capability // 10 in [8, 9]:
        #     passes.ttgpuir.add_fuse_nested_loops(pm)
        #     passes.common.add_canonicalizer(pm)
        #     passes.ttir.add_triton_licm(pm)
        #     passes.common.add_canonicalizer(pm)
        #     passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        #     nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
        #     passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        #     passes.ttgpuir.add_schedule_loops(pm)
        #     passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        # elif capability // 10 >= 10:
        #     passes.ttgpuir.add_fuse_nested_loops(pm)
        #     passes.common.add_canonicalizer(pm)
        #     passes.ttir.add_triton_licm(pm)
        #     passes.ttgpuir.add_optimize_accumulator_init(pm)
        #     passes.ttgpuir.add_hoist_tmem_alloc(pm, False)
        #     nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
        #     passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        #     passes.ttgpuir.add_schedule_loops(pm)
        #     passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
        #     passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        #     passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        #     # hoist again and allow hoisting out of if statements
        #     passes.ttgpuir.add_hoist_tmem_alloc(pm, True)
        #     nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
        # else:
        #     passes.ttir.add_triton_licm(pm)
        # passes.common.add_canonicalizer(pm)
        # passes.ttir.add_loop_aware_cse(pm)
        # passes.ttgpuir.add_prefetch(pm)
        # passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
        # passes.ttgpuir.add_coalesce_async_copy(pm)
        # nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
        # passes.ttgpuir.add_remove_layout_conversions(pm)
        # nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
        # passes.ttgpuir.add_reduce_data_duplication(pm)
        # passes.ttgpuir.add_reorder_instructions(pm)
        # # flagtree tle: Lowering load with tt.load.async attribute
        # tle.passes.add_lower_async_load(pm)
        # passes.ttir.add_loop_aware_cse(pm)
        # passes.common.add_symbol_dce(pm)
        # if capability // 10 >= 9:
        #     # flagtree tle: Apply TLE TMA copy lowering before standard NVIDIA TMA lowering
        #     tle.passes.add_lower_tma_copy(pm)
        #     nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        # nvidia.passes.ttnvgpuir.add_fence_insertion(pm, capability)
        # nvidia.passes.ttnvgpuir.add_lower_mma(pm)
        # passes.common.add_sccp(pm)
        # passes.common.add_cse(pm)
        # passes.common.add_canonicalizer(pm)

        # pm.run(mod)
        # metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
        # tensordesc_meta = mod.get_tensordesc_metadata()
        # metadata["tensordesc_meta"] = tensordesc_meta
        return mod

    def gluon_to_ttgir(self, src, metadata, options, capability):
        mod = src
        # pm.enable_debug()

        # passes.gluon.add_inliner(pm)
        # passes.gluon.add_resolve_auto_encodings(pm)
        # passes.common.add_sccp(pm)
        # passes.ttir.add_loop_aware_cse(pm)
        # passes.gluon.add_canonicalizer(pm)
        # passes.ttgpuir.add_combine_tensor_select_and_if(pm)

        # pm.run(mod)
        # metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
        return mod

    def make_llir(self, src, metadata, options, capability):
        if isinstance(src, NativeCudaIRStage):
            return src

        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        return mod
        # pm.enable_debug()

        # passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        # passes.ttgpuir.add_allocate_warp_groups(pm)
        # passes.convert.add_scf_to_cf(pm)
        # nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
        # nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
        # if knobs.compilation.enable_experimental_consan:
        #     # Call ConcurrencySanitizerPass here, before allocating global scratch memory but after allocating tensor and shared
        #     passes.ttgpuir.add_concurrency_sanitizer(pm)
        # passes.ttgpuir.add_allocate_global_scratch_memory(pm)
        # nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
        # # instrumentation point here so we can override IRs above (e.g., ttir and ttgir)
        # if CUDABackend.instrumentation:
        #     CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
        # nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
        # passes.common.add_canonicalizer(pm)
        # passes.common.add_cse(pm)
        # nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
        # nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
        # passes.common.add_canonicalizer(pm)
        # passes.common.add_cse(pm)
        # passes.common.add_symbol_dce(pm)
        # passes.convert.add_nvvm_to_llvm(pm)
        # if not knobs.compilation.disable_line_info:
        #     passes.llvmir.add_di_scope(pm)
        # if CUDABackend.instrumentation:
        #     CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)
        # # flagtree tle raw
        # tle.raw_passes.add_tle_dsl_region_inline(pm)

        # pm.run(mod)
        # # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        # llvm.init_targets()
        # context = llvm.context()
        # if knobs.compilation.enable_asan:
        #     raise RuntimeError(
        #         "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
        # llvm_mod = llvm.to_module(mod, context)
        # proc = sm_arch_from_capability(capability)
        # features = get_features(options, self.target.arch)
        # triple = 'nvptx64-nvidia-cuda'
        # nvidia.set_short_ptr()
        # llvm.attach_datalayout(llvm_mod, triple, proc, features)
        # nvidia.set_nvvm_reflect_ftz(llvm_mod)

        # if options.extern_libs and nvidia.has_extern_deps(llvm_mod):
        #     paths = [path for (name, path) in options.extern_libs]
        #     llvm.link_extern_libs(llvm_mod, paths)

        # llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # # Get some metadata
        # # warp-specialization mutates num_warps
        # total_num_warps = src.get_int_attr("ttg.total-num-warps")
        # if total_num_warps is not None:
        #     metadata["num_warps"] = total_num_warps
        # metadata["shared"] = src.get_int_attr("ttg.shared")
        # metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
        # metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
        # metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
        # metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
        # metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
        # ret = str(llvm_mod)
        # del llvm_mod
        # del context
        # return ret

    def make_ptx(self, src, metadata, opt, capability):
        src = _unwrap_native_cuda_stage(src)
        if not isinstance(src, str):
            return _compile_native_module_to_cubin(src, metadata, opt, capability)

        return src

        # triple = 'nvptx64-nvidia-cuda'
        # proc = sm_arch_from_capability(capability)
        # features = get_features(opt, self.target.arch)
        # ret = llvm.translate_to_asm(src, triple, proc, features, [], opt.enable_fp_fusion, False)
        # # Find kernel names (there should only be one)
        # names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
        # assert len(names) == 1
        # metadata["name"] = names[0]
        # # post-process
        # ptx_version = f'{ptx_version//10}.{ptx_version%10}'
        # ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
        # ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
        # # Remove the debug flag that prevents ptxas from optimizing the code
        # ret = re.sub(r",\s*debug|debug,\s*", "", ret)
        # if knobs.nvidia.dump_nvptx:
        #     print("// -----// NVPTX Dump //----- //")
        #     print(ret)
        # return ret

    def make_cubin(self, src, metadata, opt, capability):
        if isinstance(src, NativeCudaCompilation):
            _apply_native_cuda_metadata(metadata, src, opt)
            _validate_native_cubin(src.cubin, metadata["name"])
            return src.cubin

        if not isinstance(src, str):
            raise TypeError(f"make_cubin expected PTX text or native CUDA compilation result, got {type(src).__name__}")

        _initialize_cuda_kernel_metadata(metadata, _extract_ptx_entry_name(src), opt)

        ptxas = get_ptxas().path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            debug_info = []
            if knobs.compilation.disable_line_info:
                # This option is ignored if used without -lineinfo
                debug_info += ["-lineinfo", "-suppress-debug-info"]
            elif knobs.nvidia.disable_ptxas_opt:
                # Synthesize complete debug info
                debug_info += ["-g"]
            else:
                # Only emit line info
                debug_info += ["-lineinfo"]

            fmad = [] if opt.enable_fp_fusion else ["--fmad=false"]
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            ptxas_cmd = [
                ptxas, *debug_info, *fmad, '-v', *disable_opt, *ptx_extra_options, f'--gpu-name={arch}', fsrc.name,
                '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if knobs.nvidia.dump_ptxas_log:
                    with open(flog.name) as log_file:
                        print(log_file.read())

                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                error = (f"{error}\n"
                         f"`ptxas` stderr:\n{log}\n"
                         f'Repro command: {" ".join(ptxas_cmd)}\n')

                print(f"""

================================================================
{error}

{src}
================================================================
please share the reproducer above with Triton project.
""")
                raise PTXASError(error)

            with open(fbin, 'rb') as f:
                cubin = f.read()
            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    def add_stages(self, stages, options, language):
        capability = self._parse_arch(options.arch)
        if language == Language.TRITON:
            stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options, capability)
            stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, capability)
        elif language == Language.GLUON:
            stages["ttgir"] = lambda src, metadata: self.gluon_to_ttgir(src, metadata, options, capability)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, capability)
        stages["ptx"] = lambda src, metadata: self.make_ptx(src, metadata, options, capability)
        stages["cubin"] = lambda src, metadata: self.make_cubin(src, metadata, options, capability)

    @functools.lru_cache()
    def hash(self):
        version = get_ptxas_version()
        return f'{version}-{self.target.arch}'
