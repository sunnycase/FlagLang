from __future__ import annotations
import hashlib
import json
from .._C.libtriton import get_cache_invalidating_env_vars, ir
from ..backends import backends
from ..backends.compiler import Language
from ..backends.compiler import BaseBackend, GPUTarget
from .. import __version__, knobs
from ..runtime.autotuner import OutOfResources
from ..runtime.cache import get_cache_manager, get_dump_manager, get_override_manager, get_cache_key
from ..runtime.driver import driver
from ..tools.disasm import get_sass
from pathlib import Path
import re
import functools
import os
import time
import copy
import inspect
from dataclasses import dataclass

ptx_prototype_pattern = r"^\s*(?:\.(?:visible|extern)\s+)?\.entry\s+([A-Za-z_][A-Za-z0-9_$]*)\s*\(([^)]*)\)"
prototype_pattern = {
    "ptx": ptx_prototype_pattern,
}

ptx_param_storage_types = {
    "b8", "b16", "b32", "b64",
    "s8", "s16", "s32", "s64",
    "u8", "u16", "u32", "u64",
    "f16", "f32", "f64",
    "pred",
}

ptx_param_modifiers_with_value = {
    "align",
}

ptx_param_modifiers = {
    "ptr", "global", "const", "local", "shared",
}


def convert_type_repr(x):
    # Currently we only capture the pointer type and assume the pointer is on global memory.
    # TODO: Capture and support shared memory space
    match = re.search(r'!tt\.ptr<([^,]+)', x)
    tma = re.search(r'tt.nv_tma_desc = 1', x)
    if tma is not None:
        return 'nvTmaDesc'
    x = re.sub(r' {[^}]+}', '', x)
    if match is not None:
        return '*' + convert_type_repr(match.group(1))
    return x


def _extract_ptx_entry_signature(src):
    matches = re.findall(prototype_pattern["ptx"], src, flags=re.MULTILINE)
    if len(matches) != 1:
        raise ValueError(f"PTX text must contain exactly one launchable .entry symbol, found {len(matches)}.")
    return matches[0]


def _parse_ptx_param_type(param_decl):
    tokens = param_decl.strip().split()
    if not tokens or tokens[0] != ".param":
        raise ValueError(f"Malformed PTX parameter declaration: {param_decl!r}")

    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if not token.startswith("."):
            raise ValueError(f"PTX parameter declaration is missing a storage type: {param_decl!r}")

        name = token[1:]
        if name in ptx_param_modifiers_with_value:
            idx += 2
            continue
        if name in ptx_param_modifiers:
            idx += 1
            continue
        if name not in ptx_param_storage_types:
            raise ValueError(f"Unsupported PTX parameter modifier or storage type '.{name}' in {param_decl!r}")
        if idx + 1 >= len(tokens):
            raise ValueError(f"PTX parameter declaration is missing a name: {param_decl!r}")

        match = re.match(r"^[A-Za-z_.$][A-Za-z0-9_.$]*(?:\[(\d+)\])?$", tokens[idx + 1])
        if match is None:
            raise ValueError(f"Malformed PTX parameter name or shape in {param_decl!r}")
        shape = match.group(1)
        return f"{name}[{shape}]" if shape is not None else convert_type_repr(name)

    raise ValueError(f"PTX parameter declaration is missing a storage type: {param_decl!r}")


def _parse_ptx_param_types(signature):
    params = [param.strip() for param in signature.split(",") if param.strip()]
    return [_parse_ptx_param_type(param) for param in params]


@dataclass
class AttrsDescriptor:
    divisible_by_16: set | None = None
    equal_to_1: set | None = None
    divisible_by_8: set | None = None

    def __post_init__(self):
        self.divisible_by_16 = set() if self.divisible_by_16 is None else set(self.divisible_by_16)
        self.equal_to_1 = set() if self.equal_to_1 is None else set(self.equal_to_1)
        self.divisible_by_8 = set() if self.divisible_by_8 is None else set(self.divisible_by_8)

    def to_dict(self):
        return {
            "divisible_by_16": list(self.divisible_by_16),
            "equal_to_1": list(self.equal_to_1),
            "divisible_by_8": list(self.divisible_by_8),
        }

    @staticmethod
    def from_dict(data):
        return AttrsDescriptor(
            divisible_by_16=set(data.get("divisible_by_16", [])),
            equal_to_1=set(data.get("equal_to_1", [])),
            divisible_by_8=set(data.get("divisible_by_8", [])),
        )

    def hash(self):
        key = str([sorted(self.divisible_by_16), sorted(self.equal_to_1), sorted(self.divisible_by_8)])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _normalize_arg_key(fn, key):
    if isinstance(key, str):
        return key
    if isinstance(key, int):
        return fn.arg_names[key]
    raise TypeError("Signature keys must be string or integer argument indexes")


def _normalize_signature(fn, signature):
    if isinstance(signature, str):
        signature = {i: ty.strip() for i, ty in enumerate(signature.split(",")) if ty.strip()}

    return {_normalize_arg_key(fn, key): value for key, value in signature.items()}


def _normalize_path_key(fn, key):
    if isinstance(key, tuple):
        return key
    if isinstance(key, str):
        return (fn.arg_names.index(key), )
    if isinstance(key, int):
        return (key, )
    raise TypeError("Constant and attribute keys must be strings, integers, or tuple paths")


def _normalize_attrs(fn, attrs):
    if attrs is None:
        return {}

    if isinstance(attrs, AttrsDescriptor):
        normalized = {}
        for key in attrs.divisible_by_16:
            normalized[_normalize_path_key(fn, key)] = [["tt.divisibility", 16]]
        for key in attrs.divisible_by_8:
            normalized[_normalize_path_key(fn, key)] = [["tt.divisibility", 8]]
        return normalized

    return {_normalize_path_key(fn, key): value for key, value in attrs.items()}


def _equal_to_1_constant_value(signature, fn, path):
    if len(path) == 1 and isinstance(path[0], int) and path[0] < len(fn.arg_names):
        dtype = signature.get(fn.arg_names[path[0]])
        if dtype == "i1":
            return True
    return 1


class ASTSource:

    def __init__(self, fn, signature, constexprs=None, attrs=None, constants=None) -> None:
        if constants is not None:
            if constexprs is not None:
                raise TypeError("ASTSource expects either constexprs or constants, not both")
            constexprs = constants

        self.fn = fn
        self.language = Language.TRITON
        self.ext = "ttir"
        self.name = fn.__name__
        self.signature = _normalize_signature(fn, signature)
        self.constants = dict()
        if constexprs is not None:
            for k, v in constexprs.items():
                self.constants[_normalize_path_key(fn, k)] = v
        if isinstance(attrs, AttrsDescriptor):
            for key in attrs.equal_to_1:
                path = _normalize_path_key(fn, key)
                self.constants.setdefault(
                    path, _equal_to_1_constant_value(self.signature, fn, path))
        self.attrs = _normalize_attrs(fn, attrs)
        self._attrs_key = attrs.hash() if isinstance(attrs, AttrsDescriptor) else str(self.attrs)

    def hash(self):
        sorted_sig = [v for k, v in sorted(self.signature.items())]
        get_key = lambda x: x.cache_key if hasattr(x, 'cache_key') else str(x)
        constants_key = '-'.join([get_key(v) for k, v in sorted(self.constants.items())])
        key = f"{self.fn.cache_key}-{self._attrs_key}-{sorted_sig}-{constants_key}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        from .code_generator import ast_to_ttir
        return ast_to_ttir(self.fn, self, context=context, options=options, codegen_fns=codegen_fns,
                           module_map=module_map)

    def parse_options(self):
        return dict()


class IRSource:

    def __init__(self, path, *args):
        if len(args) == 1:
            context = None
            backend = args[0]
        elif len(args) == 2:
            context, backend = args
        else:
            raise TypeError("IRSource expects (path, backend) or (path, context, backend)")

        self.path = path
        path = Path(path)
        self.ext = path.suffix[1:]
        self.language = Language.TRITON
        self.src = path.read_text()

        # We don't have a easy-to-use PTX parser that we can use, so keep that regex for now.
        # TODO - replace with a proper parser
        if self.ext == "ptx":
            self.name, signature = _extract_ptx_entry_signature(self.src)
            types = _parse_ptx_param_types(signature)
            self.signature = {k: convert_type_repr(ty) for k, ty in enumerate(types)}
        else:
            self.module = ir.parse_mlir_module(self.path, context)
            fn_name = self.module.get_entry_func_name()
            self.name = "@" + fn_name
            funcOp = self.module.get_function(fn_name)
            func_ty = self.module.get_function_signature(funcOp)
            self.signature = {k: ty for k, ty in enumerate(func_ty)}

    def hash(self):
        return hashlib.sha256(self.src.encode("utf-8")).hexdigest()

    def make_ir(self, target: GPUTarget, options, codegen_fns, module_map, context):
        if self.ext == "ptx":
            return self.src

        self.module.context = context
        return self.module

    def parse_options(self):
        if self.ext == "ttgir":
            num_warps = self.module.get_int_attr("ttg.num-warps")
            assert num_warps is not None, "Unable to parse ttg.num-warps attribute"
            return {'num_warps': num_warps}
        return dict()


@functools.lru_cache()
def max_shared_mem(device):
    return driver.active.utils.get_device_properties(device)["max_shared_mem"]


def parse(full_name, ext, context):
    if ext == "ttir" or ext == "ttgir":
        module = ir.parse_mlir_module(full_name, context)
        module.context = context
        return module
    if ext == "llir" or ext == "ptx" or ext == "amdgcn":
        return Path(full_name).read_text()
    if ext == "cubin" or ext == "hsaco":
        return Path(full_name).read_bytes()


def _runtime_signature_for_source(src):
    signature = getattr(src, "signature", {})
    if not isinstance(signature, dict):
        return {}

    constexpr_names = set()
    fn = getattr(src, "fn", None)
    arg_names = list(getattr(fn, "arg_names", []))
    constants = getattr(src, "constants", {})
    for path in constants:
        if isinstance(path, tuple) and len(path) == 1 and isinstance(path[0], int) and path[0] < len(arg_names):
            constexpr_names.add(arg_names[path[0]])

    return {str(name): str(dtype) for name, dtype in signature.items() if name not in constexpr_names}


def _serialize_ir_for_storage(module, ext):
    if isinstance(module, (str, bytes)):
        return module
    to_text = getattr(module, "to_text", None)
    if callable(to_text):
        return to_text()
    return str(module)


def _supports_positional_argument(method, count: int) -> bool:
    params = inspect.signature(method).parameters.values()
    positional = {
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    }
    supported = 0
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return True
        if param.kind in positional:
            supported += 1
    return supported >= count


def _add_backend_stages(backend: BaseBackend, stages: dict, options: object, language: Language) -> None:
    add_stages = backend.add_stages
    if _supports_positional_argument(add_stages, 3):
        add_stages(stages, options, language)
    else:
        add_stages(stages, options)


def _get_backend_codegen_implementation(backend: BaseBackend, options: object):
    get_codegen = backend.get_codegen_implementation
    if _supports_positional_argument(get_codegen, 1):
        return get_codegen(options)
    return get_codegen()


def filter_traceback(e: BaseException):
    """
    Removes code_generator.py and related files from tracebacks.

    These are uninteresting to the user -- "just show me *my* code!"
    """
    if knobs.compilation.front_end_debugging:
        return

    if e.__cause__ is not None:
        filter_traceback(e.__cause__)
    if e.__context__ is not None:
        filter_traceback(e.__context__)

    # If a user has a file that matches one of these, they're out of luck.
    BAD_FILES = [
        "/triton/compiler/code_generator.py",
        "/ast.py",
    ]
    BAD_FILES = [bad_file.replace("/", os.sep) for bad_file in BAD_FILES]

    tb = e.__traceback__
    frames = []
    while tb is not None:
        if not any(f for f in BAD_FILES if tb.tb_frame.f_code.co_filename.endswith(f)):
            frames.append(tb)
        tb = tb.tb_next

    for (cur_frame, next_frame) in zip(frames, frames[1:]):
        cur_frame.tb_next = next_frame

    if not frames:
        e.__traceback__ = None
    else:
        frames[-1].tb_next = None
        e.__traceback__ = frames[0]


class CompileTimer:

    def __init__(self) -> None:
        self.start: float = time.time()
        self.ir_initialization_end: float | None = None
        self.lowering_stage_ends: list[tuple[str, float]] = []
        self.store_results_end: float | None = None

    def finished_ir_initialization(self) -> None:
        self.ir_initialization_end = time.time()

    def stage_finished(self, stage_name: str) -> None:
        self.lowering_stage_ends.append((stage_name, time.time()))

    def end(self) -> knobs.CompileTimes:
        timestamp = time.time()
        if self.ir_initialization_end is None:
            self.ir_initialization_end = timestamp
        else:
            self.store_results_end = timestamp

        def delta(start: float, end: float | None) -> int:
            if end is None:
                return 0
            return int((end - start) * 1000000)

        lowering_stage_durations = []
        stage_start = self.ir_initialization_end
        for stage_name, stage_end in self.lowering_stage_ends:
            lowering_stage_durations.append((stage_name, delta(stage_start, stage_end)))
            stage_start = stage_end

        return knobs.CompileTimes(
            ir_initialization=delta(self.start, self.ir_initialization_end),
            lowering_stages=lowering_stage_durations,
            store_results=delta(stage_start, self.store_results_end),
        )


def compile(src, target=None, options=None, _env_vars=None):
    compilation_listener = knobs.compilation.listener
    if compilation_listener:
        timer = CompileTimer()

    if target is None:
        target = driver.active.get_current_target()
    assert isinstance(target, GPUTarget), "target must be of GPUTarget type"
    backend = make_backend(target)
    ir_source = not isinstance(src, ASTSource)
    # create backend
    if ir_source:
        assert isinstance(src, str), "source must be either AST or a filepath"
        src = IRSource(src, backend)

    extra_options = src.parse_options()
    options = backend.parse_options(dict(options or dict(), **extra_options))
    context = backend.make_context(options)
    # create cache manager
    env_vars = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
    key = get_cache_key(src, backend, options, env_vars=env_vars)
    hash = hashlib.sha256(key.encode("utf-8")).hexdigest()
    fn_cache_manager = get_cache_manager(hash)
    # For dumping/overriding only hash the source as we want it to be independent of triton
    # core changes to make it easier to track kernels by hash.
    enable_override = knobs.compilation.override
    enable_ir_dump = knobs.compilation.dump_ir
    store_only_binary = knobs.compilation.store_binary_only
    fn_override_manager = get_override_manager(src.hash()) if enable_override else None
    fn_dump_manager = get_dump_manager(src.hash()) if enable_ir_dump else None
    # Pre-truncate the file name here to avoid hitting the 255 character limit on common platforms.
    # The final file name in the cache will have a format of f"{filename}.{ext}.tmp.pid_{pid}_{uuid}".
    # A PID string can be 5-character long. A UUID string has typically 36 characters. Let's truncate
    # the file name to 150 characters to be safe.
    file_name = src.name[:150]
    metadata_filename = f"{file_name}.json"
    metadata_group = fn_cache_manager.get_group(metadata_filename) or {}
    metadata_path = metadata_group.get(metadata_filename)
    always_compile = knobs.compilation.always_compile
    if not always_compile and metadata_path is not None:
        # cache hit!
        res = CompiledKernel(src, metadata_group, hash)
        if compilation_listener:
            compilation_listener(
                src=src,
                metadata=res.metadata._asdict(),
                metadata_group=metadata_group,
                times=timer.end(),
                cache_hit=True,
            )
        return res

    # initialize metadata
    metadata = {
        "hash": hash,
        "target": target,
        "name": src.name,
        **options.__dict__,
        **env_vars,
    }
    runtime_signature = _runtime_signature_for_source(src)
    metadata["runtime_argument_count"] = len(runtime_signature)
    metadata["runtime_argument_order"] = list(runtime_signature.keys())
    metadata["runtime_argument_types"] = list(runtime_signature.values())
    metadata["triton_version"] = __version__
    # run compilation pipeline  and populate metadata
    stages = dict()
    _add_backend_stages(backend, stages, options, src.language)
    first_stage = list(stages.keys()).index(src.ext)
    # when the source is an IR file, don't apply the passes related to this stage. This makes it easier to write IR level tests.
    if ir_source:
        first_stage += 1

    codegen_fns = _get_backend_codegen_implementation(backend, options)
    module_map = backend.get_module_map()
    module = src.make_ir(target, options, codegen_fns, module_map, context)
    # try:
    #     module = src.make_ir(target, options, codegen_fns, module_map, context)
    # except Exception as e:
    #     filter_traceback(e)
    #     raise

    if ir_source:
        ir_filename = f"{file_name}.{src.ext}"
        metadata_group[ir_filename] = fn_cache_manager.put(_serialize_ir_for_storage(module, src.ext), ir_filename)
    else:
        ir_filename = f"{file_name}.source"
        metadata_group[ir_filename] = fn_cache_manager.put(_serialize_ir_for_storage(module, "source"), ir_filename)

    use_ir_loc = knobs.compilation.use_ir_loc
    if ir_source and use_ir_loc:
        module.create_location_snapshot(src.path)
        print(f"Creating new locations for {src.path}")

    if compilation_listener:
        timer.finished_ir_initialization()
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        ir_filename = f"{file_name}.{ext}"
        if fn_override_manager is None:
            # Users can override kernels at scale by setting `ir_override` in autotune config
            # without TRITON_KERNEL_OVERRIDE
            if (ir_override := metadata.get("ir_override", None)) and ir_override.endswith(f".{ext}"):
                next_module = parse(ir_override, ext, context)
        elif full_name := fn_override_manager.get_file(ir_filename):
            print(f"\nOverriding kernel with file {full_name}")
            next_module = parse(full_name, ext, context)
        suppress_stage_file = getattr(next_module, "suppress_stage_file", False)
        # If TRITON_STORE_BINARY_ONLY is 1, only store cubin/hsaco/json
        if ((not store_only_binary) or (ext in ("cubin", "hsaco", "json"))) and not suppress_stage_file:
            metadata_group[ir_filename] = fn_cache_manager.put(_serialize_ir_for_storage(next_module, ext), ir_filename)
        if fn_dump_manager is not None:
            if not suppress_stage_file:
                fn_dump_manager.put(_serialize_ir_for_storage(next_module, ext), ir_filename)
            if ext == "cubin":
                sass = get_sass(next_module)
                fn_dump_manager.put(sass, file_name + ".sass")
        cache_artifacts = getattr(next_module, "cache_artifacts", None)
        if callable(cache_artifacts):
            for artifact_ext, artifact in cache_artifacts().items():
                artifact_filename = f"{file_name}.{artifact_ext}"
                metadata_group[artifact_filename] = fn_cache_manager.put(_serialize_ir_for_storage(artifact, artifact_ext),
                                                                         artifact_filename)
                if fn_dump_manager is not None:
                    fn_dump_manager.put(_serialize_ir_for_storage(artifact, artifact_ext), artifact_filename)
        # use an env variable to parse ir from file
        if use_ir_loc == ext:
            ir_full_name = fn_cache_manager.get_file(ir_filename)
            next_module.create_location_snapshot(ir_full_name)
            print(f"Creating new locations for {ir_full_name}")
        module = next_module
        if compilation_listener:
            timer.stage_finished(ext)
    # write-back metadata
    metadata_group[metadata_filename] = fn_cache_manager.put(json.dumps(metadata, default=vars), metadata_filename,
                                                             binary=False)
    fn_cache_manager.put_group(metadata_filename, metadata_group)
    # Compilation completed, disabling multithreading in context.
    # This is needed to safely finalize threads pool inside context: if current process forks before
    # python GC deletes context object, thread pool in child process will be invalid, which could
    # lead to child crash or hang.
    #
    # However disabling multithreading causes the code to hang if the ASAN pass is enabled
    # this is likely due to the llvm-symbolizer forking a process
    # TODO: Reconcile the difference here between the ASAN and non-ASAN path with enabling
    # multithreading in the MLIR context
    if not knobs.compilation.enable_asan and hasattr(context, "disable_multithreading"):
        context.disable_multithreading()

    # notify any listener
    if compilation_listener:
        compilation_listener(src=src, metadata=metadata, metadata_group=metadata_group, times=timer.end(),
                             cache_hit=False)
    # return handle to compiled kernel
    return CompiledKernel(src, metadata_group, hash)


def make_backend(target: GPUTarget) -> BaseBackend:
    actives = [x.compiler for x in backends.values() if x.compiler.supports_target(target)]
    if len(actives) != 1:
        raise RuntimeError(
            f"{len(actives)} compatible backends for target ({target.backend}) ({actives}). There should only be one.")
    return actives[0](target)


class LazyDict:

    def __init__(self, data):
        self.data = data
        self.extras = []

    def get(self):
        for func, args in self.extras:
            self.data = self.data | func(*args)
        self.extras.clear()
        return self.data

    def add(self, func, args):
        self.extras.append((func, args))


class AsmDict(dict):

    def __missing__(self, key):

        if key == "sass":
            value = get_sass(self["cubin"])
        else:
            raise KeyError("Unknown key: '%s'" % key)

        self[key] = value
        return value


def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)


_COMMON_REQUIRED_KERNEL_METADATA_FIELDS = (
    "name",
    "shared",
    "num_warps",
    "num_ctas",
    "cluster_dims",
)

_CUDA_REQUIRED_KERNEL_METADATA_FIELDS = (
    "tmem_size",
    "global_scratch_size",
    "global_scratch_align",
    "profile_scratch_size",
    "profile_scratch_align",
)


def _load_kernel_metadata(metadata_path):
    metadata = json.loads(metadata_path.read_text())

    target = metadata.get("target")
    if not isinstance(target, dict) or not {"backend", "arch", "warp_size"}.issubset(target):
        raise KeyError("Compiled kernel metadata missing target backend/arch/warp_size fields")
    target = GPUTarget(target["backend"], target["arch"], target["warp_size"])

    required_fields = list(_COMMON_REQUIRED_KERNEL_METADATA_FIELDS)
    if target.backend == "cuda":
        required_fields.extend(_CUDA_REQUIRED_KERNEL_METADATA_FIELDS)

    missing = [field for field in required_fields if field not in metadata]
    if missing:
        raise KeyError(f"Compiled kernel metadata missing required fields: {', '.join(missing)}")

    cluster_dims = metadata["cluster_dims"]
    if not isinstance(cluster_dims, (list, tuple)) or len(cluster_dims) != 3:
        raise ValueError("Compiled kernel metadata field 'cluster_dims' must contain three dimensions")
    metadata["cluster_dims"] = tuple(cluster_dims)

    metadata["target"] = target
    return metadata


class CompiledKernel:

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = _load_kernel_metadata(metadata_path)
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        # create launcher
        self._run = driver.active.launcher_cls(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        if knobs.runtime.kernel_load_start_hook is not None:
            knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        if knobs.runtime.kernel_load_end_hook is not None:
            knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)

    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run

    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_names = getattr(self.metadata, "runtime_argument_order", None) or self.src.fn.arg_names
        arg_dict = {name: arg for name, arg in zip(arg_names, args)}
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner
