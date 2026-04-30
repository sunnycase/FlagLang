from types import SimpleNamespace

import triton
from triton.backends.compiler import BaseBackend, GPUTarget, Language
from triton.compiler import compiler
from triton.compiler.compiler import (
    ASTSource,
    _add_backend_stages,
    _get_backend_codegen_implementation,
    _should_store_compilation_artifact,
)


class FakeModule:

    def to_text(self):
        return "source-like ttir"


class FakeNativeCompilation:
    suppress_stage_file = True

    def cache_artifacts(self):
        return {
            "ntt_cu": "__global__ void kernel() {}",
            "ptx": ".visible .entry kernel() { ret; }",
            "compiler_log": "nvcc --gpu-architecture=sm_80",
        }


class BinaryOnlyOptions(SimpleNamespace):

    def hash(self):
        return "binary-only-options"


class BinaryOnlyBackend(BaseBackend):
    binary_ext = "cubin"

    @staticmethod
    def supports_target(target):
        return target.backend == "cuda"

    def hash(self):
        return "binary-only-test-backend"

    def parse_options(self, options):
        return BinaryOnlyOptions(num_warps=1, num_ctas=1, cluster_dims=(1, 1, 1))

    def add_stages(self, stages, options, language):
        stages["ttir"] = lambda src, metadata: src
        stages["ptx"] = lambda src, metadata: FakeNativeCompilation()
        stages["cubin"] = self.make_cubin

    def make_cubin(self, src, metadata):
        metadata.update({
            "shared": 0,
            "num_warps": 1,
            "num_ctas": 1,
            "cluster_dims": (1, 1, 1),
            "tmem_size": 0,
            "global_scratch_size": 0,
            "global_scratch_align": 1,
            "profile_scratch_size": 0,
            "profile_scratch_align": 1,
        })
        return b"compiled cubin"

    def load_dialects(self, context):
        return None

    def make_context(self, options):
        return object()

    def get_module_map(self):
        return {}

    def get_codegen_implementation(self, options):
        return {}

    def pack_metadata(self, metadata):
        return ()


def _binary_only_kernel():
    return None


_binary_only_kernel.arg_names = []
_binary_only_kernel.cache_key = "binary-only-kernel-cache-key"


class LegacyBackendHooks:

    def add_stages(self, stages, options):
        stages["legacy"] = options

    def get_codegen_implementation(self):
        return {"kind": "legacy"}


class LanguageAwareBackendHooks:

    def add_stages(self, stages, options, language):
        stages["language"] = (options, language)

    def get_codegen_implementation(self, options):
        return {"kind": options}


def test_backend_hook_dispatch_supports_legacy_signatures():
    backend = LegacyBackendHooks()
    stages = {}

    _add_backend_stages(backend, stages, "opts", Language.TRITON)

    assert stages == {"legacy": "opts"}
    assert _get_backend_codegen_implementation(backend, "opts") == {"kind": "legacy"}


def test_backend_hook_dispatch_supports_language_aware_signatures():
    backend = LanguageAwareBackendHooks()
    stages = {}

    _add_backend_stages(backend, stages, "opts", Language.GLUON)

    assert stages == {"language": ("opts", Language.GLUON)}
    assert _get_backend_codegen_implementation(backend, "opts") == {"kind": "opts"}


def test_binary_only_cache_artifact_filter_keeps_only_binaries_and_json():
    for ext in ("cubin", "hsaco", "json"):
        assert _should_store_compilation_artifact(ext, store_only_binary=True)

    for ext in ("source", "ttir", "ttgir", "llir", "ptx", "ntt_cu", "compiler_log"):
        assert not _should_store_compilation_artifact(ext, store_only_binary=True)

    assert _should_store_compilation_artifact("ptx", store_only_binary=False)


def test_compile_binary_only_cache_skips_initial_ir_and_native_text_artifacts(monkeypatch, tmp_path):
    target = GPUTarget("cuda", 80, 32)
    backend = BinaryOnlyBackend(target)
    src = ASTSource(_binary_only_kernel, signature={})

    monkeypatch.setattr(compiler, "make_backend", lambda actual_target: backend)
    monkeypatch.setattr(ASTSource, "make_ir", lambda self, *args, **kwargs: FakeModule())

    with triton.knobs.compilation.scope(), triton.knobs.cache.scope():
        triton.knobs.compilation.store_binary_only = True
        triton.knobs.compilation.always_compile = True
        triton.knobs.cache.dir = str(tmp_path)

        compiled = compiler.compile(src, target=target)

    cached_names = set(compiled.metadata_group)
    assert f"{src.name}.json" in cached_names
    assert f"{src.name}.cubin" in cached_names
    assert f"{src.name}.source" not in cached_names
    assert f"{src.name}.ttir" not in cached_names
    assert f"{src.name}.ptx" not in cached_names
    assert f"{src.name}.ntt_cu" not in cached_names
    assert f"{src.name}.compiler_log" not in cached_names
