from triton.backends.compiler import Language
from triton.compiler.compiler import _add_backend_stages, _get_backend_codegen_implementation


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
