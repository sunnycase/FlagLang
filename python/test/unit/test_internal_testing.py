import importlib
import sys

import triton


class NoDriverConfig:

    @property
    def active(self):
        raise RuntimeError("0 active drivers ([]). There should only be one.")


def test_internal_testing_import_without_active_driver(monkeypatch):
    monkeypatch.setattr(triton.runtime, "driver", NoDriverConfig())
    original_module = sys.modules.pop("triton._internal_testing", None)

    try:
        module = importlib.import_module("triton._internal_testing")

        assert module.get_current_target() is None
        assert module.supports_tma() is False
    finally:
        sys.modules.pop("triton._internal_testing", None)
        if original_module is not None:
            sys.modules["triton._internal_testing"] = original_module
