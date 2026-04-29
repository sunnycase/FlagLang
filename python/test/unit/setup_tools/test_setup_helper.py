import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.setup_tools import setup_helper  # noqa: E402
from python.setup_tools.utils import ascend  # noqa: E402


def test_install_extension_ignores_missing_optional_hook(monkeypatch):

    class BackendWithoutHook:
        pass

    monkeypatch.setattr(setup_helper, "activated_module", BackendWithoutHook())

    assert setup_helper.install_extension(build_ext=object()) is None


def test_install_extension_propagates_backend_failures(monkeypatch):

    class BackendWithFailingHook:

        def install_extension(self, *args, **kwargs):
            raise FileNotFoundError("missing backend executable")

    monkeypatch.setattr(setup_helper, "activated_module", BackendWithFailingHook())

    with pytest.raises(FileNotFoundError, match="missing backend executable"):
        setup_helper.install_extension(build_ext=object())


def test_ascend_patch_packages_have_matching_package_dirs():
    packages = ascend.get_extra_install_packages()
    package_dirs = ascend.get_package_dir()

    assert packages == [
        "triton.triton_patch",
        "triton.triton_patch.language",
        "triton.triton_patch.compiler",
        "triton.triton_patch.runtime",
    ]
    assert set(packages) == set(package_dirs)
    assert all("/" not in package for package in packages)
    assert package_dirs["triton.triton_patch"] == "third_party/ascend/triton_patch/python/triton_patch"
