import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from python.build_helpers import get_host_toolchain_profile, update_symlink  # noqa: E402


def test_get_host_toolchain_profile_selects_existing_linux_profile():
    profile = get_host_toolchain_profile(machine="x86_64", system="Linux")

    assert profile.name == "x86_64-linux.profile.jinja"
    assert profile.is_file()


def test_get_host_toolchain_profile_rejects_missing_profile(tmp_path: Path):
    with pytest.raises(RuntimeError, match="No Conan profile is available"):
        get_host_toolchain_profile(tmp_path, machine="aarch64", system="Linux")


def test_get_host_toolchain_profile_rejects_unknown_arch(tmp_path: Path):
    with pytest.raises(RuntimeError, match="Unsupported host architecture"):
        get_host_toolchain_profile(tmp_path, machine="sparc64", system="Linux")


def test_update_symlink_replaces_regular_file(tmp_path: Path):
    source = tmp_path / "source_compile_commands.json"
    source.write_text("new")
    link = tmp_path / "compile_commands.json"
    link.write_text("old")

    update_symlink(link, source)

    assert link.is_symlink()
    assert link.read_text() == "new"
