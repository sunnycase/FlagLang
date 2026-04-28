import os
import platform
import shutil
import sysconfig
import sys
from pathlib import Path


def get_base_dir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _get_cmake_dir():
    plat_name = sysconfig.get_platform()
    python_version = sysconfig.get_python_version()
    dir_name = f"cmake.{plat_name}-{sys.implementation.name}-{python_version}"
    return Path(get_base_dir()) / "build" / dir_name


def get_cmake_dir():
    cmake_dir = os.getenv("TRITON_BUILD_DIR", default=_get_cmake_dir())
    cmake_dir = Path(cmake_dir)
    cmake_dir.mkdir(parents=True, exist_ok=True)
    return cmake_dir


def update_symlink(link_path, source_path):
    source_path = Path(source_path)
    link_path = Path(link_path)

    if link_path.is_symlink():
        link_path.unlink()
    elif link_path.exists():
        if link_path.is_dir():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()

    print(f"creating symlink: {link_path} -> {source_path}", file=sys.stderr)
    link_path.absolute().parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(source_path.absolute(), target_is_directory=True)


_CONAN_TOOLCHAIN_ARCHES = {
    "AMD64": "x86_64",
    "x86_64": "x86_64",
    "arm64": "aarch64",
    "aarch64": "aarch64",
    "riscv64": "riscv64",
}

_CONAN_TOOLCHAIN_OSES = {
    "Windows": "windows",
    "Linux": "linux",
    "Darwin": "macos",
}


def _available_toolchain_profiles(toolchains_dir):
    profiles = sorted(path.name.removesuffix(".profile.jinja") for path in toolchains_dir.glob("*.profile.jinja"))
    return ", ".join(profiles) if profiles else "<none>"


def get_host_toolchain_profile(base_dir=None, machine=None, system=None):
    machine = platform.machine() if machine is None else machine
    system = platform.system() if system is None else system
    base_dir = Path(get_base_dir() if base_dir is None else base_dir)

    try:
        toolchain_arch = _CONAN_TOOLCHAIN_ARCHES[machine]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported host architecture {machine!r} for Conan profile selection. "
            "Add an explicit toolchains/<arch>-<os>.profile.jinja profile before building on this host.") from exc

    try:
        toolchain_os = _CONAN_TOOLCHAIN_OSES[system]
    except KeyError as exc:
        raise RuntimeError(
            f"Unsupported host operating system {system!r} for Conan profile selection. "
            "Add an explicit toolchains/<arch>-<os>.profile.jinja profile before building on this host.") from exc

    profile_path = base_dir / "toolchains" / f"{toolchain_arch}-{toolchain_os}.profile.jinja"
    if not profile_path.is_file():
        supported = _available_toolchain_profiles(base_dir / "toolchains")
        raise RuntimeError(
            f"No Conan profile is available for host platform {machine}/{system}: expected {profile_path}. "
            f"Supported profiles: {supported}. Add the missing profile instead of reusing an unrelated toolchain.")
    return profile_path
