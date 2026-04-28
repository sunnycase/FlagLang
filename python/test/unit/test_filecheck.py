import pytest
import triton

from triton._filecheck import _resolve_filecheck_path, run_filecheck_test


def test_resolve_filecheck_path_from_env(tmp_path, monkeypatch):
    filecheck = tmp_path / "FileCheck"
    filecheck.write_text("#!/bin/sh\nexit 0\n")
    filecheck.chmod(0o755)
    monkeypatch.setenv("FILECHECK_PATH", str(filecheck))

    assert _resolve_filecheck_path() == str(filecheck)


def test_resolve_filecheck_path_rejects_missing_env(monkeypatch):
    monkeypatch.setenv("FILECHECK_PATH", "/missing/FileCheck")

    with pytest.raises(FileNotFoundError, match="FILECHECK_PATH points to missing FileCheck binary"):
        _resolve_filecheck_path()


@triton.jit
def anchor(v):
    pass


# Smoke test to make sure filecheck is working correctly.
def test_filecheck_positive():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: T.PrimFunc("test_kernel"
        scalar = 42
        # CHECK: test_filecheck.anchor__i32__(42)
        anchor(scalar)

    run_filecheck_test(test_kernel)


def test_filecheck_negative():

    @triton.jit
    def test_kernel():
        # CHECK-LABEL: T.PrimFunc("test_kernel"
        scalar = 11
        # CHECK: test_filecheck.anchor__i32__(42)
        anchor(scalar)

    with pytest.raises(ValueError, match=r"expected string not found in input\n # CHECK: test_filecheck\.anchor__i32__\(42\)"):
        run_filecheck_test(test_kernel)
