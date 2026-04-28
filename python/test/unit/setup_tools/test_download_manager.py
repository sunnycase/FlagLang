import importlib
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from python.setup_tools.utils import tools  # noqa: E402


def test_flagtree_submodule_dir_uses_repo_root():
    repo_root = Path(__file__).resolve().parents[4]

    assert Path(tools.flagtree_root_dir) == repo_root
    assert Path(tools.flagtree_submodule_dir) == repo_root / "third_party"


def test_flagtree_cache_uses_repo_root_for_copy_destinations(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[4]
    monkeypatch.delenv("FLAGTREE_BACKEND", raising=False)
    monkeypatch.delenv("FLAGTREE_OFFLINE_BUILD_DIR", raising=False)
    monkeypatch.delenv("TRITON_OFFLINE_BUILD", raising=False)
    monkeypatch.setenv("FLAGTREE_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)

    setup_helper = importlib.import_module("python.setup_tools.setup_helper")
    cache = setup_helper.FlagTreeCache()

    assert Path(cache.flagtree_dir) == repo_root


def test_offline_url_download_decompresses_cached_archive(tmp_path, monkeypatch):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    archive = offline_dir / "payload.tar.gz"
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "value.txt").write_text("cached")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source_dir, arcname="payload")

    dst_dir = tmp_path / "cache"
    monkeypatch.setenv("FLAGTREE_OFFLINE_BUILD_DIR", str(offline_dir))
    manager = tools.DownloadManager()

    assert manager.download(
        url="https://example.invalid/payload.tar.gz",
        path=dst_dir,
        file_name="renamed_payload",
        required=True,
    )
    assert (dst_dir / "renamed_payload" / "value.txt").read_text() == "cached"


def test_offline_url_download_copies_cached_file_name(tmp_path, monkeypatch):
    offline_dir = tmp_path / "offline"
    offline_dir.mkdir()
    (offline_dir / "plugin.so").write_text("binary")
    dst_dir = tmp_path / "cache"
    monkeypatch.setenv("FLAGTREE_OFFLINE_BUILD_DIR", str(offline_dir))
    manager = tools.DownloadManager()

    assert manager.download(
        url="https://example.invalid/plugin.tar.gz",
        path=dst_dir,
        file_name="plugin.so",
        required=True,
    )
    assert (dst_dir / "plugin.so").read_text() == "binary"


def test_sys_clone_checks_commands_and_runs_checkout_in_clone(tmp_path, monkeypatch):
    module = tools.Module(
        name="dep",
        url="https://example.invalid/dep.git",
        commit_id="abc123",
        dst_path=str(tmp_path / "dep"),
    )
    calls = []

    def fake_run(cmd, check, cwd=None):
        calls.append((cmd, check, cwd))
        assert check is True
        if cmd[0:2] == ["git", "clone"]:
            (tmp_path / "dep").mkdir()
        elif cmd[0:2] == ["git", "checkout"]:
            assert cwd == module.dst_path
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)

    assert tools.DownloadManager().sys_clone(module)
    assert calls == [
        (["git", "clone", module.url, module.dst_path], True, None),
        (["git", "checkout", module.commit_id], True, module.dst_path),
    ]


def test_sys_clone_reports_failure_and_removes_partial_clone(tmp_path, monkeypatch):
    module = tools.Module(
        name="dep",
        url="https://example.invalid/dep.git",
        dst_path=str(tmp_path / "dep"),
    )

    def fake_run(cmd, check, cwd=None):
        (tmp_path / "dep").mkdir(exist_ok=True)
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(tools.subprocess, "run", fake_run)
    monkeypatch.setattr(tools.NetConfig, "max_retry", 1)

    assert not tools.DownloadManager().sys_clone(module)
    assert not (tmp_path / "dep").exists()
