from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]


def test_cuda_runtime_module_frees_device_sections():
    header = (REPO_ROOT / "src" / "Native" / "src" / "runtime" / "cuda" / "runtime_module.h").read_text()
    source = (REPO_ROOT / "src" / "Native" / "src" / "runtime" / "cuda" / "runtime_module.cpp").read_text()

    assert "~cuda_runtime_module() override;" in header
    assert "cuda_runtime_module::~cuda_runtime_module()" in source
    assert "release_device_section(rdata_);" in source
    assert "release_device_section(thread_local_rdata_);" in source
    assert "release_device_section(warp_local_rdata_);" in source
    assert "release_device_section(block_local_rdata_);" in source
    assert "cudaFree(const_cast<std::byte *>(section.data()))" in source


def test_cuda_runtime_module_frees_section_after_copy_failure():
    source = (REPO_ROOT / "src" / "Native" / "src" / "runtime" / "cuda" / "runtime_module.cpp").read_text()

    assert "auto copy_status =" in source
    assert "if (copy_status != cudaSuccess)" in source
    assert "auto free_status = cudaFree(device_ptr);" in source
    assert "CHECK_CUDA(copy_status);" in source
