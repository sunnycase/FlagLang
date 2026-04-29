import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from third_party.tsingmicro.backend.cpu_driver import CPUDriver


def test_cpu_driver_satisfies_driver_base_contract():
    driver = CPUDriver()

    calls = []

    def kernel_call():
        calls.append(None)

    benchmarker = driver.get_benchmarker()
    median_ms = benchmarker(kernel_call, quantiles=[0.5], warmup=0, rep=1)

    assert calls
    assert isinstance(median_ms, float)


def test_cpu_driver_active_torch_device_is_cpu():
    torch = pytest.importorskip("torch")

    assert CPUDriver().get_active_torch_device() == torch.device("cpu")
