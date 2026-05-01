from abc import ABCMeta, abstractmethod
from typing import Callable, List, Protocol, Sequence


class Benchmarker(Protocol):

    def __call__(self, kernel_call: Callable, *, quantiles: List[float], **kwargs) -> Sequence[float]:
        pass


class DriverBase(metaclass=ABCMeta):

    @classmethod
    @abstractmethod
    def is_active(self):
        pass

    @staticmethod
    @abstractmethod
    def map_python_to_cpp_type(ty: str) -> str:
        """
        Converts a Triton type string to its corresponding C++ type string for this backend.

        Args:
            ty (str): The Triton type string. e.g., 'i32', '*fp16', 'fp32'.

        Returns:
            str: The C++ type string.
        """
        pass

    @abstractmethod
    def get_current_target(self):
        pass

    @abstractmethod
    def get_active_torch_device(self):
        pass

    @abstractmethod
    def get_benchmarker(self) -> Benchmarker:
        """
        Return the benchmarking function that this backend should use by default.
        """
        raise NotImplementedError

    def get_device_interface(self):
        """
        Return the torch device interface used by the generic benchmarker.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement get_device_interface() for generic do_bench")

    def get_empty_cache_for_benchmark(self):
        """
        Allocate the cache-flush buffer used by the generic benchmarker.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_empty_cache_for_benchmark() for generic do_bench")

    def clear_cache(self, cache):
        """
        Clear the cache-flush buffer before each generic benchmark iteration.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement clear_cache(cache) for generic do_bench")

    def __init__(self) -> None:
        pass


class GPUDriver(DriverBase):

    def __init__(self):
        # TODO: support other frameworks than torch
        import torch
        self.get_device_capability = torch.cuda.get_device_capability
        try:
            from torch._C import _cuda_getCurrentRawStream
            self.get_current_stream = _cuda_getCurrentRawStream
        except ImportError:
            self.get_current_stream = lambda idx: torch.cuda.current_stream(idx).cuda_stream
        self.get_current_device = torch.cuda.current_device
        self.set_current_device = torch.cuda.set_device

    # TODO: remove once TMA is cleaned up
    def assemble_tensormap_to_arg(self, tensormaps_info, args):
        return args
