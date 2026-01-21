"""
FlagLang
"""
from __future__ import annotations
import numpy
import typing
from . import hosting
from . import ir
__all__: list[str] = ['DimensionKind', 'DumpFlags', 'RuntimeTensor', 'Target', 'TensorDesc', 'get_cache_invalidating_env_vars', 'getenv', 'getenv_bool', 'hosting', 'ir']
class DimensionKind:
    """
    Members:
    
      Fixed
    
      Dynamic
    
      Unknown
    """
    Dynamic: typing.ClassVar[DimensionKind]  # value = <DimensionKind.Dynamic: 1>
    Fixed: typing.ClassVar[DimensionKind]  # value = <DimensionKind.Fixed: 0>
    Unknown: typing.ClassVar[DimensionKind]  # value = <DimensionKind.Unknown: 2>
    __members__: typing.ClassVar[dict[str, DimensionKind]]  # value = {'Fixed': <DimensionKind.Fixed: 0>, 'Dynamic': <DimensionKind.Dynamic: 1>, 'Unknown': <DimensionKind.Unknown: 2>}
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DumpFlags:
    """
    Members:
    
      Nothing
    
      ImportOps
    
      PassIR
    
      EGraphCost
    
      Rewrite
    
      Calibration
    
      Evaluator
    
      Compile
    
      Tiling
    
      Schedule
    
      CodeGen
    """
    Calibration: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Calibration: 32>
    CodeGen: typing.ClassVar[DumpFlags]  # value = <DumpFlags.CodeGen: 1024>
    Compile: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Compile: 128>
    EGraphCost: typing.ClassVar[DumpFlags]  # value = <DumpFlags.EGraphCost: 8>
    Evaluator: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Evaluator: 64>
    ImportOps: typing.ClassVar[DumpFlags]  # value = <DumpFlags.ImportOps: 2>
    Nothing: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Nothing: 0>
    PassIR: typing.ClassVar[DumpFlags]  # value = <DumpFlags.PassIR: 4>
    Rewrite: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Rewrite: 16>
    Schedule: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Schedule: 512>
    Tiling: typing.ClassVar[DumpFlags]  # value = <DumpFlags.Tiling: 256>
    __members__: typing.ClassVar[dict[str, DumpFlags]]  # value = {'Nothing': <DumpFlags.Nothing: 0>, 'ImportOps': <DumpFlags.ImportOps: 2>, 'PassIR': <DumpFlags.PassIR: 4>, 'EGraphCost': <DumpFlags.EGraphCost: 8>, 'Rewrite': <DumpFlags.Rewrite: 16>, 'Calibration': <DumpFlags.Calibration: 32>, 'Evaluator': <DumpFlags.Evaluator: 64>, 'Compile': <DumpFlags.Compile: 128>, 'Tiling': <DumpFlags.Tiling: 256>, 'Schedule': <DumpFlags.Schedule: 512>, 'CodeGen': <DumpFlags.CodeGen: 1024>}
    def __and__(self, other: typing.Any) -> typing.Any:
        ...
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __ge__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __gt__(self, other: typing.Any) -> bool:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __invert__(self) -> typing.Any:
        ...
    def __le__(self, other: typing.Any) -> bool:
        ...
    def __lt__(self, other: typing.Any) -> bool:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, other: typing.Any) -> typing.Any:
        ...
    def __rand__(self, other: typing.Any) -> typing.Any:
        ...
    def __repr__(self) -> str:
        ...
    def __ror__(self, other: typing.Any) -> typing.Any:
        ...
    def __rxor__(self, other: typing.Any) -> typing.Any:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    def __xor__(self, other: typing.Any) -> typing.Any:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class RuntimeTensor:
    @staticmethod
    def from_numpy(arg0: numpy.ndarray) -> RuntimeTensor:
        ...
    @staticmethod
    def from_object(arg0: ...) -> RuntimeTensor:
        ...
    def copy_to(self, arg0: RuntimeTensor) -> None:
        ...
    def to_numpy(self) -> numpy.ndarray:
        ...
    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        ...
    @property
    def shape(self) -> list[int]:
        ...
class Target:
    def __init__(self, arg0: str) -> None:
        ...
class TensorDesc:
    size: int
    start: int
    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        ...
    @dtype.setter
    def dtype(self, arg1: typing.Any) -> None:
        ...
def get_cache_invalidating_env_vars() -> dict[str, str]:
    ...
def getenv(*args, **kwargs):
    ...
def getenv_bool(*args, **kwargs):
    ...
__version__: str = ''
_cleanup: typing.Any  # value = <capsule object>
