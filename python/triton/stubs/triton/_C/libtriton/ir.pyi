from __future__ import annotations
import typing
__all__: list[str] = ['ACQUIRE', 'ACQUIRE_RELEASE', 'ATOMIC_OP', 'CA', 'CACHE_MODIFIER', 'CG', 'CS', 'CTA', 'EVICTION_POLICY', 'EVICT_FIRST', 'EVICT_LAST', 'GPU', 'IEEE', 'INPUT_PRECISION', 'MEM_SEMANTIC', 'MEM_SYNC_SCOPE', 'NONE', 'NORMAL', 'PADDING_OPTION', 'PAD_NAN', 'PAD_ZERO', 'PROPAGATE_NAN', 'RELAXED', 'RELEASE', 'ROUNDING_MODE', 'SYSTEM', 'TF32', 'TF32x3', 'WB', 'WT']
class ATOMIC_OP:
    """
    Members:

      ADD

      FADD

      AND

      OR

      XOR

      XCHG

      MAX

      MIN

      UMIN

      UMAX
    """
    ADD: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.ADD: 4>
    AND: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.AND: 1>
    FADD: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.FADD: 5>
    MAX: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.MAX: 6>
    MIN: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.MIN: 7>
    OR: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.OR: 2>
    UMAX: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.UMAX: 8>
    UMIN: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.UMIN: 9>
    XCHG: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.XCHG: 10>
    XOR: typing.ClassVar[ATOMIC_OP]  # value = <ATOMIC_OP.XOR: 3>
    __members__: typing.ClassVar[dict[str, ATOMIC_OP]]  # value = {'ADD': <ATOMIC_OP.ADD: 4>, 'FADD': <ATOMIC_OP.FADD: 5>, 'AND': <ATOMIC_OP.AND: 1>, 'OR': <ATOMIC_OP.OR: 2>, 'XOR': <ATOMIC_OP.XOR: 3>, 'XCHG': <ATOMIC_OP.XCHG: 10>, 'MAX': <ATOMIC_OP.MAX: 6>, 'MIN': <ATOMIC_OP.MIN: 7>, 'UMIN': <ATOMIC_OP.UMIN: 9>, 'UMAX': <ATOMIC_OP.UMAX: 8>}
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
class CACHE_MODIFIER:
    """
    Members:

      NONE

      CA

      CG

      WB

      CS

      WT
    """
    CA: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CA: 2>
    CG: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CG: 3>
    CS: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.CS: 5>
    NONE: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.NONE: 1>
    WB: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.WB: 4>
    WT: typing.ClassVar[CACHE_MODIFIER]  # value = <CACHE_MODIFIER.WT: 6>
    __members__: typing.ClassVar[dict[str, CACHE_MODIFIER]]  # value = {'NONE': <CACHE_MODIFIER.NONE: 1>, 'CA': <CACHE_MODIFIER.CA: 2>, 'CG': <CACHE_MODIFIER.CG: 3>, 'WB': <CACHE_MODIFIER.WB: 4>, 'CS': <CACHE_MODIFIER.CS: 5>, 'WT': <CACHE_MODIFIER.WT: 6>}
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
class EVICTION_POLICY:
    """
    Members:

      NORMAL

      EVICT_FIRST

      EVICT_LAST
    """
    EVICT_FIRST: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.EVICT_FIRST: 2>
    EVICT_LAST: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.EVICT_LAST: 3>
    NORMAL: typing.ClassVar[EVICTION_POLICY]  # value = <EVICTION_POLICY.NORMAL: 1>
    __members__: typing.ClassVar[dict[str, EVICTION_POLICY]]  # value = {'NORMAL': <EVICTION_POLICY.NORMAL: 1>, 'EVICT_FIRST': <EVICTION_POLICY.EVICT_FIRST: 2>, 'EVICT_LAST': <EVICTION_POLICY.EVICT_LAST: 3>}
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
class INPUT_PRECISION:
    """
    Members:

      TF32

      TF32x3

      IEEE
    """
    IEEE: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.IEEE: 2>
    TF32: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.TF32: 0>
    TF32x3: typing.ClassVar[INPUT_PRECISION]  # value = <INPUT_PRECISION.TF32x3: 1>
    __members__: typing.ClassVar[dict[str, INPUT_PRECISION]]  # value = {'TF32': <INPUT_PRECISION.TF32: 0>, 'TF32x3': <INPUT_PRECISION.TF32x3: 1>, 'IEEE': <INPUT_PRECISION.IEEE: 2>}
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
class MEM_SEMANTIC:
    """
    Members:

      ACQUIRE_RELEASE

      ACQUIRE

      RELEASE

      RELAXED
    """
    ACQUIRE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE: 2>
    ACQUIRE_RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.ACQUIRE_RELEASE: 4>
    RELAXED: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELAXED: 1>
    RELEASE: typing.ClassVar[MEM_SEMANTIC]  # value = <MEM_SEMANTIC.RELEASE: 3>
    __members__: typing.ClassVar[dict[str, MEM_SEMANTIC]]  # value = {'ACQUIRE_RELEASE': <MEM_SEMANTIC.ACQUIRE_RELEASE: 4>, 'ACQUIRE': <MEM_SEMANTIC.ACQUIRE: 2>, 'RELEASE': <MEM_SEMANTIC.RELEASE: 3>, 'RELAXED': <MEM_SEMANTIC.RELAXED: 1>}
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
class MEM_SYNC_SCOPE:
    """
    Members:

      GPU

      CTA

      SYSTEM
    """
    CTA: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.CTA: 2>
    GPU: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.GPU: 1>
    SYSTEM: typing.ClassVar[MEM_SYNC_SCOPE]  # value = <MEM_SYNC_SCOPE.SYSTEM: 3>
    __members__: typing.ClassVar[dict[str, MEM_SYNC_SCOPE]]  # value = {'GPU': <MEM_SYNC_SCOPE.GPU: 1>, 'CTA': <MEM_SYNC_SCOPE.CTA: 2>, 'SYSTEM': <MEM_SYNC_SCOPE.SYSTEM: 3>}
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
class PADDING_OPTION:
    """
    Members:

      PAD_ZERO

      PAD_NAN
    """
    PAD_NAN: typing.ClassVar[PADDING_OPTION]  # value = <PADDING_OPTION.PAD_NAN: 2>
    PAD_ZERO: typing.ClassVar[PADDING_OPTION]  # value = <PADDING_OPTION.PAD_ZERO: 1>
    __members__: typing.ClassVar[dict[str, PADDING_OPTION]]  # value = {'PAD_ZERO': <PADDING_OPTION.PAD_ZERO: 1>, 'PAD_NAN': <PADDING_OPTION.PAD_NAN: 2>}
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
class PROPAGATE_NAN:
    """
    Members:

      NONE

      ALL
    """
    ALL: typing.ClassVar[PROPAGATE_NAN]  # value = <PROPAGATE_NAN.ALL: 65535>
    NONE: typing.ClassVar[PROPAGATE_NAN]  # value = <PROPAGATE_NAN.NONE: 0>
    __members__: typing.ClassVar[dict[str, PROPAGATE_NAN]]  # value = {'NONE': <PROPAGATE_NAN.NONE: 0>, 'ALL': <PROPAGATE_NAN.ALL: 65535>}
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
class ROUNDING_MODE:
    """
    Members:

      RTZ

      RTNE
    """
    RTNE: typing.ClassVar[ROUNDING_MODE]  # value = <ROUNDING_MODE.RTNE: 1>
    RTZ: typing.ClassVar[ROUNDING_MODE]  # value = <ROUNDING_MODE.RTZ: 0>
    __members__: typing.ClassVar[dict[str, ROUNDING_MODE]]  # value = {'RTZ': <ROUNDING_MODE.RTZ: 0>, 'RTNE': <ROUNDING_MODE.RTNE: 1>}
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
ACQUIRE: MEM_SEMANTIC  # value = <MEM_SEMANTIC.ACQUIRE: 2>
ACQUIRE_RELEASE: MEM_SEMANTIC  # value = <MEM_SEMANTIC.ACQUIRE_RELEASE: 4>
CA: CACHE_MODIFIER  # value = <CACHE_MODIFIER.CA: 2>
CG: CACHE_MODIFIER  # value = <CACHE_MODIFIER.CG: 3>
CS: CACHE_MODIFIER  # value = <CACHE_MODIFIER.CS: 5>
CTA: MEM_SYNC_SCOPE  # value = <MEM_SYNC_SCOPE.CTA: 2>
EVICT_FIRST: EVICTION_POLICY  # value = <EVICTION_POLICY.EVICT_FIRST: 2>
EVICT_LAST: EVICTION_POLICY  # value = <EVICTION_POLICY.EVICT_LAST: 3>
GPU: MEM_SYNC_SCOPE  # value = <MEM_SYNC_SCOPE.GPU: 1>
IEEE: INPUT_PRECISION  # value = <INPUT_PRECISION.IEEE: 2>
NONE: CACHE_MODIFIER  # value = <CACHE_MODIFIER.NONE: 1>
NORMAL: EVICTION_POLICY  # value = <EVICTION_POLICY.NORMAL: 1>
PAD_NAN: PADDING_OPTION  # value = <PADDING_OPTION.PAD_NAN: 2>
PAD_ZERO: PADDING_OPTION  # value = <PADDING_OPTION.PAD_ZERO: 1>
RELAXED: MEM_SEMANTIC  # value = <MEM_SEMANTIC.RELAXED: 1>
RELEASE: MEM_SEMANTIC  # value = <MEM_SEMANTIC.RELEASE: 3>
SYSTEM: MEM_SYNC_SCOPE  # value = <MEM_SYNC_SCOPE.SYSTEM: 3>
TF32: INPUT_PRECISION  # value = <INPUT_PRECISION.TF32: 0>
TF32x3: INPUT_PRECISION  # value = <INPUT_PRECISION.TF32x3: 1>
WB: CACHE_MODIFIER  # value = <CACHE_MODIFIER.WB: 4>
WT: CACHE_MODIFIER  # value = <CACHE_MODIFIER.WT: 6>
