# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Named tuples and enumerated types for indicators."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.formatting import prettify

__pdoc__all__ = __all__ = [
    "Pivot",
    "TrendMode",
    "HurstMethod",
    "SuperTrendAIS",
    "SuperTrendAOS",
]

__pdoc__ = {}


# ############# Enums ############# #


class PivotT(tp.NamedTuple):
    Valley: int = -1
    Peak: int = 1


Pivot = PivotT()
"""_"""

__pdoc__[
    "Pivot"
] = f"""Pivot.

```python
{prettify(Pivot)}
```
"""


class TrendModeT(tp.NamedTuple):
    Downtrend: int = -1
    Uptrend: int = 1


TrendMode = TrendModeT()
"""_"""

__pdoc__[
    "TrendMode"
] = f"""Trend mode.

```python
{prettify(TrendMode)}
```
"""


class HurstMethodT(tp.NamedTuple):
    Standard: int = 0
    LogRS: int = 1
    RS: int = 2
    DMA: int = 3
    DSOD: int = 4


HurstMethod = HurstMethodT()
"""_"""

__pdoc__[
    "HurstMethod"
] = f"""Hurst method.

```python
{prettify(HurstMethod)}
```
"""


# ############# States ############# #


class SuperTrendAIS(tp.NamedTuple):
    i: int
    high: float
    low: float
    close: float
    prev_close: float
    prev_upper: float
    prev_lower: float
    prev_direction: int
    nobs: int
    weighted_avg: float
    old_wt: float
    period: int
    multiplier: float


__pdoc__[
    "SuperTrendAIS"
] = """A named tuple representing the input state of 
`vectorbtpro.indicators.nb.supertrend_acc_nb`."""


class SuperTrendAOS(tp.NamedTuple):
    nobs: int
    weighted_avg: float
    old_wt: float
    upper: float
    lower: float
    trend: float
    direction: int
    long: float
    short: float


__pdoc__[
    "SuperTrendAOS"
] = """A named tuple representing the output state of 
`vectorbtpro.indicators.nb.supertrend_acc_nb`."""
