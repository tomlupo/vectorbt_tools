# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RANDX`."""

import numpy as np
import pandas as pd

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_place_nb

__all__ = [
    "RANDX",
]

RANDX = SignalFactory(
    class_name="RANDX",
    module_name=__name__,
    short_name="randx",
    mode="exits",
).with_place_func(
    exit_place_func_nb=rand_place_nb,
    exit_settings=dict(
        pass_kwargs=dict(n=np.array([1])),
    ),
    seed=None,
)


class _RANDX(RANDX):
    """Random exit signal generator based on the number of signals.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.rand_place_nb`.

    See `RAND` for notes on parameters.

    Usage:
        Generate an exit for each entry:

        ```pycon
        >>> from vectorbtpro import *

        >>> entries = pd.Series([True, False, False, True, False, False])
        >>> randx = vbt.RANDX.run(entries, seed=42)

        >>> randx.exits
        0    False
        1    False
        2     True
        3    False
        4     True
        5    False
        dtype: bool
        ```
    """

    pass


setattr(RANDX, "__doc__", _RANDX.__doc__)
