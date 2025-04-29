# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RAND`."""

import numpy as np

from vectorbtpro.indicators.configs import flex_col_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_place_nb

__all__ = [
    "RAND",
]

RAND = SignalFactory(
    class_name="RAND",
    module_name=__name__,
    short_name="rand",
    mode="entries",
    param_names=["n"],
).with_place_func(
    entry_place_func_nb=rand_place_nb,
    entry_settings=dict(
        pass_params=["n"],
    ),
    param_settings=dict(
        n=flex_col_param_config,
    ),
    seed=None,
)


class _RAND(RAND):
    """Random entry signal generator based on the number of signals.

    Generates `entries` based on `vectorbtpro.signals.nb.rand_place_nb`.

    !!! hint
        Parameter `n` can be either a single value (per frame) or a NumPy array (per column).
        To generate multiple combinations, pass it as a list.

    Usage:
        Test three different entry counts values:

        ```pycon
        >>> from vectorbtpro import *

        >>> rand = vbt.RAND.run(input_shape=(6,), n=[1, 2, 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3
        0        True   True   True
        1       False  False   True
        2       False  False  False
        3       False   True  False
        4       False  False   True
        5       False  False  False
        ```

        Entry count can also be set per column:

        ```pycon
        >>> rand = vbt.RAND.run(input_shape=(8, 2), n=[np.array([1, 2]), 3], seed=42)

        >>> rand.entries
        rand_n      1      2      3      3
                    0      1      0      1
        0       False  False   True  False
        1        True  False  False  False
        2       False  False  False   True
        3       False   True   True  False
        4       False  False  False  False
        5       False  False  False   True
        6       False  False   True  False
        7       False   True  False   True
        ```
    """

    pass


setattr(RAND, "__doc__", _RAND.__doc__)
