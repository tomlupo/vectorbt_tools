# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RPROB`."""

import numpy as np

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb

__all__ = [
    "RPROB",
]

RPROB = SignalFactory(
    class_name="RPROB",
    module_name=__name__,
    short_name="rprob",
    mode="entries",
    param_names=["prob"],
).with_place_func(
    entry_place_func_nb=rand_by_prob_place_nb,
    entry_settings=dict(
        pass_params=["prob"],
        pass_kwargs=["pick_first"],
    ),
    param_settings=dict(
        prob=flex_elem_param_config,
    ),
    seed=None,
)


class _RPROB(RPROB):
    """Random entry signal generator based on probabilities.

    Generates `entries` based on `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    Usage:
        Generate three columns with different entry probabilities:

        ```pycon
        >>> from vectorbtpro import *

        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=[0., 0.5, 1.], seed=42)

        >>> rprob.entries
        rprob_prob    0.0    0.5   1.0
        0           False   True  True
        1           False   True  True
        2           False  False  True
        3           False  False  True
        4           False  False  True
        ```

        Probability can also be set per row, column, or element:

        ```pycon
        >>> rprob = vbt.RPROB.run(input_shape=(5,), prob=np.array([0., 0., 1., 1., 1.]), seed=42)

        >>> rprob.entries
        0    False
        1    False
        2     True
        3     True
        4     True
        Name: array_0, dtype: bool
        ```
    """

    pass


setattr(RPROB, "__doc__", _RPROB.__doc__)
