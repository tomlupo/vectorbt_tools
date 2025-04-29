# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RPROBNX`."""

import numpy as np

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb

__all__ = [
    "RPROBNX",
]

RPROBNX = SignalFactory(
    class_name="RPROBNX",
    module_name=__name__,
    short_name="rprobnx",
    mode="both",
    param_names=["entry_prob", "exit_prob"],
).with_place_func(
    entry_place_func_nb=rand_by_prob_place_nb,
    entry_settings=dict(
        pass_params=["entry_prob"],
        pass_kwargs=["pick_first"],
    ),
    exit_place_func_nb=rand_by_prob_place_nb,
    exit_settings=dict(
        pass_params=["exit_prob"],
        pass_kwargs=["pick_first"],
    ),
    param_settings=dict(
        entry_prob=flex_elem_param_config,
        exit_prob=flex_elem_param_config,
    ),
    seed=None,
)


class _RPROBNX(RPROBNX):
    """Random entry and exit signal generator based on probabilities.

    Generates `entries` and `exits` based on `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters.

    Usage:
        Test all probability combinations:

        ```pycon
        >>> from vectorbtpro import *

        >>> rprobnx = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[0.5, 1.],
        ...     exit_prob=[0.5, 1.],
        ...     param_product=True,
        ...     seed=42)

        >>> rprobnx.entries
        rprobnx_entry_prob    0.5    0.5    1.0    0.5
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                    True   True   True   True
        1                   False  False  False  False
        2                   False  False  False   True
        3                   False  False  False  False
        4                   False  False   True   True

        >>> rprobnx.exits
        rprobnx_entry_prob    0.5    0.5    1.0    1.0
        rprobnx_exit_prob     0.5    1.0    0.5    1.0
        0                   False  False  False  False
        1                   False   True  False   True
        2                   False  False  False  False
        3                   False  False   True   True
        4                    True  False  False  False
        ```

        Probabilities can also be set per row, column, or element:

        ```pycon
        >>> entry_prob1 = np.array([1., 0., 1., 0., 1.])
        >>> entry_prob2 = np.array([0., 1., 0., 1., 0.])
        >>> rprobnx = vbt.RPROBNX.run(
        ...     input_shape=(5,),
        ...     entry_prob=[entry_prob1, entry_prob2],
        ...     exit_prob=1.,
        ...     seed=42)

        >>> rprobnx.entries
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                     True   False
        1                    False    True
        2                     True   False
        3                    False    True
        4                     True   False

        >>> rprobnx.exits
        rprobnx_entry_prob array_0 array_1
        rprobnx_exit_prob      1.0     1.0
        0                    False   False
        1                     True   False
        2                    False    True
        3                     True   False
        4                    False    True
        ```
    """

    pass


setattr(RPROBNX, "__doc__", _RPROBNX.__doc__)
