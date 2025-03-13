# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RANDNX`."""

from vectorbtpro.indicators.configs import flex_col_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_enex_apply_nb

__all__ = [
    "RANDNX",
]

RANDNX = SignalFactory(
    class_name="RANDNX",
    module_name=__name__,
    short_name="randnx",
    mode="both",
    param_names=["n"],
).with_apply_func(
    rand_enex_apply_nb,
    require_input_shape=True,
    param_settings=dict(
        n=flex_col_param_config,
    ),
    kwargs_as_args=["entry_wait", "exit_wait"],
    entry_wait=1,
    exit_wait=1,
    seed=None,
)


class _RANDNX(RANDNX):
    """Random entry and exit signal generator based on the number of signals.

    Generates `entries` and `exits` based on `vectorbtpro.signals.nb.rand_enex_apply_nb`.

    See `RAND` for notes on parameters.

    Usage:
        Test three different entry and exit counts:

        ```pycon
        >>> from vectorbtpro import *

        >>> randnx = vbt.RANDNX.run(
        ...     input_shape=(6,),
        ...     n=[1, 2, 3],
        ...     seed=42)

        >>> randnx.entries
        randnx_n      1      2      3
        0          True   True   True
        1         False  False  False
        2         False   True   True
        3         False  False  False
        4         False  False   True
        5         False  False  False

        >>> randnx.exits
        randnx_n      1      2      3
        0         False  False  False
        1          True   True   True
        2         False  False  False
        3         False   True   True
        4         False  False  False
        5         False  False   True
        ```
    """

    pass


setattr(RANDNX, "__doc__", _RANDNX.__doc__)
