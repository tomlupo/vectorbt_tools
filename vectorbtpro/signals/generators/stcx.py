# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `STCX`."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.stx import stx_config, stx_func_config

__all__ = [
    "STCX",
]

STCX = SignalFactory(
    **stx_config.merge_with(
        dict(
            class_name="STCX",
            short_name="stcx",
            mode="chain",
        )
    )
).with_place_func(**stx_func_config)


class _STCX(STCX):
    """Exit signal generator based on stop values.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.stop_place_nb`.

    See `STX` for notes on parameters."""

    pass


setattr(STCX, "__doc__", _STCX.__doc__)
