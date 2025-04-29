# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RPROBCX`."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.rprobx import rprobx_config, rprobx_func_config

__all__ = [
    "RPROBCX",
]

RPROBCX = SignalFactory(
    **rprobx_config.merge_with(
        dict(
            class_name="RPROBCX",
            short_name="rprobcx",
            mode="chain",
        )
    ),
).with_place_func(**rprobx_func_config)


class _RPROBCX(RPROBCX):
    """Random exit signal generator based on probabilities.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBCX, "__doc__", _RPROBCX.__doc__)
