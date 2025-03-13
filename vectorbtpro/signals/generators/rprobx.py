# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RPROBX`."""

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import rand_by_prob_place_nb
from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "RPROBX",
]

rprobx_config = ReadonlyConfig(
    dict(
        class_name="RPROBX",
        module_name=__name__,
        short_name="rprobx",
        mode="exits",
        param_names=["prob"],
    ),
)
"""Factory config for `RPROBX`."""

rprobx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=rand_by_prob_place_nb,
        exit_settings=dict(
            pass_params=["prob"],
            pass_kwargs=["pick_first"],
        ),
        param_settings=dict(
            prob=flex_elem_param_config,
        ),
        seed=None,
    )
)
"""Exit function config for `RPROBX`."""

RPROBX = SignalFactory(**rprobx_config).with_place_func(**rprobx_func_config)


class _RPROBX(RPROBX):
    """Random exit signal generator based on probabilities.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.rand_by_prob_place_nb`.

    See `RPROB` for notes on parameters."""

    pass


setattr(RPROBX, "__doc__", _RPROBX.__doc__)
