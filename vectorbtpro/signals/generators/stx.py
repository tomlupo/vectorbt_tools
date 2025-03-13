# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `STX`."""

import numpy as np

from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import stop_place_nb
from vectorbtpro.utils.config import ReadonlyConfig

__all__ = [
    "STX",
]

stx_config = ReadonlyConfig(
    dict(
        class_name="STX",
        module_name=__name__,
        short_name="stx",
        mode="exits",
        input_names=["entry_ts", "ts", "follow_ts"],
        in_output_names=["stop_ts"],
        param_names=["stop", "trailing"],
    )
)
"""Factory config for `STX`."""

stx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_ts", "ts", "follow_ts"],
            pass_in_outputs=["stop_ts"],
            pass_params=["stop", "trailing"],
        ),
        param_settings=dict(
            stop=flex_elem_param_config,
            trailing=flex_elem_param_config,
        ),
        trailing=False,
        ts=np.nan,
        follow_ts=np.nan,
        stop_ts=np.nan,
    )
)
"""Exit function config for `STX`."""

STX = SignalFactory(**stx_config).with_place_func(**stx_func_config)


class _STX(STX):
    """Exit signal generator based on stop values.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.stop_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists."""

    pass


setattr(STX, "__doc__", _STX.__doc__)
