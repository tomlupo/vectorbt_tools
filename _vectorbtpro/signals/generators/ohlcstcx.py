# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `OHLCSTCX`."""

from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.generators.ohlcstx import ohlcstx_config, ohlcstx_func_config, bind_ohlcstx_plot

__all__ = [
    "OHLCSTCX",
]

OHLCSTCX = SignalFactory(
    **ohlcstx_config.merge_with(
        dict(
            class_name="OHLCSTCX",
            short_name="ohlcstcx",
            mode="chain",
        )
    ),
).with_place_func(
    **ohlcstx_func_config,
)


class _OHLCSTCX(OHLCSTCX):
    """Exit signal generator based on OHLC and stop values.

    Generates chain of `new_entries` and `exits` based on `entries` and
    `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

    See `OHLCSTX` for notes on parameters."""

    plot = bind_ohlcstx_plot(OHLCSTCX, "new_entries")


setattr(OHLCSTCX, "__doc__", _OHLCSTCX.__doc__)
setattr(OHLCSTCX, "plot", _OHLCSTCX.plot)
