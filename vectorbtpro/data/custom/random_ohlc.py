# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RandomOHLCData`."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.ohlcv import nb as ohlcv_nb
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "RandomOHLCData",
]

__pdoc__ = {}


class RandomOHLCData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_random_data_1d_nb`
    and then resampled using `vectorbtpro.ohlcv.nb.ohlc_every_1d_nb`."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.random_ohlc")

    @classmethod
    def generate_symbol(
        cls,
        symbol: tp.Symbol,
        index: tp.Index,
        n_ticks: tp.Optional[tp.ArrayLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.SymbolData:
        """Generate a symbol.

        Args:
            symbol (hashable): Symbol.
            index (pd.Index): Pandas index.
            n_ticks (int or array_like): Number of ticks per bar.

                Flexible argument. Can be a template with a context containing `symbol` and `index`.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            symmetric (bool): Whether to diminish negative returns and make them symmetric to positive ones.
            seed (int): Seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.
            template_context (dict): Template context.

        For defaults, see `custom.random_ohlc` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per symbol using `vectorbtpro.data.base.symbol_dict`.
        """
        n_ticks = cls.resolve_custom_setting(n_ticks, "n_ticks")
        template_context = merge_dicts(dict(symbol=symbol, index=index), template_context)
        n_ticks = substitute_templates(n_ticks, template_context, eval_id="n_ticks")
        n_ticks = broadcast_array_to(n_ticks, len(index))
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        symmetric = cls.resolve_custom_setting(symmetric, "symmetric")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_1d_nb, jitted)
        ticks = func(np.sum(n_ticks), start_value=start_value, mean=mean, std=std, symmetric=symmetric)
        func = jit_reg.resolve_option(ohlcv_nb.ohlc_every_1d_nb, jitted)
        out = func(ticks, n_ticks)
        return pd.DataFrame(out, index=index, columns=["Open", "High", "Low", "Close"])

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[symbol]["Open"].iloc[-1]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, start_value=start_value, **kwargs)
