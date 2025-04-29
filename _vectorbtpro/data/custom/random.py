# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RandomData`."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_1d_array
from vectorbtpro.data import nb
from vectorbtpro.data.custom.synthetic import SyntheticData
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.random_ import set_seed

__all__ = [
    "RandomData",
]

__pdoc__ = {}


class RandomData(SyntheticData):
    """`SyntheticData` for data generated using `vectorbtpro.data.nb.generate_random_data_nb`."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.random")

    @classmethod
    def generate_key(
        cls,
        key: tp.Key,
        index: tp.Index,
        columns: tp.Union[tp.Hashable, tp.IndexLike] = None,
        start_value: tp.Optional[float] = None,
        mean: tp.Optional[float] = None,
        std: tp.Optional[float] = None,
        symmetric: tp.Optional[bool] = None,
        seed: tp.Optional[int] = None,
        jitted: tp.JittedOption = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate a feature or symbol.

        Args:
            key (hashable): Feature or symbol.
            index (pd.Index): Pandas index.
            columns (hashable or index_like): Column names.

                Provide a single value (hashable) to make a Series.
            start_value (float): Value at time 0.

                Does not appear as the first value in the output data.
            mean (float): Drift, or mean of the percentage change.
            std (float): Standard deviation of the percentage change.
            symmetric (bool): Whether to diminish negative returns and make them symmetric to positive ones.
            seed (int): Seed to make output deterministic.
            jitted (any): See `vectorbtpro.utils.jitting.resolve_jitted_option`.

        For defaults, see `custom.random` in `vectorbtpro._settings.data`.

        !!! note
            When setting a seed, remember to pass a seed per feature/symbol using
            `vectorbtpro.data.base.feature_dict`/`vectorbtpro.data.base.symbol_dict` or generally
            `vectorbtpro.data.base.key_dict`.
        """
        if checks.is_hashable(columns):
            columns = [columns]
            make_series = True
        else:
            make_series = False
        if not isinstance(columns, pd.Index):
            columns = pd.Index(columns)
        start_value = cls.resolve_custom_setting(start_value, "start_value")
        mean = cls.resolve_custom_setting(mean, "mean")
        std = cls.resolve_custom_setting(std, "std")
        symmetric = cls.resolve_custom_setting(symmetric, "symmetric")
        seed = cls.resolve_custom_setting(seed, "seed")
        if seed is not None:
            set_seed(seed)

        func = jit_reg.resolve_option(nb.generate_random_data_nb, jitted)
        out = func(
            (len(index), len(columns)),
            start_value=to_1d_array(start_value),
            mean=to_1d_array(mean),
            std=to_1d_array(std),
            symmetric=to_1d_array(symmetric),
        )
        if make_series:
            return pd.Series(out[:, 0], index=index, name=columns[0])
        return pd.DataFrame(out, index=index, columns=columns)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        _ = fetch_kwargs.pop("start_value", None)
        start_value = self.data[key].iloc[-2]
        fetch_kwargs["seed"] = None
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, start_value=start_value, **kwargs)
        return self.fetch_symbol(key, start_value=start_value, **kwargs)
