# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `SyntheticData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SyntheticData",
]

__pdoc__ = {}


class SyntheticData(CustomData):
    """Data class for fetching synthetic data.

    Exposes an abstract class method `SyntheticData.generate_symbol`.
    Everything else is taken care of."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.synthetic")

    @classmethod
    def generate_key(cls, key: tp.Key, index: tp.Index, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Abstract method to generate data of a feature or symbol."""
        raise NotImplementedError

    @classmethod
    def generate_feature(cls, feature: tp.Feature, index: tp.Index, **kwargs) -> tp.FeatureData:
        """Abstract method to generate data of a feature.

        Uses `SyntheticData.generate_key` with `key_is_feature=True`."""
        return cls.generate_key(feature, index, key_is_feature=True, **kwargs)

    @classmethod
    def generate_symbol(cls, symbol: tp.Symbol, index: tp.Index, **kwargs) -> tp.SymbolData:
        """Abstract method to generate data for a symbol.

        Uses `SyntheticData.generate_key` with `key_is_feature=False`."""
        return cls.generate_key(symbol, index, key_is_feature=False, **kwargs)

    @classmethod
    def fetch_key(
        cls,
        key: tp.Symbol,
        key_is_feature: bool = False,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        periods: tp.Optional[int] = None,
        timeframe: tp.Optional[tp.FrequencyLike] = None,
        tz: tp.TimezoneLike = None,
        normalize: tp.Optional[bool] = None,
        inclusive: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.KeyData:
        """Generate data of a feature or symbol.

        Generates datetime index using `vectorbtpro.utils.datetime_.date_range` and passes it to
        `SyntheticData.generate_key` to fill the Series/DataFrame with generated data.

        For defaults, see `custom.synthetic` in `vectorbtpro._settings.data`."""
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        normalize = cls.resolve_custom_setting(normalize, "normalize")
        inclusive = cls.resolve_custom_setting(inclusive, "inclusive")

        index = dt.date_range(
            start=start,
            end=end,
            periods=periods,
            freq=timeframe,
            normalize=normalize,
            inclusive=inclusive,
        )
        if tz is None:
            tz = index.tz
        if len(index) == 0:
            raise ValueError("Date range is empty")
        if key_is_feature:
            return cls.generate_feature(key, index, **kwargs), dict(tz=tz, freq=timeframe)
        return cls.generate_symbol(key, index, **kwargs), dict(tz=tz, freq=timeframe)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Generate data of a feature.

        Uses `SyntheticData.fetch_key` with `key_is_feature=True`."""
        return cls.fetch_key(feature, key_is_feature=True, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Generate data for a symbol.

        Uses `SyntheticData.fetch_key` with `key_is_feature=False`."""
        return cls.fetch_key(symbol, key_is_feature=False, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update data of a feature or symbol."""
        fetch_kwargs = self.select_fetch_kwargs(key)
        fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Update data of a feature.

        Uses `SyntheticData.update_key` with `key_is_feature=True`."""
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `SyntheticData.update_key` with `key_is_feature=False`."""
        return self.update_key(symbol, key_is_feature=False, **kwargs)
