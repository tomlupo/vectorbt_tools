# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `FeatherData`."""

from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FeatherData",
]

__pdoc__ = {}

FeatherDataT = tp.TypeVar("FeatherDataT", bound="FeatherData")


class FeatherData(FileData):
    """Data class for fetching Feather data using PyArrow."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.feather")

    @classmethod
    def list_paths(cls, path: tp.PathLike = ".", **match_path_kwargs) -> tp.List[Path]:
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            path = path / "*.feather"
        return cls.match_path(path, **match_path_kwargs)

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        paths: tp.Any = None,
    ) -> tp.Kwargs:
        keys_meta = FileData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None and paths is None:
            keys_meta["keys"] = "*.feather"
        return keys_meta

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        tz: tp.TimezoneLike = None,
        index_col: tp.Optional[tp.MaybeSequence[tp.IntStr]] = None,
        squeeze: tp.Optional[bool] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the Feather file of a feature or symbol.

        Args:
            key (hashable): Feature or symbol.
            path (str): Path.

                If `path` is None, uses `key` as the path to the Feather file.
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            index_col (int, str, or sequence): Position(s) or name(s) of column(s) that should become the index.

                Will only apply if the fetched object has a default index.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            **read_kwargs: Other keyword arguments passed to `pd.read_feather`.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_feather.html for other arguments.

        For defaults, see `custom.feather` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pyarrow")

        tz = cls.resolve_custom_setting(tz, "tz")
        index_col = cls.resolve_custom_setting(index_col, "index_col")
        if index_col is False:
            index_col = None
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if path is None:
            path = key
        obj = pd.read_feather(path, **read_kwargs)

        if isinstance(obj, pd.DataFrame) and checks.is_default_index(obj.index):
            if index_col is not None:
                if checks.is_int(index_col):
                    keys = obj.columns[index_col]
                elif isinstance(index_col, str):
                    keys = index_col
                else:
                    keys = []
                    for col in index_col:
                        if checks.is_int(col):
                            keys.append(obj.columns[col])
                        else:
                            keys.append(col)
                obj = obj.set_index(keys)
                if not isinstance(obj.index, pd.MultiIndex):
                    if obj.index.name == "index":
                        obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the Feather file of a feature.

        Uses `FeatherData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the Feather file of a symbol.

        Uses `FeatherData.fetch_key`."""
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: tp.Key, key_is_feature: bool = False, **kwargs) -> tp.KeyData:
        """Update data of a feature or symbol."""
        fetch_kwargs = self.select_fetch_kwargs(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if key_is_feature:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Update data of a feature.

        Uses `FeatherData.update_key` with `key_is_feature=True`."""
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `FeatherData.update_key` with `key_is_feature=False`."""
        return self.update_key(symbol, key_is_feature=False, **kwargs)
