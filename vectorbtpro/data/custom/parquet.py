# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `ParquetData`."""

import re
from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ParquetData",
]

__pdoc__ = {}

ParquetDataT = tp.TypeVar("ParquetDataT", bound="ParquetData")


class ParquetData(FileData):
    """Data class for fetching Parquet data using PyArrow or FastParquet."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.parquet")

    @classmethod
    def is_parquet_file(cls, path: tp.PathLike) -> bool:
        """Return whether the path is a Parquet file."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_file() and ".parquet" in path.suffixes:
            return True
        return False

    @classmethod
    def is_parquet_group_dir(cls, path: tp.PathLike) -> bool:
        """Return whether the path is a directory that is a group of Parquet partitions.

        !!! note
            Assumes the Hive partitioning scheme."""
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            partition_regex = r"^(.+)=(.+)"
            if re.match(partition_regex, path.name):
                for p in path.iterdir():
                    if cls.is_parquet_group_dir(p) or cls.is_parquet_file(p):
                        return True
        return False

    @classmethod
    def is_parquet_dir(cls, path: tp.PathLike) -> bool:
        """Return whether the path is a directory that is a group itself or
        contains groups of Parquet partitions."""
        if cls.is_parquet_group_dir(path):
            return True
        if not isinstance(path, Path):
            path = Path(path)
        if path.exists() and path.is_dir():
            for p in path.iterdir():
                if cls.is_parquet_group_dir(p):
                    return True
        return False

    @classmethod
    def is_dir_match(cls, path: tp.PathLike) -> bool:
        return cls.is_parquet_dir(path)

    @classmethod
    def is_file_match(cls, path: tp.PathLike) -> bool:
        return cls.is_parquet_file(path)

    @classmethod
    def list_partition_cols(cls, path: tp.PathLike) -> tp.List[str]:
        """List partitioning columns under a path.

        !!! note
            Assumes the Hive partitioning scheme."""
        if not isinstance(path, Path):
            path = Path(path)
        partition_cols = []
        found_last_level = False
        while not found_last_level:
            found_new_level = False
            for p in path.iterdir():
                if cls.is_parquet_group_dir(p):
                    partition_cols.append(p.name.split("=")[0])
                    path = p
                    found_new_level = True
                    break
            if not found_new_level:
                found_last_level = True
        return partition_cols

    @classmethod
    def is_default_partition_col(cls, level: str) -> bool:
        """Return whether a partitioning column is a default partitioning column."""
        return re.match(r"^(\bgroup\b)|(group_\d+)", level) is not None

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
            keys_meta["keys"] = cls.list_paths()
        return keys_meta

    @classmethod
    def fetch_key(
        cls,
        key: tp.Key,
        path: tp.Any = None,
        tz: tp.TimezoneLike = None,
        squeeze: tp.Optional[bool] = None,
        keep_partition_cols: tp.Optional[bool] = None,
        engine: tp.Optional[str] = None,
        **read_kwargs,
    ) -> tp.KeyData:
        """Fetch the Parquet file of a feature or symbol.

        Args:
            key (hashable): Feature or symbol.
            path (str): Path.

                If `path` is None, uses `key` as the path to the Parquet file.
            tz (any): Target timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            keep_partition_cols (bool): Whether to return partitioning columns (if any).

                If None, will remove any partitioning column that is "group" or "group_{index}".

                Retrieves the list of partitioning columns with `ParquetData.list_partition_cols`.
            engine (str): See `pd.read_parquet`.
            **read_kwargs: Other keyword arguments passed to `pd.read_parquet`.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html for other arguments.

        For defaults, see `custom.parquet` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.module_ import assert_can_import, assert_can_import_any

        tz = cls.resolve_custom_setting(tz, "tz")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        keep_partition_cols = cls.resolve_custom_setting(keep_partition_cols, "keep_partition_cols")
        engine = cls.resolve_custom_setting(engine, "engine")
        read_kwargs = cls.resolve_custom_setting(read_kwargs, "read_kwargs", merge=True)

        if engine == "pyarrow":
            assert_can_import("pyarrow")
        elif engine == "fastparquet":
            assert_can_import("fastparquet")
        elif engine == "auto":
            assert_can_import_any("pyarrow", "fastparquet")
        else:
            raise ValueError(f"Invalid option engine='{engine}'")

        if path is None:
            path = key
        obj = pd.read_parquet(path, engine=engine, **read_kwargs)

        if keep_partition_cols in (None, False):
            if cls.is_parquet_dir(path):
                drop_columns = []
                partition_cols = cls.list_partition_cols(path)
                for col in obj.columns:
                    if col in partition_cols:
                        if keep_partition_cols is False or cls.is_default_partition_col(col):
                            drop_columns.append(col)
                obj = obj.drop(drop_columns, axis=1)
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: tp.Feature, **kwargs) -> tp.FeatureData:
        """Fetch the Parquet file of a feature.

        Uses `ParquetData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Fetch the Parquet file of a symbol.

        Uses `ParquetData.fetch_key`."""
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

        Uses `ParquetData.update_key` with `key_is_feature=True`."""
        return self.update_key(feature, key_is_feature=True, **kwargs)

    def update_symbol(self, symbol: tp.Symbol, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `ParquetData.update_key` with `key_is_feature=False`."""
        return self.update_key(symbol, key_is_feature=False, **kwargs)
