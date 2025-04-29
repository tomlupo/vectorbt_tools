# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `CustomData`."""

import fnmatch
import re

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import Data

__all__ = [
    "CustomData",
]

__pdoc__ = {}


class CustomData(Data):
    """Data class for fetching custom data."""

    _settings_path: tp.SettingsPath = dict(custom=None)

    @classmethod
    def get_custom_settings(cls, *args, **kwargs) -> dict:
        """`CustomData.get_settings` with `path_id="custom"`."""
        return cls.get_settings(*args, path_id="custom", **kwargs)

    @classmethod
    def has_custom_settings(cls, *args, **kwargs) -> bool:
        """`CustomData.has_settings` with `path_id="custom"`."""
        return cls.has_settings(*args, path_id="custom", **kwargs)

    @classmethod
    def get_custom_setting(cls, *args, **kwargs) -> tp.Any:
        """`CustomData.get_setting` with `path_id="custom"`."""
        return cls.get_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def has_custom_setting(cls, *args, **kwargs) -> bool:
        """`CustomData.has_setting` with `path_id="custom"`."""
        return cls.has_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def resolve_custom_setting(cls, *args, **kwargs) -> tp.Any:
        """`CustomData.resolve_setting` with `path_id="custom"`."""
        return cls.resolve_setting(*args, path_id="custom", **kwargs)

    @classmethod
    def set_custom_settings(cls, *args, **kwargs) -> None:
        """`CustomData.set_settings` with `path_id="custom"`."""
        cls.set_settings(*args, path_id="custom", **kwargs)

    @staticmethod
    def key_match(key: str, pattern: str, use_regex: bool = False):
        """Return whether key matches pattern.

        If `use_regex` is True, checks against a regular expression.
        Otherwise, checks against a glob-style pattern."""
        if use_regex:
            return re.match(pattern, key)
        return re.match(fnmatch.translate(pattern), key)
