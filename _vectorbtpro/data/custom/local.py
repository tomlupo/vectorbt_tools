# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `LocalData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "LocalData",
]

__pdoc__ = {}


class LocalData(CustomData):
    """Data class for fetching local data."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.local")
