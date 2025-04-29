# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RemoteData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.custom import CustomData

__all__ = [
    "RemoteData",
]

__pdoc__ = {}


class RemoteData(CustomData):
    """Data class for fetching remote data."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.remote")
