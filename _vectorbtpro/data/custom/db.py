# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `DBData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.local import LocalData

__all__ = [
    "DBData",
]

__pdoc__ = {}


class DBData(LocalData):
    """Data class for fetching database data."""

    _settings_path: tp.SettingsPath = dict(custom="data.custom.db")
