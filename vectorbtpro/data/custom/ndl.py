# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `NDLData`."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "NDLData",
]

__pdoc__ = {}

NDLDataT = tp.TypeVar("NDLDataT", bound="NDLData")


class NDLData(RemoteData):
    """Data class for fetching from Nasdaq Data Link.

    See https://github.com/Nasdaq/data-link-python for API.

    See `NDLData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.NDLData.set_custom_settings(
        ...     api_key="YOUR_KEY"
        ... )
        ```

        * Pull a dataset:

        ```pycon
        >>> data = vbt.NDLData.pull(
        ...     "FRED/GDP",
        ...     start="2001-12-31",
        ...     end="2005-12-31"
        ... )
        ```

        * Pull a datatable:

        ```pycon
        >>> data = vbt.NDLData.pull(
        ...     "MER/F1",
        ...     data_format="datatable",
        ...     compnumber="39102",
        ...     paginate=True
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.ndl")

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        api_key: tp.Optional[str] = None,
        data_format: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        tz: tp.TimezoneLike = None,
        column_indices: tp.Optional[tp.MaybeIterable[int]] = None,
        **params,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Nasdaq Data Link.

        Args:
            symbol (str): Symbol.
            api_key (str): API key.
            data_format (str): Data format.

                Supported are "dataset" and "datatable".
            start (any): Retrieve data rows on and after the specified start date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): Retrieve data rows up to and including the specified end date.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            column_indices (int or iterable): Request one or more specific columns.

                Column 0 is the date column and is always returned. Data begins at column 1.
            **params: Keyword arguments sent as field/value params to Nasdaq Data Link with no interference.

        For defaults, see `custom.ndl` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("nasdaqdatalink")

        import nasdaqdatalink

        api_key = cls.resolve_custom_setting(api_key, "api_key")
        data_format = cls.resolve_custom_setting(data_format, "data_format")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        tz = cls.resolve_custom_setting(tz, "tz")
        column_indices = cls.resolve_custom_setting(column_indices, "column_indices")
        if column_indices is not None:
            if isinstance(column_indices, int):
                dataset = symbol + "." + str(column_indices)
            else:
                dataset = [symbol + "." + str(index) for index in column_indices]
        else:
            dataset = symbol
        params = cls.resolve_custom_setting(params, "params", merge=True)

        # Establish the timestamps
        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
            start_date = pd.Timestamp(start).isoformat()
            if "start_date" not in params:
                params["start_date"] = start_date
        else:
            start_date = None
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
            end_date = pd.Timestamp(end).isoformat()
            if "end_date" not in params:
                params["end_date"] = end_date
        else:
            end_date = None

        # Collect and format the data
        if data_format.lower() == "dataset":
            df = nasdaqdatalink.get(
                dataset,
                api_key=api_key,
                **params,
            )
        else:
            df = nasdaqdatalink.get_table(
                dataset,
                api_key=api_key,
                **params,
            )
        new_columns = []
        for c in df.columns:
            new_c = c
            if isinstance(symbol, str):
                new_c = new_c.replace(symbol + " - ", "")
            if new_c == "Last":
                new_c = "Close"
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        if df.index.name == "None":
            df.index.name = None

        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            if start is not None:
                start = dt.to_timestamp(start, tz=df.index.tz)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = dt.to_timestamp(end, tz=df.index.tz)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
