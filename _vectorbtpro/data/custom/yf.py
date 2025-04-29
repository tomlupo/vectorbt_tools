# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `YFData`."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.parsing import get_func_kwargs

__all__ = [
    "YFData",
]

__pdoc__ = {}


class YFData(RemoteData):
    """Data class for fetching from Yahoo Finance.

    See https://github.com/ranaroussi/yfinance for API.

    See `YFData.fetch_symbol` for arguments.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> data = vbt.YFData.pull(
        ...     "BTC-USD",
        ...     start="2020-01-01",
        ...     end="2021-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.yf")

    _feature_config: tp.ClassVar[Config] = HybridConfig(
        {
            "Dividends": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
            "Stock Splits": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.nonzero_prod_reduce_nb,
                )
            ),
            "Capital Gains": dict(
                resample_func=lambda self, obj, resampler: obj.vbt.resample_apply(
                    resampler,
                    generic_nb.sum_reduce_nb,
                )
            ),
        }
    )

    @property
    def feature_config(self) -> Config:
        return self._feature_config

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        period: tp.Optional[str] = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        **history_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Yahoo Finance.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            **history_kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.

        For defaults, see `custom.yf` in `vectorbtpro._settings.data`.

        !!! warning
            Data coming from Yahoo is not the most stable data out there. Yahoo may manipulate data
            how they want, add noise, return missing data points (see volume in the example below), etc.
            It's only used in vectorbt for demonstration purposes.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("yfinance")
        import yfinance as yf

        period = cls.resolve_custom_setting(period, "period")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        history_kwargs = cls.resolve_custom_setting(history_kwargs, "history_kwargs", merge=True)

        ticker = yf.Ticker(symbol)
        def_history_kwargs = get_func_kwargs(yf.Tickers.history)
        ticker_tz = ticker._get_ticker_tz(
            history_kwargs.get("proxy", def_history_kwargs["proxy"]),
            history_kwargs.get("timeout", def_history_kwargs["timeout"]),
        )
        if tz is None:
            tz = ticker_tz
        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz=ticker_tz)
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz=ticker_tz)
        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is not None:
            multiplier, unit = split
            if unit == "D":
                unit = "d"
            elif unit == "W":
                unit = "wk"
            elif unit == "M":
                unit = "mo"
            timeframe = str(multiplier) + unit

        df = ticker.history(period=period, start=start, end=end, interval=timeframe, **history_kwargs)
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize(ticker_tz)

        if not df.empty:
            if start is not None:
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)


YFData.override_feature_config_doc(__pdoc__)
