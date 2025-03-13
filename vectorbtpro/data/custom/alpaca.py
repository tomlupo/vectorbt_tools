# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `AlpacaData`."""

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from alpaca.common.rest import RESTClient as AlpacaClientT
except ImportError:
    AlpacaClientT = tp.Any

__all__ = [
    "AlpacaData",
]

AlpacaDataT = tp.TypeVar("AlpacaDataT", bound="AlpacaData")


class AlpacaData(RemoteData):
    """Data class for fetching from Alpaca.

    See https://github.com/alpacahq/alpaca-py for API.

    See `AlpacaData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional for crypto):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.AlpacaData.set_custom_settings(
        ...     client_config=dict(
        ...         api_key="YOUR_KEY",
        ...         secret_key="YOUR_SECRET"
        ...     )
        ... )
        ```

        * Pull stock data:

        ```pycon
        >>> data = vbt.AlpacaData.pull(
        ...     "AAPL",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```

        * Pull crypto data:

        ```pycon
        >>> data = vbt.AlpacaData.pull(
        ...     "BTCUSD",
        ...     client_type="crypto",
        ...     start="2021-01-01",
        ...     end="2022-01-01",
        ...     timeframe="1 day"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.alpaca")

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        status: tp.Optional[str] = None,
        asset_class: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        trading_client: tp.Optional[AlpacaClientT] = None,
        client_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all symbols.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`.

        Arguments `status`, `asset_class`, and `exchange` can be strings, such as `asset_class="crypto"`.
        For possible values, take a look into `alpaca.trading.enums`.

        !!! note
            If you get an authorization error, make sure that you either enable or disable
            the `paper` flag in `client_config` depending upon the account whose credentials you used.
            By default, the credentials are assumed to be of a live trading account (`paper=False`)."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetStatus, AssetClass, AssetExchange

        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if trading_client is None:
            arg_names = get_func_arg_names(TradingClient.__init__)
            client_config = {k: v for k, v in client_config.items() if k in arg_names}
            trading_client = TradingClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")

        if status is not None:
            if isinstance(status, str):
                status = getattr(AssetStatus, status.upper())
        if asset_class is not None:
            if isinstance(asset_class, str):
                asset_class = getattr(AssetClass, asset_class.upper())
        if exchange is not None:
            if isinstance(exchange, str):
                exchange = getattr(AssetExchange, exchange.upper())
        search_params = GetAssetsRequest(status=status, asset_class=asset_class, exchange=exchange)
        assets = trading_client.get_all_assets(search_params)
        all_symbols = []
        for asset in assets:
            symbol = asset.symbol
            if pattern is not None:
                if not cls.key_match(symbol, pattern, use_regex=use_regex):
                    continue
            all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def resolve_client(
        cls,
        client: tp.Optional[AlpacaClientT] = None,
        client_type: tp.Optional[str] = None,
        **client_config,
    ) -> AlpacaClientT:
        """Resolve the client.

        If provided, must be of the type `alpaca.data.historical.CryptoHistoricalDataClient`
        for `client_type="crypto"` and `alpaca.data.historical.StockHistoricalDataClient` for
        `client_type="stocks"`. Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient

        client = cls.resolve_custom_setting(client, "client")
        client_type = cls.resolve_custom_setting(client_type, "client_type")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            if client_type == "crypto":
                arg_names = get_func_arg_names(CryptoHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = CryptoHistoricalDataClient(**client_config)
            elif client_type == "stocks":
                arg_names = get_func_arg_names(StockHistoricalDataClient.__init__)
                client_config = {k: v for k, v in client_config.items() if k in arg_names}
                client = StockHistoricalDataClient(**client_config)
            else:
                raise ValueError(f"Invalid client type '{client_type}'")
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[AlpacaClientT] = None,
        client_type: tp.Optional[str] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        adjustment: tp.Optional[str] = None,
        feed: tp.Optional[str] = None,
        limit: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Alpaca.

        Args:
            symbol (str): Symbol.
            client (alpaca.common.rest.RESTClient): Client.

                See `AlpacaData.resolve_client`.
            client_type (str): Client type.

                See `AlpacaData.resolve_client`.
            client_config (dict): Client config.

                See `AlpacaData.resolve_client`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjustment (str): Specifies the corporate action adjustment for the returned bars.

                Options are: "raw", "split", "dividend" or "all". Default is "raw".
            feed (str): The feed to pull market data from.

                This is either "iex", "otc", or "sip". Feeds "sip" and "otc" are only available to
                those with a subscription. Default is "iex" for free plans and "sip" for paid.
            limit (int): The maximum number of returned items.

        For defaults, see `custom.alpaca` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("alpaca")
        from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
        from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, client_type=client_type, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        adjustment = cls.resolve_custom_setting(adjustment, "adjustment")
        feed = cls.resolve_custom_setting(feed, "feed")
        limit = cls.resolve_custom_setting(limit, "limit")

        freq = timeframe
        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "m":
            unit = TimeFrameUnit.Minute
        elif unit == "h":
            unit = TimeFrameUnit.Hour
        elif unit == "D":
            unit = TimeFrameUnit.Day
        elif unit == "W":
            unit = TimeFrameUnit.Week
        elif unit == "M":
            unit = TimeFrameUnit.Month
        else:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        timeframe = TimeFrame(multiplier, unit)

        if start is not None:
            start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
            start_str = start.replace(tzinfo=None).isoformat("T")
        else:
            start_str = None
        if end is not None:
            end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
            end_str = end.replace(tzinfo=None).isoformat("T")
        else:
            end_str = None

        if isinstance(client, CryptoHistoricalDataClient):
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
            )
            df = client.get_crypto_bars(request).df
        elif isinstance(client, StockHistoricalDataClient):
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_str,
                end=end_str,
                limit=limit,
                adjustment=adjustment,
                feed=feed,
            )
            df = client.get_stock_bars(request).df
        else:
            raise TypeError(f"Invalid client of type {type(client)}")

        df = df.droplevel("symbol", axis=0)
        df.index = df.index.rename("Open time")
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "trade_count": "Trade count",
                "vwap": "VWAP",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")

        if "Open" in df.columns:
            df["Open"] = df["Open"].astype(float)
        if "High" in df.columns:
            df["High"] = df["High"].astype(float)
        if "Low" in df.columns:
            df["Low"] = df["Low"].astype(float)
        if "Close" in df.columns:
            df["Close"] = df["Close"].astype(float)
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].astype(float)
        if "Trade count" in df.columns:
            df["Trade count"] = df["Trade count"].astype(int, errors="ignore")
        if "VWAP" in df.columns:
            df["VWAP"] = df["VWAP"].astype(float)

        if not df.empty:
            if start is not None:
                start = dt.to_timestamp(start, tz=df.index.tz)
                if df.index[0] < start:
                    df = df[df.index >= start]
            if end is not None:
                end = dt.to_timestamp(end, tz=df.index.tz)
                if df.index[-1] >= end:
                    df = df[df.index < end]
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
