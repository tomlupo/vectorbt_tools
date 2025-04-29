# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `FinPyData`."""

import pandas as pd

from itertools import product

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from findatapy.market import Market as MarketT
    from findatapy.util import ConfigManager as ConfigManagerT
except ImportError:
    MarketT = tp.Any
    ConfigManagerT = tp.Any

__all__ = [
    "FinPyData",
]

FinPyDataT = tp.TypeVar("FinPyDataT", bound="FinPyData")


class FinPyData(RemoteData):
    """Data class for fetching using findatapy.

    See https://github.com/cuemacro/findatapy for API.

    See `FinPyData.fetch_symbol` for arguments.

    Usage:
        * Pull data (keyword argument format):

        ```pycon
        >>> data = vbt.FinPyData.pull(
        ...     "EURUSD",
        ...     start="14 June 2016",
        ...     end="15 June 2016",
        ...     timeframe="tick",
        ...     category="fx",
        ...     fields=["bid", "ask"],
        ...     data_source="dukascopy"
        ... )
        ```

        * Pull data (string format):

        ```pycon
        >>> data = vbt.FinPyData.pull(
        ...     "fx.dukascopy.tick.NYC.EURUSD.bid,ask",
        ...     start="14 June 2016",
        ...     end="15 June 2016",
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.finpy")

    @classmethod
    def resolve_market(
        cls,
        market: tp.Optional[MarketT] = None,
        **market_config,
    ) -> MarketT:
        """Resolve the market.

        If provided, must be of the type `findatapy.market.market.Market`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.market import Market, MarketDataGenerator

        market = cls.resolve_custom_setting(market, "market")
        if market_config is None:
            market_config = {}
        has_market_config = len(market_config) > 0
        market_config = cls.resolve_custom_setting(market_config, "market_config", merge=True)
        if "market_data_generator" not in market_config:
            market_config["market_data_generator"] = MarketDataGenerator()
        if market is None:
            market = Market(**market_config)
        elif has_market_config:
            raise ValueError("Cannot apply market_config on already created market")
        return market

    @classmethod
    def resolve_config_manager(
        cls,
        config_manager: tp.Optional[ConfigManagerT] = None,
        **config_manager_config,
    ) -> MarketT:
        """Resolve the config manager.

        If provided, must be of the type `findatapy.util.ConfigManager`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.util import ConfigManager

        config_manager = cls.resolve_custom_setting(config_manager, "config_manager")
        if config_manager_config is None:
            config_manager_config = {}
        has_config_manager_config = len(config_manager_config) > 0
        config_manager_config = cls.resolve_custom_setting(config_manager_config, "config_manager_config", merge=True)
        if config_manager is None:
            config_manager = ConfigManager().get_instance(**config_manager_config)
        elif has_config_manager_config:
            raise ValueError("Cannot apply config_manager_config on already created config_manager")
        return config_manager

    @classmethod
    def list_symbols(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        config_manager: tp.Optional[ConfigManagerT] = None,
        config_manager_config: tp.KwargsLike = None,
        category: tp.Optional[tp.MaybeList[str]] = None,
        data_source: tp.Optional[tp.MaybeList[str]] = None,
        freq: tp.Optional[tp.MaybeList[str]] = None,
        cut: tp.Optional[tp.MaybeList[str]] = None,
        tickers: tp.Optional[tp.MaybeList[str]] = None,
        dict_filter: tp.DictLike = None,
        smart_group: bool = False,
        return_fields: tp.Optional[tp.MaybeList[str]] = None,
        combine_parts: bool = True,
    ) -> tp.List[str]:
        """List all symbols.

        Passes most arguments to `findatapy.util.ConfigManager.free_form_tickers_regex_query`.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`."""
        if config_manager_config is None:
            config_manager_config = {}
        config_manager = cls.resolve_config_manager(config_manager=config_manager, **config_manager_config)
        if dict_filter is None:
            dict_filter = {}
        def_ret_fields = ["category", "data_source", "freq", "cut", "tickers"]
        if return_fields is None:
            ret_fields = def_ret_fields
        elif isinstance(return_fields, str):
            if return_fields.lower() == "all":
                ret_fields = def_ret_fields + ["fields"]
            else:
                ret_fields = [return_fields]
        else:
            ret_fields = return_fields

        df = config_manager.free_form_tickers_regex_query(
            category=category,
            data_source=data_source,
            freq=freq,
            cut=cut,
            tickers=tickers,
            dict_filter=dict_filter,
            smart_group=smart_group,
            ret_fields=ret_fields,
        )
        all_symbols = []
        for _, row in df.iterrows():
            parts = []
            if "category" in row.index:
                parts.append(row.loc["category"])
            if "data_source" in row.index:
                parts.append(row.loc["data_source"])
            if "freq" in row.index:
                parts.append(row.loc["freq"])
            if "cut" in row.index:
                parts.append(row.loc["cut"])
            if "tickers" in row.index:
                parts.append(row.loc["tickers"])
            if "fields" in row.index:
                parts.append(row.loc["fields"])
            if combine_parts:
                split_parts = [part.split(',') for part in parts]
                combinations = list(product(*split_parts))
            else:
                combinations = [parts]
            for symbol in ['.'.join(combination) for combination in combinations]:
                if pattern is not None:
                    if not cls.key_match(symbol, pattern, use_regex=use_regex):
                        continue
                all_symbols.append(symbol)

        if sort:
            return sorted(dict.fromkeys(all_symbols))
        return list(dict.fromkeys(all_symbols))

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        market: tp.Optional[MarketT] = None,
        market_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        **request_kwargs,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from findatapy.

        Args:
            symbol (str): Symbol.

                Also accepts the format such as "fx.bloomberg.daily.NYC.EURUSD.close".
                The fields `freq`, `cut`, `tickers`, and `fields` here are optional.
            market (findatapy.market.market.Market): Market.

                See `FinPyData.resolve_market`.
            market_config (dict): Client config.

                See `FinPyData.resolve_market`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            **request_kwargs: Other keyword arguments passed to `findatapy.market.marketdatarequest.MarketDataRequest`.

        For defaults, see `custom.finpy` in `vectorbtpro._settings.data`.
        Global settings can be provided per exchange id using the `exchanges` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("findatapy")
        from findatapy.market import MarketDataRequest

        if market_config is None:
            market_config = {}
        market = cls.resolve_market(market=market, **market_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        request_kwargs = cls.resolve_custom_setting(request_kwargs, "request_kwargs", merge=True)

        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split

        if unit == "s":
            unit = "second"
            freq = timeframe
        elif unit == "m":
            unit = "minute"
            freq = timeframe
        elif unit == "h":
            unit = "hourly"
            freq = timeframe
        elif unit == "D":
            unit = "daily"
            freq = timeframe
        elif unit == "W":
            unit = "weekly"
            freq = timeframe
        elif unit == "M":
            unit = "monthly"
            freq = timeframe
        elif unit == "Q":
            unit = "quarterly"
            freq = timeframe
        elif unit == "Y":
            unit = "annually"
            freq = timeframe
        else:
            freq = None
        if "resample" in request_kwargs:
            freq = request_kwargs["resample"]

        if start is not None:
            start = dt.to_naive_datetime(dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc"))
        if end is not None:
            end = dt.to_naive_datetime(dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc"))

        if "md_request" in request_kwargs:
            md_request = request_kwargs["md_request"]
        elif "md_request_df" in request_kwargs:
            md_request = market.create_md_request_from_dataframe(
                md_request_df=request_kwargs["md_request_df"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif "md_request_str" in request_kwargs:
            md_request = market.create_md_request_from_str(
                md_request_str=request_kwargs["md_request_str"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif "md_request_dict" in request_kwargs:
            md_request = market.create_md_request_from_dict(
                md_request_dict=request_kwargs["md_request_dict"],
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        elif symbol.count(".") >= 2:
            md_request = market.create_md_request_from_str(
                md_request_str=symbol,
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )
        else:
            md_request = MarketDataRequest(
                tickers=symbol,
                start_date=start,
                finish_date=end,
                freq_mult=multiplier,
                freq=unit,
                **request_kwargs,
            )

        df = market.fetch_market(md_request=md_request)
        if df is None:
            return None
        if isinstance(md_request.tickers, str):
            ticker = md_request.tickers
        elif len(md_request.tickers) == 1:
            ticker = md_request.tickers[0]
        else:
            ticker = None
        if ticker is not None:
            df.columns = df.columns.map(lambda x: x.replace(ticker + ".", ""))
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
