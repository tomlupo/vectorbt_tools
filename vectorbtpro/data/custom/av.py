# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `AVData`."""

import re
import urllib.parse
import warnings
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.module_ import check_installed
from vectorbtpro.utils.parsing import get_func_arg_names

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from alpha_vantage.alphavantage import AlphaVantage as AlphaVantageT
except ImportError:
    AlphaVantageT = tp.Any

__all__ = [
    "AVData",
]

__pdoc__ = {}

AVDataT = tp.TypeVar("AVDataT", bound="AVData")


class AVData(RemoteData):
    """Data class for fetching from Alpha Vantage.

    See https://www.alphavantage.co/documentation/ for API.

    Apart of using https://github.com/RomelTorres/alpha_vantage package, this class can also
    parse the API documentation with `AVData.parse_api_meta` using `BeautifulSoup4` and build
    the API query based on this metadata (pass `use_parser=True`).

    This approach is the most flexible we can get since we can instantly react to Alpha Vantage's changes
    in the API. If the data provider changes its API documentation, you can always adapt the parsing
    procedure by overriding `AVData.parse_api_meta`.

    If parser still fails, you can disable parsing entirely and specify all information manually
    by setting `function` and disabling `match_params`

    See `AVData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.AVData.set_custom_settings(
        ...     apikey="YOUR_KEY"
        ... )
        ```

        * Pull data:

        ```pycon
        >>> data = vbt.AVData.pull(
        ...     "GOOGL",
        ...     timeframe="1 day",
        ... )

        >>> data = vbt.AVData.pull(
        ...     "BTC_USD",
        ...     timeframe="30 minutes",  # premium?
        ...     category="digital-currency",
        ...     outputsize="full"
        ... )

        >>> data = vbt.AVData.pull(
        ...     "REAL_GDP",
        ...     category="economic-indicators"
        ... )

        >>> data = vbt.AVData.pull(
        ...     "IBM",
        ...     category="technical-indicators",
        ...     function="STOCHRSI",
        ...     params=dict(fastkperiod=14)
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.av")

    @classmethod
    def list_symbols(cls, keywords: str, apikey: tp.Optional[str] = None, sort: bool = True) -> tp.List[str]:
        """List all symbols."""
        apikey = cls.resolve_custom_setting(apikey, "apikey")

        query = dict()
        query["function"] = "SYMBOL_SEARCH"
        query["keywords"] = keywords
        query["datatype"] = "csv"
        query["apikey"] = apikey
        url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(query)
        df = pd.read_csv(url)

        if sort:
            return sorted(dict.fromkeys(df["symbol"].tolist()))
        return list(dict.fromkeys(df["symbol"].tolist()))

    @classmethod
    @lru_cache()
    def parse_api_meta(cls) -> dict:
        """Parse API metadata from the documentation at https://www.alphavantage.co/documentation

        Cached class method. To avoid re-parsing the same metadata in different runtimes, save it manually."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("bs4")

        from bs4 import BeautifulSoup

        page = requests.get("https://www.alphavantage.co/documentation")
        soup = BeautifulSoup(page.content, "html.parser")
        api_meta = {}
        for section in soup.select("article section"):
            category = {}
            function = None
            function_args = dict(req_args=set(), opt_args=set())
            for tag in section.find_all(True):
                if tag.name == "h6":
                    if function is not None and tag.select("b")[0].getText().strip() == "API Parameters":
                        category[function] = function_args
                        function = None
                        function_args = dict(req_args=set(), opt_args=set())
                if tag.name == "b":
                    b_text = tag.getText().strip()
                    if b_text.startswith("❚ Required"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["req_args"].add(arg)
                if tag.name == "p":
                    p_text = tag.getText().strip()
                    if p_text.startswith("❚ Optional"):
                        arg = tag.select("code")[0].getText().strip()
                        function_args["opt_args"].add(arg)
                if tag.name == "code":
                    code_text = tag.getText().strip()
                    if code_text.startswith("function="):
                        function = code_text.replace("function=", "")
            if function is not None:
                category[function] = function_args
            api_meta[section.select("h2")[0]["id"]] = category

        return api_meta

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        use_parser: tp.Optional[bool] = None,
        apikey: tp.Optional[str] = None,
        api_meta: tp.Optional[dict] = None,
        category: tp.Union[None, str, AlphaVantageT, tp.Type[AlphaVantageT]] = None,
        function: tp.Union[None, str, tp.Callable] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        adjusted: tp.Optional[bool] = None,
        extended: tp.Optional[bool] = None,
        slice: tp.Optional[str] = None,
        series_type: tp.Optional[str] = None,
        time_period: tp.Optional[int] = None,
        outputsize: tp.Optional[str] = None,
        match_params: tp.Optional[bool] = None,
        params: tp.KwargsLike = None,
        read_csv_kwargs: tp.KwargsLike = None,
        silence_warnings: tp.Optional[bool] = None,
    ) -> tp.SymbolData:
        """Fetch a symbol from Alpha Vantage.

        If `use_parser` is False, or None and `alpha_vantage` is installed, uses the package.
        Otherwise, parses the API documentation and pulls data directly.

        See https://www.alphavantage.co/documentation/ for API endpoints and their parameters.

        !!! note
            Supports the CSV format only.

        Args:
            symbol (str): Symbol.

                May combine symbol/from_currency and market/to_currency using an underscore.
            use_parser (bool): Whether to use the parser instead of the `alpha_vantage` package.
            apikey (str): API key.
            api_meta (dict): API meta.

                If None, will use `AVData.parse_api_meta` if `function` is not provided
                or `match_params` is True.
            category (str or AlphaVantage): API category of your choice.

                Used if `function` is not provided or `match_params` is True.

                Supported are:

                * `alpha_vantage.alphavantage.AlphaVantage` instance, class, or class name
                * "time-series-data" or "time-series"
                * "fundamental-data" or "fundamentals"
                * "foreign-exchange", "forex", or "fx"
                * "digital-currency", "cryptocurrencies", "cryptocurrency", or "crypto"
                * "commodities"
                * "economic-indicators"
                * "technical-indicators" or "indicators"
            function (str or callable): API function of your choice.

                If None, will try to resolve it based on other arguments, such as `timeframe`,
                `adjusted`, and `extended`. Required for technical indicators, economic indicators,
                and fundamental data.

                See the keys in sub-dictionaries returned by `AVData.parse_api_meta`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".

                For time series, forex, and crypto, looks for interval type in the function's name.
                Defaults to "60min" if extended, otherwise to "daily".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            adjusted (bool): Whether to return time series adjusted by historical split and dividend events.
            extended (bool): Whether to return historical intraday time series for the trailing 2 years.
            slice (str): Slice of the trailing 2 years.
            series_type (str): The desired price type in the time series.
            time_period (int): Number of data points used to calculate each window value.
            outputsize (str): Output size.

                Supported are

                * "compact" that returns only the latest 100 data points
                * "full" that returns the full-length time series
            match_params (bool): Whether to match parameters with the ones required by the endpoint.

                Otherwise, uses only (resolved) `function`, `apikey`, `datatype="csv"`, and `params`.
            params: Additional keyword arguments passed as key/value pairs in the URL.
            read_csv_kwargs (dict): Keyword arguments passed to `pd.read_csv`.
            silence_warnings (bool): Whether to silence all warnings.

        For defaults, see `custom.av` in `vectorbtpro._settings.data`.
        """
        use_parser = cls.resolve_custom_setting(use_parser, "use_parser")
        apikey = cls.resolve_custom_setting(apikey, "apikey")
        api_meta = cls.resolve_custom_setting(api_meta, "api_meta")
        category = cls.resolve_custom_setting(category, "category")
        function = cls.resolve_custom_setting(function, "function")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        adjusted = cls.resolve_custom_setting(adjusted, "adjusted")
        extended = cls.resolve_custom_setting(extended, "extended")
        slice = cls.resolve_custom_setting(slice, "slice")
        series_type = cls.resolve_custom_setting(series_type, "series_type")
        time_period = cls.resolve_custom_setting(time_period, "time_period")
        outputsize = cls.resolve_custom_setting(outputsize, "outputsize")
        read_csv_kwargs = cls.resolve_custom_setting(read_csv_kwargs, "read_csv_kwargs", merge=True)
        match_params = cls.resolve_custom_setting(match_params, "match_params")
        params = cls.resolve_custom_setting(params, "params", merge=True)
        silence_warnings = cls.resolve_custom_setting(silence_warnings, "silence_warnings")

        if use_parser is None:
            if api_meta is None and check_installed("alpha_vantage"):
                use_parser = False
            else:
                use_parser = True
        if not use_parser:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("alpha_vantage")

        if use_parser and api_meta is None and (function is None or match_params):
            if not silence_warnings and cls.parse_api_meta.cache_info().misses == 0:
                warnings.warn("Parsing API documentation...", stacklevel=2)
            try:
                api_meta = cls.parse_api_meta()
            except Exception as e:
                raise ValueError("Can't fetch/parse the API documentation. Specify function and disable match_params.")

        freq = timeframe
        interval = None
        interval_type = None
        if timeframe is not None:
            if not isinstance(timeframe, str):
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            split = dt.split_freq_str(timeframe)
            if split is None:
                raise ValueError(f"Invalid timeframe '{timeframe}'")
            multiplier, unit = split
            if unit == "m":
                interval = str(multiplier) + "min"
                interval_type = "intraday"
            elif unit == "h":
                interval = str(60 * multiplier) + "min"
                interval_type = "intraday"
            elif unit == "D":
                interval = "daily"
                interval_type = "daily"
            elif unit == "W":
                interval = "weekly"
                interval_type = "weekly"
            elif unit == "M":
                interval = "monthly"
                interval_type = "monthly"
            elif unit == "Q":
                interval = "quarterly"
                interval_type = "quarterly"
            elif unit == "Y":
                interval = "annual"
                interval_type = "annual"
            if interval is None and multiplier > 1:
                raise ValueError("Multipliers are supported only for intraday timeframes")
        else:
            if extended:
                interval_type = "intraday"
                interval = "60min"
            else:
                interval_type = "daily"
                interval = "daily"

        if category is not None:
            if isinstance(category, str):
                if category.lower() in ("time-series-data", "time-series", "timeseries"):
                    if use_parser:
                        category = "time-series-data"
                    else:
                        from alpha_vantage.timeseries import TimeSeries

                        category = TimeSeries
                elif category.lower() in ("fundamentals", "fundamental-data", "fundamentaldata"):
                    if use_parser:
                        category = "fundamentals"
                    else:
                        from alpha_vantage.fundamentaldata import FundamentalData

                        category = FundamentalData
                elif category.lower() in ("fx", "forex", "foreign-exchange", "foreignexchange"):
                    if use_parser:
                        category = "fx"
                    else:
                        from alpha_vantage.foreignexchange import ForeignExchange

                        category = ForeignExchange
                elif category.lower() in ("digital-currency", "cryptocurrencies", "cryptocurrency", "crypto"):
                    if use_parser:
                        category = "digital-currency"
                    else:
                        from alpha_vantage.cryptocurrencies import CryptoCurrencies

                        category = CryptoCurrencies
                elif category.lower() in ("commodities",):
                    if use_parser:
                        category = "commodities"
                    else:
                        raise NotImplementedError(f"Category '{category}' not supported by alpha_vantage. Use parser.")
                elif category.lower() in ("economic-indicators",):
                    if use_parser:
                        category = "economic-indicators"
                    else:
                        raise NotImplementedError(f"Category '{category}' not supported by alpha_vantage. Use parser.")
                elif category.lower() in ("technical-indicators", "techindicators", "indicators"):
                    if use_parser:
                        category = "technical-indicators"
                    else:
                        from alpha_vantage.techindicators import TechIndicators

                        category = TechIndicators
                else:
                    raise ValueError(f"Invalid option category='{category}'")
            else:
                if use_parser:
                    raise TypeError("Category must be a string")
                else:
                    from alpha_vantage.alphavantage import AlphaVantage

                    if isinstance(category, type):
                        if not issubclass(category, AlphaVantage):
                            raise TypeError("Category must be a subclass of AlphaVantage")
                    elif not isinstance(category, AlphaVantage):
                        raise TypeError("Category must be an instance of AlphaVantage")

        if use_parser:
            if function is None:
                if category is not None:
                    if category in ("commodities", "economic-indicators"):
                        function = symbol
            if function is None:
                if category is None:
                    category = "time-series-data"
                if category in ("fundamentals", "technical-indicators"):
                    raise ValueError("Function is required")
                adjusted_in_functions = False
                extended_in_functions = False
                matched_functions = []
                for k in api_meta[category]:
                    if interval_type is None or interval_type.upper() in k:
                        if "ADJUSTED" in k:
                            adjusted_in_functions = True
                        if "EXTENDED" in k:
                            extended_in_functions = True
                        matched_functions.append(k)

                if adjusted_in_functions:
                    matched_functions = [
                        k
                        for k in matched_functions
                        if (adjusted and "ADJUSTED" in k) or (not adjusted and "ADJUSTED" not in k)
                    ]
                if extended_in_functions:
                    matched_functions = [
                        k
                        for k in matched_functions
                        if (extended and "EXTENDED" in k) or (not extended and "EXTENDED" not in k)
                    ]
                if len(matched_functions) == 0:
                    raise ValueError("No functions satisfy the requirements")
                if len(matched_functions) > 1:
                    raise ValueError("More than one function satisfies the requirements")
                function = matched_functions[0]

            if match_params:
                if function is not None and category is None:
                    category = None
                    for k, v in api_meta.items():
                        if function in v:
                            category = k
                            break
                if category is None:
                    raise ValueError("Category is required")
                req_args = api_meta[category][function]["req_args"]
                opt_args = api_meta[category][function]["opt_args"]
                args = set(req_args) | set(opt_args)

                matched_params = dict()
                matched_params["function"] = function
                matched_params["datatype"] = "csv"
                matched_params["apikey"] = apikey
                if "symbol" in args and "market" in args:
                    matched_params["symbol"] = symbol.split("_")[0]
                    matched_params["market"] = symbol.split("_")[1]
                elif "from_" in args and "to_currency" in args:
                    matched_params["from_currency"] = symbol.split("_")[0]
                    matched_params["to_currency"] = symbol.split("_")[1]
                elif "from_currency" in args and "to_currency" in args:
                    matched_params["from_currency"] = symbol.split("_")[0]
                    matched_params["to_currency"] = symbol.split("_")[1]
                elif "symbol" in args:
                    matched_params["symbol"] = symbol
                if "interval" in args:
                    matched_params["interval"] = interval
                if "adjusted" in args:
                    matched_params["adjusted"] = adjusted
                if "extended" in args:
                    matched_params["extended"] = extended
                if "extended_hours" in args:
                    matched_params["extended_hours"] = extended
                if "slice" in args:
                    matched_params["slice"] = slice
                if "series_type" in args:
                    matched_params["series_type"] = series_type
                if "time_period" in args:
                    matched_params["time_period"] = time_period
                if "outputsize" in args:
                    matched_params["outputsize"] = outputsize
                for k, v in params.items():
                    if k in args:
                        matched_params[k] = v
                    else:
                        raise ValueError(f"Function '{function}' does not expect parameter '{k}'")
                for arg in req_args:
                    if arg not in matched_params:
                        raise ValueError(f"Function '{function}' requires parameter '{arg}'")
            else:
                matched_params = dict(params)
                matched_params["function"] = function
                matched_params["apikey"] = apikey
                matched_params["datatype"] = "csv"

            url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(matched_params)
            df = pd.read_csv(url, **read_csv_kwargs)
        else:
            from alpha_vantage.alphavantage import AlphaVantage
            from alpha_vantage.timeseries import TimeSeries
            from alpha_vantage.fundamentaldata import FundamentalData
            from alpha_vantage.foreignexchange import ForeignExchange
            from alpha_vantage.cryptocurrencies import CryptoCurrencies
            from alpha_vantage.techindicators import TechIndicators

            if isinstance(category, type) and issubclass(category, AlphaVantage):
                category = category(key=apikey, output_format="pandas")

            if function is None:
                if category is None:
                    category = TimeSeries(key=apikey, output_format="pandas")
                if isinstance(category, (TechIndicators, FundamentalData)):
                    raise ValueError("Function is required")
                adjusted_in_methods = False
                extended_in_methods = False
                matched_methods = []
                for k in dir(category):
                    if interval_type is None or interval_type in k:
                        if "adjusted" in k:
                            adjusted_in_methods = True
                        if "extended" in k:
                            extended_in_methods = True
                        matched_methods.append(k)

                if adjusted_in_methods:
                    matched_methods = [
                        k
                        for k in matched_methods
                        if (adjusted and "adjusted" in k) or (not adjusted and "adjusted" not in k)
                    ]
                if extended_in_methods:
                    matched_methods = [
                        k
                        for k in matched_methods
                        if (extended and "extended" in k) or (not extended and "extended" not in k)
                    ]
                if len(matched_methods) == 0:
                    raise ValueError("No methods satisfy the requirements")
                if len(matched_methods) > 1:
                    raise ValueError("More than one method satisfies the requirements")
                function = matched_methods[0]
            if isinstance(function, str):
                function = function.lower()
                if not function.startswith("get_"):
                    function = "get_" + function
                if category is not None:
                    function = getattr(category, function)
                else:
                    categories = [
                        TimeSeries,
                        FundamentalData,
                        ForeignExchange,
                        CryptoCurrencies,
                        TechIndicators,
                    ]
                    matched_methods = []
                    for category in categories:
                        if function in dir(category):
                            matched_methods.append(getattr(category, function))
                    if len(matched_methods) == 0:
                        raise ValueError("No methods satisfy the requirements")
                    if len(matched_methods) > 1:
                        raise ValueError("More than one method satisfies the requirements")
                    function = matched_methods[0]

            if match_params:
                args = set(get_func_arg_names(function))

                matched_params = dict()
                if "symbol" in args and "market" in args:
                    matched_params["symbol"] = symbol.split("_")[0]
                    matched_params["market"] = symbol.split("_")[1]
                elif "from_" in args and "to_currency" in args:
                    matched_params["from_currency"] = symbol.split("_")[0]
                    matched_params["to_currency"] = symbol.split("_")[1]
                elif "from_currency" in args and "to_currency" in args:
                    matched_params["from_currency"] = symbol.split("_")[0]
                    matched_params["to_currency"] = symbol.split("_")[1]
                elif "symbol" in args:
                    matched_params["symbol"] = symbol
                if "interval" in args:
                    matched_params["interval"] = interval
                if "adjusted" in args:
                    matched_params["adjusted"] = adjusted
                if "extended" in args:
                    matched_params["extended"] = extended
                if "extended_hours" in args:
                    matched_params["extended_hours"] = extended
                if "slice" in args:
                    matched_params["slice"] = slice
                if "series_type" in args:
                    matched_params["series_type"] = series_type
                if "time_period" in args:
                    matched_params["time_period"] = time_period
                if "outputsize" in args:
                    matched_params["outputsize"] = outputsize
            else:
                matched_params = dict(params)

            df, df_metadata = function(**matched_params)
            for k, v in df_metadata.items():
                if "Time Zone" in k:
                    if tz is None:
                        if v.endswith(" Time"):
                            v = v[: -len(" Time")]
                        tz = v

        df.index.name = None
        new_columns = []
        for c in df.columns:
            new_c = re.sub(r"^\d+\w*\.\s*", "", c)
            new_c = new_c[0].title() + new_c[1:]
            if new_c.endswith(" (USD)"):
                new_c = new_c[: -len(" (USD)")]
            new_columns.append(new_c)
        df = df.rename(columns=dict(zip(df.columns, new_columns)))
        df = df.loc[:, ~df.columns.duplicated()]
        for c in df.columns:
            if df[c].dtype == "O":
                df[c] = df[c].replace({".": np.nan})
        df = df.apply(pd.to_numeric, errors="ignore")
        if not df.empty and df.index[0] > df.index[1]:
            df = df.iloc[::-1]
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None and tz is not None:
            df = df.tz_localize(tz)

        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
