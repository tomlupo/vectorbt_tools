# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `TVData`."""

import datetime
import json
import math
import random
import re
import string
import time

import pandas as pd
import requests
from websocket import WebSocket

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts, Configured
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate

__all__ = [
    "TVClient",
    "TVData",
]

SIGNIN_URL = "https://www.tradingview.com/accounts/signin/"
"""Sign-in URL."""

SEARCH_URL = (
    "https://symbol-search.tradingview.com/symbol_search/v3/?"
    "text={text}&"
    "start={start}&"
    "hl=1&"
    "exchange={exchange}&"
    "lang=en&"
    "search_type=undefined&"
    "domain=production&"
    "sort_by_country=US"
)
"""Symbol search URL."""

SCAN_URL = "https://scanner.tradingview.com/{market}/scan"
"""Market scanner URL."""

ORIGIN_URL = "https://data.tradingview.com"
"""Origin URL."""

REFERER_URL = "https://www.tradingview.com"
"""Referer URL."""

WS_URL = "wss://data.tradingview.com/socket.io/websocket"
"""Websocket URL."""

PRO_WS_URL = "wss://prodata.tradingview.com/socket.io/websocket"
"""Websocket URL (Pro)."""

WS_TIMEOUT = 5
"""Websocket timeout."""

MARKET_LIST = [
    "america",
    "argentina",
    "australia",
    "austria",
    "bahrain",
    "bangladesh",
    "belgium",
    "brazil",
    "canada",
    "chile",
    "china",
    "colombia",
    "cyprus",
    "czech",
    "denmark",
    "egypt",
    "estonia",
    "euronext",
    "finland",
    "france",
    "germany",
    "greece",
    "hongkong",
    "hungary",
    "iceland",
    "india",
    "indonesia",
    "israel",
    "italy",
    "japan",
    "kenya",
    "korea",
    "ksa",
    "kuwait",
    "latvia",
    "lithuania",
    "luxembourg",
    "malaysia",
    "mexico",
    "morocco",
    "netherlands",
    "newzealand",
    "nigeria",
    "norway",
    "pakistan",
    "peru",
    "philippines",
    "poland",
    "portugal",
    "qatar",
    "romania",
    "rsa",
    "russia",
    "serbia",
    "singapore",
    "slovakia",
    "spain",
    "srilanka",
    "sweden",
    "switzerland",
    "taiwan",
    "thailand",
    "tunisia",
    "turkey",
    "uae",
    "uk",
    "venezuela",
    "vietnam",
]
"""List of markets supported by the market scanner (list may be incomplete)."""

FIELD_LIST = [
    "name",
    "description",
    "logoid",
    "update_mode",
    "type",
    "typespecs",
    "close",
    "pricescale",
    "minmov",
    "fractional",
    "minmove2",
    "currency",
    "change",
    "change_abs",
    "Recommend.All",
    "volume",
    "Value.Traded",
    "market_cap_basic",
    "fundamental_currency_code",
    "Perf.1Y.MarketCap",
    "price_earnings_ttm",
    "earnings_per_share_basic_ttm",
    "number_of_employees_fy",
    "sector",
    "market",
]
"""List of fields supported by the market scanner (list may be incomplete)."""

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
"""User agent."""


class TVClient(Configured):
    """Client for TradingView."""

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "username",
        "password",
        "auth_token",
    }

    def __init__(
        self,
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
        auth_token: tp.Optional[str] = None,
        **kwargs,
    ) -> None:
        """Client for TradingView."""
        Configured.__init__(
            self,
            username=username,
            password=password,
            auth_token=auth_token,
            **kwargs,
        )

        if auth_token is None:
            auth_token = self.auth(username, password)
        elif username is not None or password is not None:
            raise ValueError("Must provide either username and password, or auth_token")

        self._auth_token = auth_token
        self._ws = None
        self._session = self.generate_session()
        self._chart_session = self.generate_chart_session()

    @property
    def auth_token(self) -> str:
        """Authentication token."""
        return self._auth_token

    @property
    def ws(self) -> WebSocket:
        """Instance of `websocket.Websocket`."""
        return self._ws

    @property
    def session(self) -> str:
        """Session."""
        return self._session

    @property
    def chart_session(self) -> str:
        """Chart session."""
        return self._chart_session

    @classmethod
    def auth(
        cls,
        username: tp.Optional[str] = None,
        password: tp.Optional[str] = None,
    ) -> str:
        """Authenticate."""
        if username is not None and password is not None:
            data = {"username": username, "password": password, "remember": "on"}
            headers = {"Referer": REFERER_URL, "User-Agent": USER_AGENT}
            response = requests.post(url=SIGNIN_URL, data=data, headers=headers)
            response.raise_for_status()
            json = response.json()
            if "user" not in json or "auth_token" not in json["user"]:
                raise ValueError(json)
            return json["user"]["auth_token"]
        if username is not None or password is not None:
            raise ValueError("Must provide both username and password")
        return "unauthorized_user_token"

    @classmethod
    def generate_session(cls) -> str:
        """Generate session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "qs_" + random_string

    @classmethod
    def generate_chart_session(cls) -> str:
        """Generate chart session."""
        stringLength = 12
        letters = string.ascii_lowercase
        random_string = "".join(random.choice(letters) for _ in range(stringLength))
        return "cs_" + random_string

    def create_connection(self, pro_data: bool = True) -> None:
        """Create a websocket connection."""
        from websocket import create_connection

        if pro_data:
            self._ws = create_connection(
                PRO_WS_URL,
                headers=json.dumps({"Origin": ORIGIN_URL}),
                timeout=WS_TIMEOUT,
            )
        else:
            self._ws = create_connection(
                WS_URL,
                headers=json.dumps({"Origin": ORIGIN_URL}),
                timeout=WS_TIMEOUT,
            )

    @classmethod
    def filter_raw_message(cls, text) -> tp.Tuple[str, str]:
        """Filter raw message."""
        found = re.search('"m":"(.+?)",', text).group(1)
        found2 = re.search('"p":(.+?"}"])}', text).group(1)
        return found, found2

    @classmethod
    def prepend_header(cls, st: str) -> str:
        """Prepend a header."""
        return "~m~" + str(len(st)) + "~m~" + st

    @classmethod
    def construct_message(cls, func: str, param_list: tp.List[str]) -> str:
        """Construct a message."""
        return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

    def create_message(self, func: str, param_list: tp.List[str]) -> str:
        """Create a message."""
        return self.prepend_header(self.construct_message(func, param_list))

    def send_message(self, func: str, param_list: tp.List[str]) -> None:
        """Send a message."""
        m = self.create_message(func, param_list)
        self.ws.send(m)

    @classmethod
    def convert_raw_data(cls, raw_data: str, symbol: str) -> pd.DataFrame:
        """Process raw data into a DataFrame."""
        search_result = re.search(r'"s":\[(.+?)\}\]', raw_data)
        if search_result is None:
            raise ValueError("Couldn't parse data returned by TradingView: {}".format(raw_data))
        out = search_result.group(1)
        x = out.split(',{"')
        data = list()
        volume_data = True
        for xi in x:
            xi = re.split(r"\[|:|,|\]", xi)
            ts = datetime.datetime.utcfromtimestamp(float(xi[4]))
            row = [ts]
            for i in range(5, 10):
                # skip converting volume data if does not exists
                if not volume_data and i == 9:
                    row.append(0.0)
                    continue
                try:
                    row.append(float(xi[i]))
                except ValueError:
                    volume_data = False
                    row.append(0.0)
            data.append(row)
        data = pd.DataFrame(data, columns=["datetime", "open", "high", "low", "close", "volume"])
        data = data.set_index("datetime")
        data.insert(0, "symbol", value=symbol)
        return data

    @classmethod
    def format_symbol(cls, symbol: str, exchange: str, fut_contract: tp.Optional[int] = None) -> str:
        """Format a symbol."""
        if ":" in symbol:
            pass
        elif fut_contract is None:
            symbol = f"{exchange}:{symbol}"
        elif isinstance(fut_contract, int):
            symbol = f"{exchange}:{symbol}{fut_contract}!"
        else:
            raise ValueError(f"Invalid option fut_contract='{fut_contract}'")
        return symbol

    def get_hist(
        self,
        symbol: str,
        exchange: str = "NSE",
        interval: str = "1D",
        fut_contract: tp.Optional[int] = None,
        adjustment: str = "splits",
        extended_session: bool = False,
        pro_data: bool = True,
        limit: int = 20000,
        return_raw: bool = False,
    ) -> tp.Union[str, tp.Frame]:
        """Get historical data."""
        symbol = self.format_symbol(symbol=symbol, exchange=exchange, fut_contract=fut_contract)

        backadjustment = False
        if symbol.endswith("!A"):
            backadjustment = True
            symbol = symbol.replace("!A", "!")

        self.create_connection(pro_data=pro_data)
        self.send_message("set_auth_token", [self.auth_token])
        self.send_message("chart_create_session", [self.chart_session, ""])
        self.send_message("quote_create_session", [self.session])
        self.send_message(
            "quote_set_fields",
            [
                self.session,
                "ch",
                "chp",
                "current_session",
                "description",
                "local_description",
                "language",
                "exchange",
                "fractional",
                "is_tradable",
                "lp",
                "lp_time",
                "minmov",
                "minmove2",
                "original_name",
                "pricescale",
                "pro_name",
                "short_name",
                "type",
                "update_mode",
                "volume",
                "currency_code",
                "rchp",
                "rtc",
            ],
        )
        self.send_message("quote_add_symbols", [self.session, symbol, {"flags": ["force_permission"]}])
        self.send_message("quote_fast_symbols", [self.session, symbol])
        self.send_message(
            "resolve_symbol",
            [
                self.chart_session,
                "symbol_1",
                '={"symbol":"'
                + symbol
                + '","adjustment":"'
                + adjustment
                + ('' if not backadjustment else '","backadjustment":"default')
                + '","session":'
                + ('"regular"' if not extended_session else '"extended"')
                + "}",
            ],
        )
        self.send_message("create_series", [self.chart_session, "s1", "s1", "symbol_1", interval, limit])
        self.send_message("switch_timezone", [self.chart_session, "exchange"])

        raw_data = ""
        while True:
            try:
                result = self.ws.recv()
                raw_data += result + "\n"
            except Exception as e:
                break
            if "series_completed" in result:
                break
        if return_raw:
            return raw_data
        return self.convert_raw_data(raw_data, symbol)

    @classmethod
    def search_symbol(
        cls,
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        pages: tp.Optional[int] = None,
        delay: tp.Optional[int] = None,
        retries: int = 3,
        show_progress: bool = True,
        pbar_kwargs: tp.KwargsLike = None,
    ) -> tp.List[dict]:
        """Search for a symbol."""
        if text is None:
            text = ""
        if exchange is None:
            exchange = ""
        if pbar_kwargs is None:
            pbar_kwargs = {}

        symbols_list = []
        pbar = None
        pages_fetched = 0
        while True:
            for i in range(retries):
                try:
                    url = SEARCH_URL.format(text=text, exchange=exchange.upper(), start=len(symbols_list))
                    headers = {"Referer": REFERER_URL, "Origin": ORIGIN_URL, "User-Agent": USER_AGENT}
                    resp = requests.get(url, headers=headers)
                    symbols_data = json.loads(resp.text.replace("</em>", "").replace("<em>", ""))
                    break
                except json.JSONDecodeError as e:
                    if i == retries - 1:
                        raise e
                    if delay is not None:
                        time.sleep(delay)
            symbols_remaining = symbols_data.get("symbols_remaining", 0)
            new_symbols = symbols_data.get("symbols", [])
            symbols_list.extend(new_symbols)
            if pages is None and symbols_remaining > 0:
                show_pbar = True
            elif pages is not None and pages > 1:
                show_pbar = True
            else:
                show_pbar = False
            if pbar is None and show_pbar:
                if pages is not None:
                    total = pages
                else:
                    total = math.ceil((len(new_symbols) + symbols_remaining) / len(new_symbols))
                pbar = ProgressBar(
                    total=total,
                    show_progress=show_progress,
                    **pbar_kwargs,
                )
                pbar.enter()
            if pbar is not None:
                max_symbols = len(symbols_list) + symbols_remaining
                if pages is not None:
                    max_symbols = min(max_symbols, pages * len(new_symbols))
                pbar.set_description(dict(symbols="%d/%d" % (len(symbols_list), max_symbols)))
                pbar.update()
            if symbols_remaining == 0:
                break
            pages_fetched += 1
            if pages is not None and pages_fetched >= pages:
                break
            if delay is not None:
                time.sleep(delay)
        if pbar is not None:
            pbar.exit()

        return symbols_list

    @classmethod
    def scan_symbols(cls, market: tp.Optional[str] = None, **kwargs) -> tp.List[dict]:
        """Scan symbols in a region/market."""
        if market is None:
            market = "global"
        url = SCAN_URL.format(market=market.lower())
        headers = {"Referer": REFERER_URL, "Origin": ORIGIN_URL, "User-Agent": USER_AGENT}
        resp = requests.post(url, json.dumps(kwargs), headers=headers)
        symbols_list = json.loads(resp.text)["data"]
        return symbols_list


TVDataT = tp.TypeVar("TVDataT", bound="TVData")


class TVData(RemoteData):
    """Data class for fetching from TradingView.

    See `TVData.fetch_symbol` for arguments.

    !!! note
        If you're getting the error "Please confirm that you are not a robot by clicking the captcha box."
        when attempting to authenticate, use `auth_token` instead of `username` and `password`.
        To get the authentication token, go to TradingView, log in, visit any chart, open your console's
        developer tools, and search for "auth_token".

    Usage:
        * Set up the credentials globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.TVData.set_custom_settings(
        ...     client_config=dict(
        ...         username="YOUR_USERNAME",
        ...         password="YOUR_PASSWORD",
        ...         auth_token="YOUR_AUTH_TOKEN",  # optional, instead of username and password
        ...     )
        ... )
        ```

        * Pull data:

        ```pycon
        >>> data = vbt.TVData.pull(
        ...     "NASDAQ:AAPL",
        ...     timeframe="1 hour"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.tv")

    @classmethod
    def list_symbols(
        cls,
        *,
        exchange_pattern: tp.Optional[str] = None,
        symbol_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        client: tp.Optional[TVClient] = None,
        client_config: tp.DictLike = None,
        text: tp.Optional[str] = None,
        exchange: tp.Optional[str] = None,
        pages: tp.Optional[int] = None,
        delay: tp.Optional[int] = None,
        retries: tp.Optional[int] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        market: tp.Optional[str] = None,
        markets: tp.Optional[tp.List[str]] = None,
        fields: tp.Optional[tp.MaybeIterable[str]] = None,
        filter_by: tp.Union[None, tp.Callable, CustomTemplate] = None,
        groups: tp.Optional[tp.MaybeIterable[tp.Dict[str, tp.MaybeIterable[str]]]] = None,
        template_context: tp.KwargsLike = None,
        return_field_data: bool = False,
        **scanner_kwargs,
    ) -> tp.Union[tp.List[str], tp.List[tp.Kwargs]]:
        """List all symbols.

        Uses symbol search when either `text` or `exchange` is provided (returns a subset of symbols).
        Otherwise, uses the market scanner (returns all symbols, big payload).

        When using the market scanner, use `market` to filter by one or multiple markets. For the list
        of available markets, see `MARKET_LIST`.

        Use `fields` to make the market scanner return additional information that can be used for
        filtering with `filter_by`. Such information is passed to the function as a dictionary where
        fields are keys. The function can also be a template that can use the same information provided
        as a context, or a list of values that should be matched against the values corresponding to their fields.
        For the list of available fields, see `FIELD_LIST`. Argument `fields` can also be "all".
        Set `return_field_data` to True to return a list with (filtered) field data.

        Use `groups` to provide a single dictionary or a list of dictionaries with groups.
        Each dictionary can be provided either in a compressed format, such as `dict(index=index)`,
        or in a full format, such as `dict(type="index", values=[index])`.

        Keyword arguments `scanner_kwargs` are encoded and passed directly to the market scanner.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each exchange against
        `exchange_pattern` and each symbol against `symbol_pattern`.

        Usage:
            * List all symbols (market scanner):

            ```pycon
            >>> from vectorbtpro import *

            >>> vbt.TVData.list_symbols()
            ```

            * Search for symbols matching a pattern (market scanner, client-side):

            ```pycon
            >>> vbt.TVData.list_symbols(symbol_pattern="BTC*")
            ```

            * Search for exchanges matching a pattern (market scanner, client-side):

            ```pycon
            >>> vbt.TVData.list_symbols(exchange_pattern="NASDAQ")
            ```

            * Search for symbols containing a text (symbol search, server-side):

            ```pycon
            >>> vbt.TVData.list_symbols(text="BTC")
            ```

            * List symbols from an exchange (symbol search):

            ```pycon
            >>> vbt.TVData.list_symbols(exchange="NASDAQ")
            ```

            * List symbols from a market (market scanner):

            ```pycon
            >>> vbt.TVData.list_symbols(market="poland")
            ```

            * List index constituents (market scanner):

            ```pycon
            >>> vbt.TVData.list_symbols(groups=dict(index="NASDAQ:NDX"))
            ```

            * Filter symbols by fields using a function (market scanner):

            ```pycon
            >>> vbt.TVData.list_symbols(
            ...     market="america",
            ...     fields=["sector"],
            ...     filter_by=lambda context: context["sector"] == "Technology Services"
            ... )
            ```

            * Filter symbols by fields using a template (market scanner):

            ```pycon
            >>> vbt.TVData.list_symbols(
            ...     market="america",
            ...     fields=["sector"],
            ...     filter_by=vbt.RepEval("sector == 'Technology Services'")
            ... )
            ```
        """
        pages = cls.resolve_custom_setting(pages, "pages", sub_path="search", sub_path_only=True)
        delay = cls.resolve_custom_setting(delay, "delay", sub_path="search", sub_path_only=True)
        retries = cls.resolve_custom_setting(retries, "retries", sub_path="search", sub_path_only=True)
        show_progress = cls.resolve_custom_setting(
            show_progress, "show_progress", sub_path="search", sub_path_only=True
        )
        pbar_kwargs = cls.resolve_custom_setting(
            pbar_kwargs, "pbar_kwargs", merge=True, sub_path="search", sub_path_only=True
        )
        markets = cls.resolve_custom_setting(markets, "markets", sub_path="scanner", sub_path_only=True)
        fields = cls.resolve_custom_setting(fields, "fields", sub_path="scanner", sub_path_only=True)
        filter_by = cls.resolve_custom_setting(filter_by, "filter_by", sub_path="scanner", sub_path_only=True)
        groups = cls.resolve_custom_setting(groups, "groups", sub_path="scanner", sub_path_only=True)
        template_context = cls.resolve_custom_setting(
            template_context, "template_context", merge=True, sub_path="scanner", sub_path_only=True
        )
        scanner_kwargs = cls.resolve_custom_setting(
            scanner_kwargs, "scanner_kwargs", merge=True, sub_path="scanner", sub_path_only=True
        )

        if market is None and text is None and exchange is None:
            market = "global"
        if market is not None and (text is not None or exchange is not None):
            raise ValueError("Please provide either market, or text and/or exchange")
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        if market is None:
            data = client.search_symbol(
                text=text,
                exchange=exchange,
                pages=pages,
                delay=delay,
                retries=retries,
                show_progress=show_progress,
                pbar_kwargs=pbar_kwargs,
            )
            all_symbols = map(lambda x: x["exchange"] + ":" + x["symbol"], data)
            return_field_data = False
        else:
            if markets is not None:
                scanner_kwargs["markets"] = markets
            if fields is not None:
                if "columns" in scanner_kwargs:
                    raise ValueError("Use fields instead of columns")
                if isinstance(fields, str):
                    if fields.lower() == "all":
                        fields = FIELD_LIST
                    else:
                        fields = [fields]
                scanner_kwargs["columns"] = fields
            if groups is not None:
                if isinstance(groups, dict):
                    groups = [groups]
                new_groups = []
                for group in groups:
                    if "type" in group:
                        new_groups.append(group)
                    else:
                        for k, v in group.items():
                            if isinstance(v, str):
                                v = [v]
                            new_groups.append(dict(type=k, values=v))
                groups = new_groups
                if "symbols" in scanner_kwargs:
                    scanner_kwargs["symbols"] = dict(scanner_kwargs["symbols"])
                else:
                    scanner_kwargs["symbols"] = dict()
                scanner_kwargs["symbols"]["groups"] = groups
            if filter_by is not None:
                if isinstance(filter_by, str):
                    filter_by = [filter_by]
            data = client.scan_symbols(market.lower(), **scanner_kwargs)
            if data is None:
                raise ValueError("No data returned by TradingView")
            all_symbols = []
            for item in data:
                if fields is not None:
                    item = {"symbol": item["s"], **dict(zip(fields, item["d"]))}
                else:
                    item = {"symbol": item["s"]}
                if filter_by is not None:
                    if fields is not None:
                        context = merge_dicts(item, template_context)
                    else:
                        raise ValueError("Must provide fields for filter_by")
                    if isinstance(filter_by, CustomTemplate):
                        if not filter_by.substitute(context, eval_id="filter_by"):
                            continue
                    elif callable(filter_by):
                        if not filter_by(context):
                            continue
                    else:
                        if len(fields) != len(filter_by):
                            raise ValueError("Fields and filter_by must have the same number of values")
                        conditions_met = True
                        for i in range(len(fields)):
                            if context[fields[i]] != filter_by[i]:
                                conditions_met = False
                                break
                        if not conditions_met:
                            continue
                if return_field_data:
                    all_symbols.append(item)
                else:
                    all_symbols.append(item["symbol"])
        found_symbols = []
        for symbol in all_symbols:
            if return_field_data:
                item = symbol
                symbol = item["symbol"]
            else:
                item = symbol
            if '"symbol"' in symbol:
                continue
            if exchange_pattern is not None:
                if not cls.key_match(symbol.split(":")[0], exchange_pattern, use_regex=use_regex):
                    continue
            if symbol_pattern is not None:
                if not cls.key_match(symbol.split(":")[1], symbol_pattern, use_regex=use_regex):
                    continue
            found_symbols.append(item)

        if sort:
            if return_field_data:
                return sorted(found_symbols, key=lambda x: x["symbol"])
            return sorted(dict.fromkeys(found_symbols))
        if return_field_data:
            return found_symbols
        return list(dict.fromkeys(found_symbols))

    @classmethod
    def resolve_client(cls, client: tp.Optional[TVClient] = None, **client_config) -> TVClient:
        """Resolve the client.

        If provided, must be of the type `TVClient`. Otherwise, will be created using `client_config`."""
        client = cls.resolve_custom_setting(client, "client")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            client = TVClient(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[TVClient] = None,
        client_config: tp.KwargsLike = None,
        exchange: tp.Optional[str] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        fut_contract: tp.Optional[int] = None,
        adjustment: tp.Optional[str] = None,
        extended_session: tp.Optional[bool] = None,
        pro_data: tp.Optional[bool] = None,
        limit: tp.Optional[int] = None,
        delay: tp.Optional[int] = None,
        retries: tp.Optional[int] = None,
    ) -> tp.SymbolData:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from TradingView.

        Args:
            symbol (str): Symbol.

                Symbol must be in the `EXCHANGE:SYMBOL` format if `exchange` is None.
            client (TVClient): Client.

                See `TVData.resolve_client`.
            client_config (dict): Client config.

                See `TVData.resolve_client`.
            exchange (str): Exchange.

                Can be omitted if already provided via `symbol`.
            timeframe (str): Timeframe.

                Allows human-readable strings such as "15 minutes".
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            fut_contract (int): None for cash, 1 for continuous current contract in front,
                2 for continuous next contract in front.
            adjustment (str): Adjustment.

                Either "splits" (default) or "dividends".
            extended_session (bool): Regular session if False, extended session if True.
            pro_data (bool): Whether to use pro data.
            limit (int): The maximum number of returned items.
            delay (float): Time to sleep after each request (in seconds).
            retries (int): The number of retries on failure to fetch data.

        For defaults, see `custom.tv` in `vectorbtpro._settings.data`.
        """
        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        exchange = cls.resolve_custom_setting(exchange, "exchange")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        fut_contract = cls.resolve_custom_setting(fut_contract, "fut_contract")
        adjustment = cls.resolve_custom_setting(adjustment, "adjustment")
        extended_session = cls.resolve_custom_setting(extended_session, "extended_session")
        pro_data = cls.resolve_custom_setting(pro_data, "pro_data")
        limit = cls.resolve_custom_setting(limit, "limit")
        delay = cls.resolve_custom_setting(delay, "delay")
        retries = cls.resolve_custom_setting(retries, "retries")

        freq = timeframe
        if not isinstance(timeframe, str):
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        split = dt.split_freq_str(timeframe)
        if split is None:
            raise ValueError(f"Invalid timeframe '{timeframe}'")
        multiplier, unit = split
        if unit == "s":
            interval = f"{str(multiplier)}S"
        elif unit == "m":
            interval = str(multiplier)
        elif unit == "h":
            interval = f"{str(multiplier)}H"
        elif unit == "D":
            interval = f"{str(multiplier)}D"
        elif unit == "W":
            interval = f"{str(multiplier)}W"
        elif unit == "M":
            interval = f"{str(multiplier)}M"
        else:
            raise ValueError(f"Invalid timeframe '{timeframe}'")

        for i in range(retries):
            try:
                df = client.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    fut_contract=fut_contract,
                    adjustment=adjustment,
                    extended_session=extended_session,
                    pro_data=pro_data,
                    limit=limit,
                )
                break
            except Exception as e:
                if i == retries - 1:
                    raise e
                if delay is not None:
                    time.sleep(delay)
        df.rename(
            columns={
                "symbol": "Symbol",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            df = df.tz_localize("utc")

        if "Symbol" in df:
            del df["Symbol"]
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

        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
