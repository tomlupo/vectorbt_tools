# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `BentoData`."""

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.remote import RemoteData
from vectorbtpro.utils import datetime_ as dt
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from databento import Historical as HistoricalT
except ImportError:
    HistoricalT = tp.Any

__all__ = [
    "BentoData",
]


class BentoData(RemoteData):
    """Data class for fetching from Databento.

    See https://github.com/databento/databento-python for API.

    See `BentoData.fetch_symbol` for arguments.

    Usage:
        * Set up the API key globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.BentoData.set_custom_settings(
        ...     client_config=dict(
        ...         key="YOUR_KEY"
        ...     )
        ... )
        ```

        * Pull data:

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "AAPL",
        ...     dataset="XNAS.ITCH"
        ... )
        ```

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "AAPL",
        ...     dataset="XNAS.ITCH",
        ...     timeframe="hourly",
        ...     start="one week ago"
        ... )
        ```

        ```pycon
        >>> data = vbt.BentoData.pull(
        ...     "ES.FUT",
        ...     dataset="GLBX.MDP3",
        ...     stype_in="parent",
        ...     schema="mbo",
        ...     start="2022-06-10T14:30",
        ...     end="2022-06-11",
        ...     limit=1000
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.bento")

    @classmethod
    def resolve_client(cls, client: tp.Optional[HistoricalT] = None, **client_config) -> HistoricalT:
        """Resolve the client.

        If provided, must be of the type `databento.historical.client.Historical`.
        Otherwise, will be created using `client_config`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("databento")
        from databento import Historical

        client = cls.resolve_custom_setting(client, "client")
        if client_config is None:
            client_config = {}
        has_client_config = len(client_config) > 0
        client_config = cls.resolve_custom_setting(client_config, "client_config", merge=True)
        if client is None:
            client = Historical(**client_config)
        elif has_client_config:
            raise ValueError("Cannot apply client_config on already created client")
        return client

    @classmethod
    def get_cost(cls, symbols: tp.MaybeSymbols, **kwargs) -> float:
        """Get the cost of calling `BentoData.fetch_symbol` on one or more symbols."""
        if isinstance(symbols, str):
            symbols = [symbols]
        costs = []
        for symbol in symbols:
            client, params = cls.fetch_symbol(symbol, **kwargs, return_params=True)
            cost_arg_names = get_func_arg_names(client.metadata.get_cost)
            for k in list(params.keys()):
                if k not in cost_arg_names:
                    del params[k]
            costs.append(client.metadata.get_cost(**params, mode="historical"))
        return sum(costs)

    @classmethod
    def fetch_symbol(
        cls,
        symbol: str,
        client: tp.Optional[HistoricalT] = None,
        client_config: tp.KwargsLike = None,
        start: tp.Optional[tp.DatetimeLike] = None,
        end: tp.Optional[tp.DatetimeLike] = None,
        resolve_dates: tp.Optional[bool] = None,
        timeframe: tp.Optional[str] = None,
        tz: tp.TimezoneLike = None,
        dataset: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        return_params: bool = False,
        df_kwargs: tp.KwargsLike = None,
        **params,
    ) -> tp.Union[float, tp.SymbolData]:
        """Override `vectorbtpro.data.base.Data.fetch_symbol` to fetch a symbol from Databento.

        Args:
            symbol (str): Symbol.

                Symbol can be in the `DATASET:SYMBOL` format if `dataset` is None.
            client (binance.client.Client): Client.

                See `BentoData.resolve_client`.
            client_config (dict): Client config.

                See `BentoData.resolve_client`.
            start (any): Start datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbtpro.utils.datetime_.to_tzaware_datetime`.
            resolve_dates (bool): Whether to resolve `start` and `end`, or pass them as they are.
            timeframe (str): Timeframe to create `schema` from.

                Allows human-readable strings such as "1 minute".

                If `timeframe` and `schema` are both not None, will raise an error.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            dataset (str): See `databento.historical.client.Historical.get_range`.
            schema (str): See `databento.historical.client.Historical.get_range`.
            return_params (bool): Whether to return the client and (final) parameters instead of data.

                Used by `BentoData.get_cost`.
            df_kwargs (dict): Keyword arguments passed to `databento.common.dbnstore.DBNStore.to_df`.
            **params: Keyword arguments passed to `databento.historical.client.Historical.get_range`.

        For defaults, see `custom.bento` in `vectorbtpro._settings.data`.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("databento")

        if client_config is None:
            client_config = {}
        client = cls.resolve_client(client=client, **client_config)

        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        resolve_dates = cls.resolve_custom_setting(resolve_dates, "resolve_dates")
        timeframe = cls.resolve_custom_setting(timeframe, "timeframe")
        tz = cls.resolve_custom_setting(tz, "tz")
        dataset = cls.resolve_custom_setting(dataset, "dataset")
        schema = cls.resolve_custom_setting(schema, "schema")
        params = cls.resolve_custom_setting(params, "params", merge=True)
        df_kwargs = cls.resolve_custom_setting(df_kwargs, "df_kwargs", merge=True)

        if dataset is None:
            if ":" in symbol:
                dataset, symbol = symbol.split(":")
        if timeframe is None and schema is None:
            schema = "ohlcv-1d"
            freq = "1d"
        elif timeframe is not None:
            freq = timeframe
            split = dt.split_freq_str(timeframe)
            if split is not None:
                multiplier, unit = split
                timeframe = str(multiplier) + unit
                if schema is None or schema.lower() == "ohlcv":
                    schema = f"ohlcv-{timeframe}"
                else:
                    raise ValueError("Timeframe cannot be used together with schema")
        else:
            if schema.startswith("ohlcv-"):
                freq = schema[len("ohlcv-") :]
            else:
                freq = None
        if resolve_dates:
            dataset_range = client.metadata.get_dataset_range(dataset)
            if "start_date" in dataset_range:
                start_date = dt.to_tzaware_timestamp(dataset_range["start_date"], naive_tz="utc", tz="utc")
            else:
                start_date = dt.to_tzaware_timestamp(dataset_range["start"], naive_tz="utc", tz="utc")
            if "end_date" in dataset_range:
                end_date = dt.to_tzaware_timestamp(dataset_range["end_date"], naive_tz="utc", tz="utc")
            else:
                end_date = dt.to_tzaware_timestamp(dataset_range["end"], naive_tz="utc", tz="utc")
            if start is not None:
                start = dt.to_tzaware_timestamp(start, naive_tz=tz, tz="utc")
                if start < start_date:
                    start = start_date
            else:
                start = start_date
            if end is not None:
                end = dt.to_tzaware_timestamp(end, naive_tz=tz, tz="utc")
                if end > end_date:
                    end = end_date
            else:
                end = end_date
            if start.floor("d") == start:
                start = start.date().isoformat()
            else:
                start = start.isoformat()
            if end.floor("d") == end:
                end = end.date().isoformat()
            else:
                end = end.isoformat()

        params = merge_dicts(
            dict(
                dataset=dataset,
                start=start,
                end=end,
                symbols=symbol,
                schema=schema,
            ),
            params,
        )
        if return_params:
            return client, params

        df = client.timeseries.get_range(**params).to_df(**df_kwargs)
        return df, dict(tz=tz, freq=freq)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        fetch_kwargs = self.select_fetch_kwargs(symbol)
        fetch_kwargs["start"] = self.select_last_index(symbol)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        return self.fetch_symbol(symbol, **kwargs)
