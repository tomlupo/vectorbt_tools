# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `SQLData`."""

from typing import Iterator

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.custom.db import DBData
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from sqlalchemy import Engine as EngineT, Selectable as SelectableT, Table as TableT
except ImportError:
    EngineT = tp.Any
    SelectableT = tp.Any
    TableT = tp.Any

__all__ = [
    "SQLData",
]

__pdoc__ = {}

SQLDataT = tp.TypeVar("SQLDataT", bound="SQLData")


class SQLData(DBData):
    """Data class for fetching data from a database using SQLAlchemy.

    See https://www.sqlalchemy.org/ for the SQLAlchemy's API.

    See https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html for the read method.

    See `SQLData.pull` and `SQLData.fetch_key` for arguments.

    Usage:
        * Set up the engine settings globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.SQLData.set_engine_settings(
        ...     engine_name="postgresql",
        ...     populate_=True,
        ...     engine="postgresql+psycopg2://...",
        ...     engine_config=dict(),
        ...     schema="public"
        ... )
        ```

        * Pull tables:

        ```pycon
        >>> data = vbt.SQLData.pull(
        ...     ["TABLE1", "TABLE2"],
        ...     engine="postgresql",
        ...     start="2020-01-01",
        ...     end="2021-01-01"
        ... )
        ```

        * Pull queries:

        ```pycon
        >>> data = vbt.SQLData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     query=vbt.key_dict({
        ...         "SYMBOL1": "SELECT * FROM TABLE1",
        ...         "SYMBOL2": "SELECT * FROM TABLE2"
        ...     }),
        ...     engine="postgresql"
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.sql")

    @classmethod
    def get_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> dict:
        """`SQLData.get_custom_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """`SQLData.has_custom_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def get_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """`SQLData.get_custom_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """`SQLData.has_custom_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """`SQLData.resolve_custom_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.resolve_custom_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def set_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> None:
        """`SQLData.set_custom_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        cls.set_custom_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine(
        cls,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        return_meta: bool = False,
        **engine_config,
    ) -> tp.Union[EngineT, dict]:
        """Resolve the engine.

        Argument `engine` can be

        1) an object of the type `sqlalchemy.engine.base.Engine`,
        2) a URL of the engine as a string, which will be used to create an engine with
        `sqlalchemy.engine.create.create_engine` and `engine_config` passed as keyword arguments
        (you should not include `url` in the `engine_config`), or
        3) an engine name, which is the name of a sub-config with engine settings under `custom.sql.engines`
        in `vectorbtpro._settings.data`. Such a sub-config can then contain the actual engine as an object or a URL.

        Argument `engine_name` can be provided instead of `engine`, or also together with `engine`
        to pull other settings from a sub-config. URLs can also be used as engine names, but not the
        other way around."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import create_engine

        if engine is None and engine_name is None:
            engine_name = cls.resolve_engine_setting(engine_name, "engine_name")
        if engine_name is not None:
            engine = cls.resolve_engine_setting(engine, "engine", engine_name=engine_name)
            if engine is None:
                raise ValueError("Must provide engine or URL (via engine argument)")
        else:
            engine = cls.resolve_engine_setting(engine, "engine")
            if engine is None:
                raise ValueError("Must provide engine or URL (via engine argument)")
            if isinstance(engine, str):
                engine_name = engine
            else:
                engine_name = None
            if engine_name is not None:
                if cls.has_engine_setting("engine", engine_name=engine_name, sub_path_only=True):
                    engine = cls.get_engine_setting("engine", engine_name=engine_name, sub_path_only=True)
        has_engine_config = len(engine_config) > 0
        engine_config = cls.resolve_engine_setting(engine_config, "engine_config", merge=True, engine_name=engine_name)
        if isinstance(engine, str):
            if engine.startswith("duckdb:"):
                assert_can_import("duckdb_engine")
            engine = create_engine(engine, **engine_config)
            should_dispose = True
        else:
            if has_engine_config:
                raise ValueError("Cannot apply engine_config on already created engine")
            should_dispose = False
        if return_meta:
            return dict(
                engine=engine,
                engine_name=engine_name,
                should_dispose=should_dispose,
            )
        return engine

    @classmethod
    def list_schemas(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all schemas.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`.

        Keyword arguments `**kwargs` are passed to `inspector.get_schema_names`.

        If `dispose_engine` is None, disposes the engine if it wasn't provided."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        inspector = inspect(engine)
        all_schemas = inspector.get_schema_names(**kwargs)
        schemas = []
        for schema in all_schemas:
            if pattern is not None:
                if not cls.key_match(schema, pattern, use_regex=use_regex):
                    continue
            if schema == "information_schema":
                continue
            schemas.append(schema)

        if dispose_engine:
            engine.dispose()
        if sort:
            return sorted(dict.fromkeys(schemas))
        return list(dict.fromkeys(schemas))

    @classmethod
    def list_tables(
        cls,
        *,
        schema_pattern: tp.Optional[str] = None,
        table_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        schema: tp.Optional[str] = None,
        incl_views: bool = True,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.List[str]:
        """List all tables and views.

        If `schema` is None, searches for all schema names in the database and prefixes each table
        with the respective schema name (unless there's only one schema "main"). If `schema` is False,
        sets the schema to None. If `schema` is provided, returns the tables corresponding to this
        schema without a prefix.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each schema against
        `schema_pattern` and each table against `table_pattern`.

        Keyword arguments `**kwargs` are passed to `inspector.get_table_names`.

        If `dispose_engine` is None, disposes the engine if it wasn't provided."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        schema = cls.resolve_engine_setting(schema, "schema", engine_name=engine_name)
        if schema is None:
            schemas = cls.list_schemas(
                pattern=schema_pattern,
                use_regex=use_regex,
                sort=sort,
                engine=engine,
                engine_name=engine_name,
                **kwargs,
            )
            if len(schemas) == 0:
                schemas = [None]
                prefix_schema = False
            elif len(schemas) == 1 and schemas[0] == "main":
                prefix_schema = False
            else:
                prefix_schema = True
        elif schema is False:
            schemas = [None]
            prefix_schema = False
        else:
            schemas = [schema]
            prefix_schema = False
        inspector = inspect(engine)
        tables = []
        for schema in schemas:
            all_tables = inspector.get_table_names(schema, **kwargs)
            if incl_views:
                try:
                    all_tables += inspector.get_view_names(schema, **kwargs)
                except NotImplementedError as e:
                    pass
                try:
                    all_tables += inspector.get_materialized_view_names(schema, **kwargs)
                except NotImplementedError as e:
                    pass
            for table in all_tables:
                if table_pattern is not None:
                    if not cls.key_match(table, table_pattern, use_regex=use_regex):
                        continue
                if prefix_schema and schema is not None:
                    table = str(schema) + ":" + table
                tables.append(table)

        if dispose_engine:
            engine.dispose()
        if sort:
            return sorted(dict.fromkeys(tables))
        return list(dict.fromkeys(tables))

    @classmethod
    def has_schema(
        cls,
        schema: str,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> bool:
        """Check whether the database has a schema."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        return inspect(engine).has_schema(schema)

    @classmethod
    def create_schema(
        cls,
        schema: str,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> None:
        """Create a schema if it doesn't exist yet."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy.schema import CreateSchema

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        if not cls.has_schema(schema, engine=engine, engine_name=engine_name):
            with engine.connect() as connection:
                connection.execute(CreateSchema(schema))
                connection.commit()

    @classmethod
    def has_table(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> bool:
        """Check whether the database has a table."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import inspect

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        return inspect(engine).has_table(table, schema=schema)

    @classmethod
    def get_table_relation(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> TableT:
        """Get table relation."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import MetaData

        if engine_config is None:
            engine_config = {}
        engine = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            **engine_config,
        )
        schema = cls.resolve_engine_setting(schema, "schema", engine_name=engine_name)
        metadata_obj = MetaData()
        metadata_obj.reflect(bind=engine, schema=schema, only=[table], views=True)
        if schema is not None and schema + "." + table in metadata_obj.tables:
            return metadata_obj.tables[schema + "." + table]
        return metadata_obj.tables[table]

    @classmethod
    def get_last_row_number(
        cls,
        table: str,
        schema: tp.Optional[str] = None,
        row_number_column: tp.Optional[str] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> TableT:
        """Get last row number."""
        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        row_number_column = cls.resolve_engine_setting(
            row_number_column,
            "row_number_column",
            engine_name=engine_name,
        )
        table_relation = cls.get_table_relation(table, schema=schema, engine=engine, engine_name=engine_name)
        table_column_names = []
        for column in table_relation.columns:
            table_column_names.append(column.name)
        if row_number_column not in table_column_names:
            raise ValueError(f"Row number column '{row_number_column}' not found")
        query = (
            table_relation.select()
            .with_only_columns(table_relation.columns.get(row_number_column))
            .order_by(table_relation.columns.get(row_number_column).desc())
            .limit(1)
        )
        with engine.connect() as connection:
            results = connection.execute(query)
            last_row_number = results.first()[0]
            connection.commit()
        return last_row_number

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        keys_meta = DBData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None:
            if cls.has_key_dict(schema):
                raise ValueError("Cannot populate keys if schema is defined per key")
            if cls.has_key_dict(list_tables_kwargs):
                raise ValueError("Cannot populate keys if list_tables_kwargs is defined per key")
            if cls.has_key_dict(engine):
                raise ValueError("Cannot populate keys if engine is defined per key")
            if cls.has_key_dict(engine_name):
                raise ValueError("Cannot populate keys if engine_name is defined per key")
            if cls.has_key_dict(engine_config):
                raise ValueError("Cannot populate keys if engine_config is defined per key")
            if list_tables_kwargs is None:
                list_tables_kwargs = {}
            keys_meta["keys"] = cls.list_tables(
                schema=schema,
                engine=engine,
                engine_name=engine_name,
                engine_config=engine_config,
                **list_tables_kwargs,
            )
        return keys_meta

    @classmethod
    def pull(
        cls: tp.Type[SQLDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        share_engine: tp.Optional[bool] = None,
        **kwargs,
    ) -> SQLDataT:
        """Override `vectorbtpro.data.base.Data.pull` to resolve and share the engine among the keys
        and use the table names available in the database in case no keys were provided."""
        if share_engine is None:
            if (
                not cls.has_key_dict(engine)
                and not cls.has_key_dict(engine_name)
                and not cls.has_key_dict(engine_config)
            ):
                share_engine = True
            else:
                share_engine = False
        if share_engine:
            if engine_config is None:
                engine_config = {}
            engine_meta = cls.resolve_engine(
                engine=engine,
                engine_name=engine_name,
                return_meta=True,
                **engine_config,
            )
            engine = engine_meta["engine"]
            engine_name = engine_meta["engine_name"]
            should_dispose = engine_meta["should_dispose"]
            if dispose_engine is None:
                dispose_engine = should_dispose
        else:
            engine_name = None
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            schema=schema,
            list_tables_kwargs=list_tables_kwargs,
            engine=engine,
            engine_name=engine_name,
            engine_config=engine_config,
        )
        keys = keys_meta["keys"]
        keys_are_features = keys_meta["keys_are_features"]
        outputs = super(DBData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            schema=schema,
            engine=engine,
            engine_name=engine_name,
            engine_config=engine_config,
            dispose_engine=False if share_engine else dispose_engine,
            **kwargs,
        )
        if share_engine and dispose_engine:
            engine.dispose()
        return outputs

    @classmethod
    def fetch_key(
        cls,
        key: str,
        table: tp.Union[None, str, TableT] = None,
        schema: tp.Optional[str] = None,
        query: tp.Union[None, str, SelectableT] = None,
        engine: tp.Union[None, str, EngineT] = None,
        engine_name: tp.Optional[str] = None,
        engine_config: tp.KwargsLike = None,
        dispose_engine: tp.Optional[bool] = None,
        start: tp.Optional[tp.Any] = None,
        end: tp.Optional[tp.Any] = None,
        align_dates: tp.Optional[bool] = None,
        parse_dates: tp.Union[None, bool, tp.List[tp.IntStr], tp.Dict[tp.IntStr, tp.Any]] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str]] = None,
        tz: tp.TimezoneLike = None,
        start_row: tp.Optional[int] = None,
        end_row: tp.Optional[int] = None,
        keep_row_number: tp.Optional[bool] = None,
        row_number_column: tp.Optional[str] = None,
        index_col: tp.Union[None, bool, tp.MaybeList[tp.IntStr]] = None,
        columns: tp.Optional[tp.MaybeList[tp.IntStr]] = None,
        dtype: tp.Union[None, tp.DTypeLike, tp.Dict[tp.IntStr, tp.DTypeLike]] = None,
        chunksize: tp.Optional[int] = None,
        chunk_func: tp.Optional[tp.Callable] = None,
        squeeze: tp.Optional[bool] = None,
        **read_sql_kwargs,
    ) -> tp.KeyData:
        """Fetch a feature or symbol from a SQL database.

        Can use a table name (which defaults to the key) or a custom query.

        Args:
            key (str): Feature or symbol.

                If `table` and `query` are both None, becomes the table name.

                Key can be in the `SCHEMA:TABLE` format, in this case `schema` argument will be ignored.
            table (str or Table): Table name or actual object.

                Cannot be used together with `query`.
            schema (str): Schema.

                Cannot be used together with `query`.
            query (str or Selectable): Custom query.

                Cannot be used together with `table` and `schema`.
            engine (str or object): See `SQLData.resolve_engine`.
            engine_name (str): See `SQLData.resolve_engine`.
            engine_config (dict): See `SQLData.resolve_engine`.
            dispose_engine (bool): See `SQLData.resolve_engine`.
            start (any): Start datetime (if datetime index) or any other start value.

                Will parse with `vectorbtpro.utils.datetime_.to_timestamp` if `align_dates` is True
                and the index is a datetime index. Otherwise, you must ensure the correct type is provided.

                If the index is a multi-index, start value must be a tuple.

                Cannot be used together with `query`. Include the condition into the query.
            end (any): End datetime (if datetime index) or any other end value.

                Will parse with `vectorbtpro.utils.datetime_.to_timestamp` if `align_dates` is True
                and the index is a datetime index. Otherwise, you must ensure the correct type is provided.

                If the index is a multi-index, end value must be a tuple.

                Cannot be used together with `query`. Include the condition into the query.
            align_dates (bool): Whether to align `start` and `end` to the timezone of the index.

                Will pull one row (using `LIMIT 1`) and use `SQLData.prepare_dt` to get the index.
            parse_dates (bool, list, or dict): Whether to parse dates and how to do it.

                If `query` is not used, will get mapped into column names. Otherwise,
                usage of integers is not allowed and column names directly must be used.
                If enabled, will also try to parse the datetime columns that couldn't be parsed
                by Pandas after the object has been fetched.

                For dict format, see `pd.read_sql_query`.
            to_utc (bool, str, or sequence of str): See `SQLData.prepare_dt`.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            start_row (int): Start row.

                Table must contain the column defined in `row_number_column`.

                Cannot be used together with `query`. Include the condition into the query.
            end_row (int): End row.

                Table must contain the column defined in `row_number_column`.

                Cannot be used together with `query`. Include the condition into the query.
            keep_row_number (bool): Whether to return the column defined in `row_number_column`.
            row_number_column (str): Name of the column with row numbers.
            index_col (int, str, or list): One or more columns that should become the index.

                If `query` is not used, will get mapped into column names. Otherwise,
                usage of integers is not allowed and column names directly must be used.
            columns (int, str, or list): One or more columns to select.

                Will get mapped into column names. Cannot be used together with `query`.
            dtype (dtype_like or dict): Data type of each column.

                If `query` is not used, will get mapped into column names. Otherwise,
                usage of integers is not allowed and column names directly must be used.

                For dict format, see `pd.read_sql_query`.
            chunksize (int): See `pd.read_sql_query`.
            chunk_func (callable): Function to select and concatenate chunks from `Iterator`.

                Gets called only if `chunksize` is set.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            **read_sql_kwargs: Other keyword arguments passed to `pd.read_sql_query`.

        See https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html for other arguments.

        For defaults, see `custom.sql` in `vectorbtpro._settings.data`.
        Global settings can be provided per engine name using the `engines` dictionary.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("sqlalchemy")
        from sqlalchemy import Selectable, Select, FromClause, and_, text

        if engine_config is None:
            engine_config = {}
        engine_meta = cls.resolve_engine(
            engine=engine,
            engine_name=engine_name,
            return_meta=True,
            **engine_config,
        )
        engine = engine_meta["engine"]
        engine_name = engine_meta["engine_name"]
        should_dispose = engine_meta["should_dispose"]
        if dispose_engine is None:
            dispose_engine = should_dispose
        if table is not None and query is not None:
            raise ValueError("Must provide either table name or query, not both")
        if schema is not None and query is not None:
            raise ValueError("Schema cannot be applied to custom queries")
        if table is None and query is None:
            if ":" in key:
                schema, table = key.split(":")
            else:
                table = key

        start = cls.resolve_engine_setting(start, "start", engine_name=engine_name)
        end = cls.resolve_engine_setting(end, "end", engine_name=engine_name)
        align_dates = cls.resolve_engine_setting(align_dates, "align_dates", engine_name=engine_name)
        parse_dates = cls.resolve_engine_setting(parse_dates, "parse_dates", engine_name=engine_name)
        to_utc = cls.resolve_engine_setting(to_utc, "to_utc", engine_name=engine_name)
        tz = cls.resolve_engine_setting(tz, "tz", engine_name=engine_name)
        start_row = cls.resolve_engine_setting(start_row, "start_row", engine_name=engine_name)
        end_row = cls.resolve_engine_setting(end_row, "end_row", engine_name=engine_name)
        keep_row_number = cls.resolve_engine_setting(keep_row_number, "keep_row_number", engine_name=engine_name)
        row_number_column = cls.resolve_engine_setting(row_number_column, "row_number_column", engine_name=engine_name)
        index_col = cls.resolve_engine_setting(index_col, "index_col", engine_name=engine_name)
        columns = cls.resolve_engine_setting(columns, "columns", engine_name=engine_name)
        dtype = cls.resolve_engine_setting(dtype, "dtype", engine_name=engine_name)
        chunksize = cls.resolve_engine_setting(chunksize, "chunksize", engine_name=engine_name)
        chunk_func = cls.resolve_engine_setting(chunk_func, "chunk_func", engine_name=engine_name)
        squeeze = cls.resolve_engine_setting(squeeze, "squeeze", engine_name=engine_name)
        read_sql_kwargs = cls.resolve_engine_setting(
            read_sql_kwargs, "read_sql_kwargs", merge=True, engine_name=engine_name
        )

        if query is None or isinstance(query, (Selectable, FromClause)):
            if query is None:
                if isinstance(table, str):
                    table = cls.get_table_relation(table, schema=schema, engine=engine, engine_name=engine_name)
            else:
                table = query

            table_column_names = []
            for column in table.columns:
                table_column_names.append(column.name)

            def _resolve_columns(c):
                if checks.is_int(c):
                    c = table_column_names[int(c)]
                elif not isinstance(c, str):
                    new_c = []
                    for _c in c:
                        if checks.is_int(_c):
                            new_c.append(table_column_names[int(_c)])
                        else:
                            if _c not in table_column_names:
                                for __c in table_column_names:
                                    if _c.lower() == __c.lower():
                                        _c = __c
                                        break
                            new_c.append(_c)
                    c = new_c
                else:
                    if c not in table_column_names:
                        for _c in table_column_names:
                            if c.lower() == _c.lower():
                                return _c
                return c

            if index_col is False:
                index_col = None
            if index_col is not None:
                index_col = _resolve_columns(index_col)
                if isinstance(index_col, str):
                    index_col = [index_col]
            if columns is not None:
                columns = _resolve_columns(columns)
                if isinstance(columns, str):
                    columns = [columns]
            if parse_dates is not None:
                if not isinstance(parse_dates, bool):
                    if isinstance(parse_dates, dict):
                        parse_dates = dict(zip(_resolve_columns(parse_dates.keys()), parse_dates.values()))
                    else:
                        parse_dates = _resolve_columns(parse_dates)
                    if isinstance(parse_dates, str):
                        parse_dates = [parse_dates]
            if dtype is not None:
                if isinstance(dtype, dict):
                    dtype = dict(zip(_resolve_columns(dtype.keys()), dtype.values()))

            if not isinstance(table, Select):
                query = table.select()
            else:
                query = table
            if index_col is not None:
                for col in index_col:
                    query = query.order_by(col)
            if index_col is not None and columns is not None:
                pre_columns = []
                for col in index_col:
                    if col not in columns:
                        pre_columns.append(col)
                columns = pre_columns + columns
            if keep_row_number and columns is not None:
                if row_number_column in table_column_names and row_number_column not in columns:
                    columns = [row_number_column] + columns
            elif not keep_row_number and columns is None:
                if row_number_column in table_column_names:
                    columns = [col for col in table_column_names if col != row_number_column]
            if columns is not None:
                query = query.with_only_columns(*[table.columns.get(c) for c in columns])

            def _to_native_type(x):
                if checks.is_np_scalar(x):
                    return x.item()
                return x

            if start_row is not None or end_row is not None:
                if start is not None or end is not None:
                    raise ValueError("Can either filter by row numbers or by index, not both")
                _row_number_column = table.columns.get(row_number_column)
                if _row_number_column is None:
                    raise ValueError(f"Row number column '{row_number_column}' not found")
                and_list = []
                if start_row is not None:
                    and_list.append(_row_number_column >= _to_native_type(start_row))
                if end_row is not None:
                    and_list.append(_row_number_column < _to_native_type(end_row))
                query = query.where(and_(*and_list))
            if start is not None or end is not None:
                if index_col is None:
                    raise ValueError("Must provide index column for filtering by start and end")
                if align_dates:
                    first_obj = pd.read_sql_query(
                        query.limit(1),
                        engine.connect(),
                        index_col=index_col,
                        parse_dates=None if isinstance(parse_dates, bool) else parse_dates,  # bool not accepted
                        dtype=dtype,
                        chunksize=None,
                        **read_sql_kwargs,
                    )
                    first_obj = cls.prepare_dt(
                        first_obj,
                        parse_dates=list(parse_dates) if isinstance(parse_dates, dict) else parse_dates,
                        to_utc=False,
                    )
                    if isinstance(first_obj.index, pd.DatetimeIndex):
                        if tz is None:
                            tz = first_obj.index.tz
                        if first_obj.index.tz is not None:
                            if start is not None:
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz=first_obj.index.tz)
                            if end is not None:
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz=first_obj.index.tz)
                        else:
                            if start is not None:
                                if (
                                    to_utc is True
                                    or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                    or (checks.is_sequence(to_utc) and first_obj.index.name in to_utc)
                                ):
                                    start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                                    start = dt.to_naive_datetime(start)
                                else:
                                    start = dt.to_naive_datetime(start, tz=tz)
                            if end is not None:
                                if (
                                    to_utc is True
                                    or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                    or (checks.is_sequence(to_utc) and first_obj.index.name in to_utc)
                                ):
                                    end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                                    end = dt.to_naive_datetime(end)
                                else:
                                    end = dt.to_naive_datetime(end, tz=tz)

                and_list = []
                if start is not None:
                    if len(index_col) > 1:
                        if not isinstance(start, tuple):
                            raise TypeError("Start must be a tuple if the index is a multi-index")
                        if len(start) != len(index_col):
                            raise ValueError("Start tuple must match the number of levels in the multi-index")
                        for i in range(len(index_col)):
                            index_column = table.columns.get(index_col[i])
                            and_list.append(index_column >= _to_native_type(start[i]))
                    else:
                        index_column = table.columns.get(index_col[0])
                        and_list.append(index_column >= _to_native_type(start))
                if end is not None:
                    if len(index_col) > 1:
                        if not isinstance(end, tuple):
                            raise TypeError("End must be a tuple if the index is a multi-index")
                        if len(end) != len(index_col):
                            raise ValueError("End tuple must match the number of levels in the multi-index")
                        for i in range(len(index_col)):
                            index_column = table.columns.get(index_col[i])
                            and_list.append(index_column < _to_native_type(end[i]))
                    else:
                        index_column = table.columns.get(index_col[0])
                        and_list.append(index_column < _to_native_type(end))
                query = query.where(and_(*and_list))
        else:

            def _check_columns(c, arg_name):
                if checks.is_int(c):
                    raise ValueError(f"Must provide column as a string for '{arg_name}'")
                elif not isinstance(c, str):
                    for _c in c:
                        if checks.is_int(_c):
                            raise ValueError(f"Must provide each column as a string for '{arg_name}'")

            if start is not None:
                raise ValueError("Start cannot be applied to custom queries")
            if end is not None:
                raise ValueError("End cannot be applied to custom queries")
            if start_row is not None:
                raise ValueError("Start row cannot be applied to custom queries")
            if end_row is not None:
                raise ValueError("End row cannot be applied to custom queries")
            if index_col is False:
                index_col = None
            if index_col is not None:
                _check_columns(index_col, "index_col")
                if isinstance(index_col, str):
                    index_col = [index_col]
            if columns is not None:
                raise ValueError("Columns cannot be applied to custom queries")
            if parse_dates is not None:
                if not isinstance(parse_dates, bool):
                    if isinstance(parse_dates, dict):
                        _check_columns(parse_dates.keys(), "parse_dates")
                    else:
                        _check_columns(parse_dates, "parse_dates")
                    if isinstance(parse_dates, str):
                        parse_dates = [parse_dates]
            if dtype is not None:
                _check_columns(dtype.keys(), "dtype")

        if isinstance(query, str):
            query = text(query)
        obj = pd.read_sql_query(
            query,
            engine.connect(),
            index_col=index_col,
            parse_dates=None if isinstance(parse_dates, bool) else parse_dates,  # bool not accepted
            dtype=dtype,
            chunksize=chunksize,
            **read_sql_kwargs,
        )
        if isinstance(obj, Iterator):
            if chunk_func is None:
                obj = pd.concat(list(obj), axis=0)
            else:
                obj = chunk_func(obj)
        obj = cls.prepare_dt(
            obj,
            parse_dates=list(parse_dates) if isinstance(parse_dates, dict) else parse_dates,
            to_utc=to_utc,
        )
        if not isinstance(obj.index, pd.MultiIndex):
            if obj.index.name == "index":
                obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None

        if dispose_engine:
            engine.dispose()
        if keep_row_number:
            return obj, dict(tz=tz, row_number_column=row_number_column)
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: str, **kwargs) -> tp.FeatureData:
        """Fetch the table of a feature.

        Uses `SQLData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: str, **kwargs) -> tp.SymbolData:
        """Fetch the table for a symbol.

        Uses `SQLData.fetch_key`."""
        return cls.fetch_key(symbol, **kwargs)

    def update_key(
        self,
        key: str,
        from_last_row: tp.Optional[bool] = None,
        from_last_index: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.KeyData:
        """Update data of a feature or symbol."""
        fetch_kwargs = self.select_fetch_kwargs(key)
        returned_kwargs = self.select_returned_kwargs(key)
        pre_kwargs = merge_dicts(fetch_kwargs, kwargs)
        if from_last_row is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_row = False
            elif from_last_index is True:
                from_last_row = False
            elif pre_kwargs.get("start", None) is not None or pre_kwargs.get("end", None) is not None:
                from_last_row = False
            elif "row_number_column" not in returned_kwargs:
                from_last_row = False
            elif returned_kwargs["row_number_column"] not in self.wrapper.columns:
                from_last_row = False
            else:
                from_last_row = True
        if from_last_index is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_index = False
            elif from_last_row is True:
                from_last_index = False
            elif pre_kwargs.get("start_row", None) is not None or pre_kwargs.get("end_row", None) is not None:
                from_last_index = False
            else:
                from_last_index = True
        if from_last_row:
            if "row_number_column" not in returned_kwargs:
                raise ValueError("Argument row_number_column must be in returned_kwargs for from_last_row")
            row_number_column = returned_kwargs["row_number_column"]
            fetch_kwargs["start_row"] = self.data[key][row_number_column].iloc[-1]
        if from_last_index:
            fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if self.feature_oriented:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: str, **kwargs) -> tp.FeatureData:
        """Update data of a feature.

        Uses `SQLData.update_key`."""
        return self.update_key(feature, **kwargs)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `SQLData.update_key`."""
        return self.update_key(symbol, **kwargs)
