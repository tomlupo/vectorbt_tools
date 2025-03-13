# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `DuckDBData`."""

from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.data.base import key_dict
from vectorbtpro.data.custom.db import DBData
from vectorbtpro.data.custom.file import FileData
from vectorbtpro.utils import checks, datetime_ as dt
from vectorbtpro.utils.config import merge_dicts

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from duckdb import DuckDBPyConnection as DuckDBPyConnectionT, DuckDBPyRelation as DuckDBPyRelationT
except ImportError:
    DuckDBPyConnectionT = tp.Any
    DuckDBPyRelationT = tp.Any

__all__ = [
    "DuckDBData",
]

__pdoc__ = {}

DuckDBDataT = tp.TypeVar("DuckDBDataT", bound="DuckDBData")


class DuckDBData(DBData):
    """Data class for fetching data using DuckDB.

    See `DuckDBData.pull` and `DuckDBData.fetch_key` for arguments.

    Usage:
        * Set up the connection settings globally (optional):

        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.DuckDBData.set_custom_settings(connection="database.duckdb")
        ```

        * Pull tables:

        ```pycon
        >>> data = vbt.DuckDBData.pull(["TABLE1", "TABLE2"])
        ```

        * Rename tables:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     table=vbt.key_dict({
        ...         "SYMBOL1": "TABLE1",
        ...         "SYMBOL2": "TABLE2"
        ...     })
        ... )
        ```

        * Pull queries:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     query=vbt.key_dict({
        ...         "SYMBOL1": "SELECT * FROM TABLE1",
        ...         "SYMBOL2": "SELECT * FROM TABLE2"
        ...     })
        ... )
        ```

        * Pull Parquet files:

        ```pycon
        >>> data = vbt.DuckDBData.pull(
        ...     ["SYMBOL1", "SYMBOL2"],
        ...     read_path=vbt.key_dict({
        ...         "SYMBOL1": "s1.parquet",
        ...         "SYMBOL2": "s2.parquet"
        ...     })
        ... )
        ```
    """

    _settings_path: tp.SettingsPath = dict(custom="data.custom.duckdb")

    @classmethod
    def resolve_connection(
        cls,
        connection: tp.Union[None, str, tp.PathLike, DuckDBPyConnectionT] = None,
        read_only: bool = True,
        return_meta: bool = False,
        **connection_config,
    ) -> tp.Union[DuckDBPyConnectionT, dict]:
        """Resolve the connection."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from duckdb import connect, default_connection

        connection_meta = {}
        connection = cls.resolve_custom_setting(connection, "connection")
        if connection_config is None:
            connection_config = {}
        has_connection_config = len(connection_config) > 0
        connection_config["read_only"] = read_only
        connection_config = cls.resolve_custom_setting(connection_config, "connection_config", merge=True)
        read_only = connection_config.pop("read_only", read_only)
        should_close = False
        if connection is None:
            if len(connection_config) == 0:
                connection = default_connection
            else:
                database = connection_config.pop("database", None)
                if "config" in connection_config or len(connection_config) == 0:
                    connection = connect(database, read_only=read_only, **connection_config)
                else:
                    connection = connect(database, read_only=read_only, config=connection_config)
                should_close = True
        elif isinstance(connection, (str, Path)):
            if "config" in connection_config or len(connection_config) == 0:
                connection = connect(str(connection), read_only=read_only, **connection_config)
            else:
                connection = connect(str(connection), read_only=read_only, config=connection_config)
            should_close = True
        elif has_connection_config:
            raise ValueError("Cannot apply connection_config on already created connection")

        if return_meta:
            connection_meta["connection"] = connection
            connection_meta["should_close"] = should_close
            return connection_meta
        return connection

    @classmethod
    def list_catalogs(
        cls,
        pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        incl_system: bool = False,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all catalogs.

        Catalogs "system" and "temp" are skipped if `incl_system` is False.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`."""
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        schemata_df = connection.sql("SELECT * FROM information_schema.schemata").df()
        catalogs = []
        for catalog in schemata_df["catalog_name"].tolist():
            if pattern is not None:
                if not cls.key_match(catalog, pattern, use_regex=use_regex):
                    continue
            if not incl_system and catalog == "system":
                continue
            if not incl_system and catalog == "temp":
                continue
            catalogs.append(catalog)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(catalogs))
        return list(dict.fromkeys(catalogs))

    @classmethod
    def list_schemas(
        cls,
        catalog_pattern: tp.Optional[str] = None,
        schema_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        catalog: tp.Optional[str] = None,
        incl_system: bool = False,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all schemas.

        If `catalog` is None, searches for all catalog names in the database and prefixes each schema
        with the respective catalog name. If `catalog` is provided, returns the schemas corresponding
        to this catalog without a prefix. Schemas "information_schema" and "pg_catalog" are skipped
        if `incl_system` is False.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each symbol against `pattern`."""
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is None:
            catalogs = cls.list_catalogs(
                pattern=catalog_pattern,
                use_regex=use_regex,
                sort=sort,
                incl_system=incl_system,
                connection=connection,
                connection_config=connection_config,
            )
            if len(catalogs) == 1:
                prefix_catalog = False
            else:
                prefix_catalog = True
        else:
            catalogs = [catalog]
            prefix_catalog = False
        schemata_df = connection.sql("SELECT * FROM information_schema.schemata").df()
        schemas = []
        for catalog in catalogs:
            all_schemas = schemata_df[schemata_df["catalog_name"] == catalog]["schema_name"].tolist()
            for schema in all_schemas:
                if schema_pattern is not None:
                    if not cls.key_match(schema, schema_pattern, use_regex=use_regex):
                        continue
                if not incl_system and schema == "information_schema":
                    continue
                if not incl_system and schema == "pg_catalog":
                    continue
                if prefix_catalog:
                    schema = catalog + ":" + schema
                schemas.append(schema)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(schemas))
        return list(dict.fromkeys(schemas))

    @classmethod
    def get_current_schema(
        cls,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> str:
        """Get the current schema."""
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        current_schema = connection.sql("SELECT current_schema()").fetchall()[0][0]

        if should_close:
            connection.close()
        return current_schema

    @classmethod
    def list_tables(
        cls,
        *,
        catalog_pattern: tp.Optional[str] = None,
        schema_pattern: tp.Optional[str] = None,
        table_pattern: tp.Optional[str] = None,
        use_regex: bool = False,
        sort: bool = True,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        incl_system: bool = False,
        incl_temporary: bool = False,
        incl_views: bool = True,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.List[str]:
        """List all tables and views.

        If `schema` is None, searches for all schema names in the database and prefixes each table
        with the respective catalog and schema name (unless there's only one schema which is the current
        schema or `schema` is `current_schema`). If `schema` is provided, returns the tables corresponding
        to this schema without a prefix.

        Uses `vectorbtpro.data.custom.custom.CustomData.key_match` to check each schema against
        `schema_pattern` and each table against `table_pattern`."""
        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is None:
            catalogs = cls.list_catalogs(
                pattern=catalog_pattern,
                use_regex=use_regex,
                sort=sort,
                incl_system=incl_system,
                connection=connection,
                connection_config=connection_config,
            )
            if catalog_pattern is None and len(catalogs) == 1:
                prefix_catalog = False
            else:
                prefix_catalog = True
        else:
            catalogs = [catalog]
            prefix_catalog = False
        current_schema = cls.get_current_schema(
            connection=connection,
            connection_config=connection_config,
        )
        if schema is None:
            catalogs_schemas = []
            for catalog in catalogs:
                catalog_schemas = cls.list_schemas(
                    schema_pattern=schema_pattern,
                    use_regex=use_regex,
                    sort=sort,
                    catalog=catalog,
                    incl_system=incl_system,
                    connection=connection,
                    connection_config=connection_config,
                )
                for schema in catalog_schemas:
                    catalogs_schemas.append((catalog, schema))
            if len(catalogs_schemas) == 1 and catalogs_schemas[0][1] == current_schema:
                prefix_schema = False
            else:
                prefix_schema = True
        else:
            if schema == "current_schema":
                schema = current_schema
            catalogs_schemas = []
            for catalog in catalogs:
                catalogs_schemas.append((catalog, schema))
            prefix_schema = prefix_catalog
        tables_df = connection.sql("SELECT * FROM information_schema.tables").df()
        tables = []
        for catalog, schema in catalogs_schemas:
            all_tables = []
            all_tables.extend(
                tables_df[
                    (tables_df["table_catalog"] == catalog)
                    & (tables_df["table_schema"] == schema)
                    & (tables_df["table_type"] == "BASE TABLE")
                ]["table_name"].tolist()
            )
            if incl_temporary:
                all_tables.extend(
                    tables_df[
                        (tables_df["table_catalog"] == catalog)
                        & (tables_df["table_schema"] == schema)
                        & (tables_df["table_type"] == "LOCAL TEMPORARY")
                    ]["table_name"].tolist()
                )
            if incl_views:
                all_tables.extend(
                    tables_df[
                        (tables_df["table_catalog"] == catalog)
                        & (tables_df["table_schema"] == schema)
                        & (tables_df["table_type"] == "VIEW")
                    ]["table_name"].tolist()
                )
            for table in all_tables:
                if table_pattern is not None:
                    if not cls.key_match(table, table_pattern, use_regex=use_regex):
                        continue
                if not prefix_catalog and prefix_schema:
                    table = schema + ":" + table
                elif prefix_catalog or prefix_schema:
                    table = catalog + ":" + schema + ":" + table
                tables.append(table)

        if should_close:
            connection.close()
        if sort:
            return sorted(dict.fromkeys(tables))
        return list(dict.fromkeys(tables))

    @classmethod
    def resolve_keys_meta(
        cls,
        keys: tp.Union[None, dict, tp.MaybeKeys] = None,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[None, dict, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, dict, tp.MaybeSymbols] = None,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        keys_meta = DBData.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
        )
        if keys_meta["keys"] is None:
            if cls.has_key_dict(catalog):
                raise ValueError("Cannot populate keys if catalog is defined per key")
            if cls.has_key_dict(schema):
                raise ValueError("Cannot populate keys if schema is defined per key")
            if cls.has_key_dict(list_tables_kwargs):
                raise ValueError("Cannot populate keys if list_tables_kwargs is defined per key")
            if cls.has_key_dict(connection):
                raise ValueError("Cannot populate keys if connection is defined per key")
            if cls.has_key_dict(connection_config):
                raise ValueError("Cannot populate keys if connection_config is defined per key")
            if cls.has_key_dict(read_path):
                raise ValueError("Cannot populate keys if read_path is defined per key")
            if cls.has_key_dict(read_format):
                raise ValueError("Cannot populate keys if read_format is defined per key")
            if read_path is not None or read_format is not None:
                if read_path is None:
                    read_path = "."
                if read_format is not None:
                    read_format = read_format.lower()
                    checks.assert_in(read_format, ["csv", "parquet", "json"], arg_name="read_format")
                keys_meta["keys"] = FileData.list_paths(read_path, extension=read_format)
            else:
                if list_tables_kwargs is None:
                    list_tables_kwargs = {}
                keys_meta["keys"] = cls.list_tables(
                    catalog=catalog,
                    schema=schema,
                    connection=connection,
                    connection_config=connection_config,
                    **list_tables_kwargs,
                )
        return keys_meta

    @classmethod
    def pull(
        cls: tp.Type[DuckDBDataT],
        keys: tp.Union[tp.MaybeKeys] = None,
        *,
        keys_are_features: tp.Optional[bool] = None,
        features: tp.Union[tp.MaybeFeatures] = None,
        symbols: tp.Union[tp.MaybeSymbols] = None,
        catalog: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        list_tables_kwargs: tp.KwargsLike = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
        share_connection: tp.Optional[bool] = None,
        **kwargs,
    ) -> DuckDBDataT:
        """Override `vectorbtpro.data.base.Data.pull` to resolve and share the connection among the keys
        and use the table names available in the database in case no keys were provided."""
        if share_connection is None:
            if not cls.has_key_dict(connection) and not cls.has_key_dict(connection_config):
                share_connection = True
            else:
                share_connection = False
        if share_connection:
            if connection_config is None:
                connection_config = {}
            connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
            connection = connection_meta["connection"]
            should_close = connection_meta["should_close"]
        else:
            should_close = False
        keys_meta = cls.resolve_keys_meta(
            keys=keys,
            keys_are_features=keys_are_features,
            features=features,
            symbols=symbols,
            catalog=catalog,
            schema=schema,
            list_tables_kwargs=list_tables_kwargs,
            read_path=read_path,
            read_format=read_format,
            connection=connection,
            connection_config=connection_config,
        )
        keys = keys_meta["keys"]
        if isinstance(read_path, key_dict):
            new_read_path = read_path.copy()
        else:
            new_read_path = key_dict()
        if isinstance(keys, dict):
            new_keys = {}
            for k, v in keys.items():
                if isinstance(k, Path):
                    new_k = FileData.path_to_key(k)
                    new_read_path[new_k] = k
                    k = new_k
                new_keys[k] = v
            keys = new_keys
        elif cls.has_multiple_keys(keys):
            new_keys = []
            for k in keys:
                if isinstance(k, Path):
                    new_k = FileData.path_to_key(k)
                    new_read_path[new_k] = k
                    k = new_k
                new_keys.append(k)
            keys = new_keys
        else:
            if isinstance(keys, Path):
                new_keys = FileData.path_to_key(keys)
                new_read_path[new_keys] = keys
                keys = new_keys
        if len(new_read_path) > 0:
            read_path = new_read_path
        keys_are_features = keys_meta["keys_are_features"]
        outputs = super(DBData, cls).pull(
            keys,
            keys_are_features=keys_are_features,
            catalog=catalog,
            schema=schema,
            read_path=read_path,
            read_format=read_format,
            connection=connection,
            connection_config=connection_config,
            **kwargs,
        )

        if should_close:
            connection.close()
        return outputs

    @classmethod
    def format_write_option(cls, option: tp.Any) -> str:
        """Format a write option."""
        if isinstance(option, str):
            return f"'{option}'"
        if isinstance(option, (tuple, list)):
            return "(" + ", ".join(map(str, option)) + ")"
        if isinstance(option, dict):
            return "{" + ", ".join(map(lambda y: f"{y[0]}: {cls.format_write_option(y[1])}", option.items())) + "}"
        return f"{option}"

    @classmethod
    def format_write_options(cls, options: tp.Union[str, dict]) -> str:
        """Format write options."""
        if isinstance(options, str):
            return options
        new_options = []
        for k, v in options.items():
            new_options.append(f"{k.upper()} {cls.format_write_option(v)}")
        return ", ".join(new_options)

    @classmethod
    def format_read_option(cls, option: tp.Any) -> str:
        """Format a read option."""
        if isinstance(option, str):
            return f"'{option}'"
        if isinstance(option, (tuple, list)):
            return "[" + ", ".join(map(cls.format_read_option, option)) + "]"
        if isinstance(option, dict):
            return "{" + ", ".join(map(lambda y: f"'{y[0]}': {cls.format_read_option(y[1])}", option.items())) + "}"
        return f"{option}"

    @classmethod
    def format_read_options(cls, options: tp.Union[str, dict]) -> str:
        """Format read options."""
        if isinstance(options, str):
            return options
        new_options = []
        for k, v in options.items():
            new_options.append(f"{k.lower()}={cls.format_read_option(v)}")
        return ", ".join(new_options)

    @classmethod
    def fetch_key(
        cls,
        key: str,
        table: tp.Optional[str] = None,
        schema: tp.Optional[str] = None,
        catalog: tp.Optional[str] = None,
        read_path: tp.Optional[tp.PathLike] = None,
        read_format: tp.Optional[str] = None,
        read_options: tp.Union[None, str, dict] = None,
        query: tp.Union[None, str, DuckDBPyRelationT] = None,
        connection: tp.Union[None, str, DuckDBPyConnectionT] = None,
        connection_config: tp.KwargsLike = None,
        start: tp.Optional[tp.Any] = None,
        end: tp.Optional[tp.Any] = None,
        align_dates: tp.Optional[bool] = None,
        parse_dates: tp.Union[None, bool, tp.Sequence[str]] = None,
        to_utc: tp.Union[None, bool, str, tp.Sequence[str]] = None,
        tz: tp.TimezoneLike = None,
        index_col: tp.Optional[tp.MaybeSequence[tp.IntStr]] = None,
        squeeze: tp.Optional[bool] = None,
        df_kwargs: tp.KwargsLike = None,
        **sql_kwargs,
    ) -> tp.KeyData:
        """Fetch a feature or symbol from a DuckDB database.

        Can use a table name (which defaults to the key) or a custom query.

        Args:
            key (str): Feature or symbol.

                If `table` and `query` are both None, becomes the table name.

                Key can be in the `SCHEMA:TABLE` format, in this case `schema` argument will be ignored.
            table (str): Table name.

                Cannot be used together with `file` or `query`.
            schema (str): Schema name.

                Cannot be used together with `file` or `query`.
            catalog (str): Catalog name.

                Cannot be used together with ``file` or query`.
            read_path (path_like): Path to a file to read.

                Cannot be used together with `table`, `schema`, `catalog`, or `query`.
            read_format (str): Format of the file to read.

                Allowed values are "csv", "parquet", and "json".

                Requires `read_path` to be set.
            read_options (str or dict): Options used to read the file.

                Requires `read_path` and `read_format` to be set.

                Uses `DuckDBData.format_read_options` to transform a dictionary to a string.
            query (str or DuckDBPyRelation): Custom query.

                Cannot be used together with `catalog`, `schema`, and `table`.
            connection (str or object): See `DuckDBData.resolve_connection`.
            connection_config (dict): See `DuckDBData.resolve_connection`.
            start (any): Start datetime (if datetime index) or any other start value.

                Will parse with `vectorbtpro.utils.datetime_.to_timestamp` if `align_dates` is True
                and the index is a datetime index. Otherwise, you must ensure the correct type is provided.

                Cannot be used together with `query`. Include the condition into the query.
            end (any): End datetime (if datetime index) or any other end value.

                Will parse with `vectorbtpro.utils.datetime_.to_timestamp` if `align_dates` is True
                and the index is a datetime index. Otherwise, you must ensure the correct type is provided.

                Cannot be used together with `query`. Include the condition into the query.
            align_dates (bool): Whether to align `start` and `end` to the timezone of the index.

                Will pull one row (using `LIMIT 1`) and use `SQLData.prepare_dt` to get the index.
            parse_dates (bool or sequence of str): See `DuckDBData.prepare_dt`.
            to_utc (bool, str, or sequence of str): See `DuckDBData.prepare_dt`.
            tz (any): Timezone.

                See `vectorbtpro.utils.datetime_.to_timezone`.
            index_col (int, str, or list): One or more columns that should become the index.
            squeeze (int): Whether to squeeze a DataFrame with one column into a Series.
            df_kwargs (dict): Keyword arguments passed to `relation.df` to convert a relation to a DataFrame.
            **sql_kwargs: Other keyword arguments passed to `connection.execute` to run a SQL query.

        For defaults, see `custom.duckdb` in `vectorbtpro._settings.data`."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("duckdb")
        from duckdb import DuckDBPyRelation

        if connection_config is None:
            connection_config = {}
        connection_meta = cls.resolve_connection(connection, return_meta=True, **connection_config)
        connection = connection_meta["connection"]
        should_close = connection_meta["should_close"]
        if catalog is not None and query is not None:
            raise ValueError("Cannot use catalog and query together")
        if schema is not None and query is not None:
            raise ValueError("Cannot use schema and query together")
        if table is not None and query is not None:
            raise ValueError("Cannot use table and query together")
        if read_path is not None and query is not None:
            raise ValueError("Cannot use read_path and query together")
        if read_path is not None and (catalog is not None or schema is not None or table is not None):
            raise ValueError("Cannot use read_path and catalog/schema/table together")
        if table is None and read_path is None and read_format is None and query is None:
            if ":" in key:
                key_parts = key.split(":")
                if len(key_parts) == 2:
                    schema, table = key_parts
                else:
                    catalog, schema, table = key_parts
            else:
                table = key
        if read_format is not None:
            read_format = read_format.lower()
            checks.assert_in(read_format, ["csv", "parquet", "json"], arg_name="read_format")
            if read_path is None:
                read_path = (Path(".") / key).with_suffix("." + read_format)
        else:
            if read_path is not None:
                if isinstance(read_path, str):
                    read_path = Path(read_path)
                if read_path.suffix[1:] in ["csv", "parquet", "json"]:
                    read_format = read_path.suffix[1:]
        if read_path is not None:
            if isinstance(read_path, Path):
                read_path = str(read_path)
            read_path = cls.format_read_option(read_path)
        if read_options is not None:
            if read_format is None:
                raise ValueError("Must provide read_format for read_options")
            read_options = cls.format_read_options(read_options)

        catalog = cls.resolve_custom_setting(catalog, "catalog")
        schema = cls.resolve_custom_setting(schema, "schema")
        start = cls.resolve_custom_setting(start, "start")
        end = cls.resolve_custom_setting(end, "end")
        align_dates = cls.resolve_custom_setting(align_dates, "align_dates")
        parse_dates = cls.resolve_custom_setting(parse_dates, "parse_dates")
        to_utc = cls.resolve_custom_setting(to_utc, "to_utc")
        tz = cls.resolve_custom_setting(tz, "tz")
        index_col = cls.resolve_custom_setting(index_col, "index_col")
        squeeze = cls.resolve_custom_setting(squeeze, "squeeze")
        df_kwargs = cls.resolve_custom_setting(df_kwargs, "df_kwargs", merge=True)
        sql_kwargs = cls.resolve_custom_setting(sql_kwargs, "sql_kwargs", merge=True)

        if query is None:
            if read_path is not None:
                if read_options is not None:
                    query = f"SELECT * FROM read_{read_format}({read_path}, {read_options})"
                elif read_format is not None:
                    query = f"SELECT * FROM read_{read_format}({read_path})"
                else:
                    query = f"SELECT * FROM {read_path}"
            else:
                if catalog is not None:
                    if schema is None:
                        schema = cls.get_current_schema(
                            connection=connection,
                            connection_config=connection_config,
                        )
                    query = f'SELECT * FROM "{catalog}"."{schema}"."{table}"'
                elif schema is not None:
                    query = f'SELECT * FROM "{schema}"."{table}"'
                else:
                    query = f'SELECT * FROM "{table}"'
            if start is not None or end is not None:
                if index_col is None:
                    raise ValueError("Must provide index column for filtering by start and end")
                if not checks.is_int(index_col) and not isinstance(index_col, str):
                    raise ValueError("Index column must be integer or string for filtering by start and end")
                if checks.is_int(index_col) or align_dates:
                    metadata_df = connection.sql("DESCRIBE " + query + " LIMIT 1").df()
                else:
                    metadata_df = None
                if checks.is_int(index_col):
                    index_name = metadata_df["column_name"].tolist()[0]
                else:
                    index_name = index_col
                if parse_dates:
                    index_column_type = metadata_df[metadata_df["column_name"] == index_name]["column_type"].item()
                    if index_column_type in (
                        "TIMESTAMP_NS",
                        "TIMESTAMP_MS",
                        "TIMESTAMP_S",
                        "TIMESTAMP",
                        "DATETIME",
                    ):
                        if start is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                                start = dt.to_naive_datetime(start)
                            else:
                                start = dt.to_naive_datetime(start, tz=tz)
                        if end is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                                end = dt.to_naive_datetime(end)
                            else:
                                end = dt.to_naive_datetime(end, tz=tz)
                    elif index_column_type in ("TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"):
                        if start is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                start = dt.to_tzaware_datetime(start, naive_tz=tz, tz="utc")
                            else:
                                start = dt.to_tzaware_datetime(start, naive_tz=tz)
                        if end is not None:
                            if (
                                to_utc is True
                                or (isinstance(to_utc, str) and to_utc.lower() == "index")
                                or (checks.is_sequence(to_utc) and index_name in to_utc)
                            ):
                                end = dt.to_tzaware_datetime(end, naive_tz=tz, tz="utc")
                            else:
                                end = dt.to_tzaware_datetime(end, naive_tz=tz)
                if start is not None and end is not None:
                    query += f' WHERE "{index_name}" >= $start AND "{index_name}" < $end'
                elif start is not None:
                    query += f' WHERE "{index_name}" >= $start'
                elif end is not None:
                    query += f' WHERE "{index_name}" < $end'
                params = sql_kwargs.get("params", None)
                if params is None:
                    params = {}
                else:
                    params = dict(params)
                if not isinstance(params, dict):
                    raise ValueError("Parameters must be a dictionary for filtering by start and end")
                if start is not None:
                    if "start" in params:
                        raise ValueError("Start is already in params")
                    params["start"] = start
                if end is not None:
                    if "end" in params:
                        raise ValueError("End is already in params")
                    params["end"] = end
                sql_kwargs["params"] = params
        else:
            if start is not None:
                raise ValueError("Start cannot be applied to custom queries")
            if end is not None:
                raise ValueError("End cannot be applied to custom queries")

        if not isinstance(query, DuckDBPyRelation):
            relation = connection.sql(query, **sql_kwargs)
        else:
            relation = query
        obj = relation.df(**df_kwargs)

        if isinstance(obj, pd.DataFrame) and checks.is_default_index(obj.index):
            if index_col is not None:
                if checks.is_int(index_col):
                    keys = obj.columns[index_col]
                elif isinstance(index_col, str):
                    keys = index_col
                else:
                    keys = []
                    for col in index_col:
                        if checks.is_int(col):
                            keys.append(obj.columns[col])
                        else:
                            keys.append(col)
                obj = obj.set_index(keys)
                if not isinstance(obj.index, pd.MultiIndex):
                    if obj.index.name == "index":
                        obj.index.name = None
        obj = cls.prepare_dt(obj, to_utc=to_utc, parse_dates=parse_dates)
        if not isinstance(obj.index, pd.MultiIndex):
            if obj.index.name == "index":
                obj.index.name = None
        if isinstance(obj.index, pd.DatetimeIndex) and tz is None:
            tz = obj.index.tz
        if isinstance(obj, pd.DataFrame) and squeeze:
            obj = obj.squeeze("columns")
        if isinstance(obj, pd.Series) and obj.name == "0":
            obj.name = None

        if should_close:
            connection.close()
        return obj, dict(tz=tz)

    @classmethod
    def fetch_feature(cls, feature: str, **kwargs) -> tp.FeatureData:
        """Fetch the table of a feature.

        Uses `DuckDBData.fetch_key`."""
        return cls.fetch_key(feature, **kwargs)

    @classmethod
    def fetch_symbol(cls, symbol: str, **kwargs) -> tp.SymbolData:
        """Fetch the table for a symbol.

        Uses `DuckDBData.fetch_key`."""
        return cls.fetch_key(symbol, **kwargs)

    def update_key(self, key: str, from_last_index: tp.Optional[bool] = None, **kwargs) -> tp.KeyData:
        """Update data of a feature or symbol."""
        fetch_kwargs = self.select_fetch_kwargs(key)
        pre_kwargs = merge_dicts(fetch_kwargs, kwargs)
        if from_last_index is None:
            if pre_kwargs.get("query", None) is not None:
                from_last_index = False
            else:
                from_last_index = True
        if from_last_index:
            fetch_kwargs["start"] = self.select_last_index(key)
        kwargs = merge_dicts(fetch_kwargs, kwargs)
        if self.feature_oriented:
            return self.fetch_feature(key, **kwargs)
        return self.fetch_symbol(key, **kwargs)

    def update_feature(self, feature: str, **kwargs) -> tp.FeatureData:
        """Update data of a feature.

        Uses `DuckDBData.update_key`."""
        return self.update_key(feature, **kwargs)

    def update_symbol(self, symbol: str, **kwargs) -> tp.SymbolData:
        """Update data for a symbol.

        Uses `DuckDBData.update_key`."""
        return self.update_key(symbol, **kwargs)
