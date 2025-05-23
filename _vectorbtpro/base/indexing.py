# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Classes and functions for indexing."""

import functools
from datetime import time
from functools import partial

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BaseOffset

from vectorbtpro import _typing as tp
from vectorbtpro.registries.jit_registry import jit_reg
from vectorbtpro.utils import checks, datetime_ as dt, datetime_nb as dt_nb
from vectorbtpro.utils.attr_ import DefineMixin, define, MISSING
from vectorbtpro.utils.config import hdict, merge_dicts
from vectorbtpro.utils.mapping import to_field_mapping
from vectorbtpro.utils.pickling import pdict
from vectorbtpro.utils.selection import PosSel, LabelSel
from vectorbtpro.utils.template import CustomTemplate

__all__ = [
    "PandasIndexer",
    "ExtPandasIndexer",
    "hslice",
    "get_index_points",
    "get_index_ranges",
    "get_idxs",
    "index_dict",
    "IdxSetter",
    "IdxSetterFactory",
    "IdxDict",
    "IdxSeries",
    "IdxFrame",
    "IdxRecords",
    "posidx",
    "maskidx",
    "lbidx",
    "dtidx",
    "dtcidx",
    "pointidx",
    "rangeidx",
    "autoidx",
    "rowidx",
    "colidx",
    "idx",
]

__pdoc__ = {}


class IndexingError(Exception):
    """Exception raised when an indexing error has occurred."""


IndexingBaseT = tp.TypeVar("IndexingBaseT", bound="IndexingBase")


class IndexingBase:
    """Class that supports indexing through `IndexingBase.indexing_func`."""

    def indexing_func(self: IndexingBaseT, pd_indexing_func: tp.Callable, **kwargs) -> IndexingBaseT:
        """Apply `pd_indexing_func` on all pandas objects in question and return a new instance of the class.

        Should be overridden."""
        raise NotImplementedError

    def indexing_setter_func(self, pd_indexing_setter_func: tp.Callable, **kwargs) -> None:
        """Apply `pd_indexing_setter_func` on all pandas objects in question.

        Should be overridden."""
        raise NotImplementedError


class LocBase:
    """Class that implements location-based indexing."""

    def __init__(
        self,
        indexing_func: tp.Callable,
        indexing_setter_func: tp.Optional[tp.Callable] = None,
        **kwargs,
    ) -> None:
        self._indexing_func = indexing_func
        self._indexing_setter_func = indexing_setter_func
        self._indexing_kwargs = kwargs

    @property
    def indexing_func(self) -> tp.Callable:
        """Indexing function."""
        return self._indexing_func

    @property
    def indexing_setter_func(self) -> tp.Optional[tp.Callable]:
        """Indexing setter function."""
        return self._indexing_setter_func

    @property
    def indexing_kwargs(self) -> dict:
        """Keyword arguments passed to `LocBase.indexing_func`."""
        return self._indexing_kwargs

    def __getitem__(self, key: tp.Any) -> tp.Any:
        raise NotImplementedError

    def __setitem__(self, key: tp.Any, value: tp.Any) -> None:
        raise NotImplementedError

    def __iter__(self):
        raise TypeError(f"'{type(self).__name__}' object is not iterable")


class pdLoc(LocBase):
    """Forwards a Pandas-like indexing operation to each Series/DataFrame and returns a new class instance."""

    @classmethod
    def pd_indexing_func(cls, obj: tp.SeriesFrame, key: tp.Any) -> tp.MaybeSeriesFrame:
        """Pandas-like indexing operation."""
        raise NotImplementedError

    @classmethod
    def pd_indexing_setter_func(cls, obj: tp.SeriesFrame, key: tp.Any, value: tp.Any) -> None:
        """Pandas-like indexing setter operation."""
        raise NotImplementedError

    def __getitem__(self, key: tp.Any) -> tp.Any:
        return self.indexing_func(partial(self.pd_indexing_func, key=key), **self.indexing_kwargs)

    def __setitem__(self, key: tp.Any, value: tp.Any) -> None:
        self.indexing_setter_func(partial(self.pd_indexing_setter_func, key=key, value=value), **self.indexing_kwargs)


class iLoc(pdLoc):
    """Forwards `pd.Series.iloc`/`pd.DataFrame.iloc` operation to each
    Series/DataFrame and returns a new class instance."""

    @classmethod
    def pd_indexing_func(cls, obj: tp.SeriesFrame, key: tp.Any) -> tp.MaybeSeriesFrame:
        return obj.iloc.__getitem__(key)

    @classmethod
    def pd_indexing_setter_func(cls, obj: tp.SeriesFrame, key: tp.Any, value: tp.Any) -> None:
        obj.iloc.__setitem__(key, value)


class Loc(pdLoc):
    """Forwards `pd.Series.loc`/`pd.DataFrame.loc` operation to each
    Series/DataFrame and returns a new class instance."""

    @classmethod
    def pd_indexing_func(cls, obj: tp.SeriesFrame, key: tp.Any) -> tp.MaybeSeriesFrame:
        return obj.loc.__getitem__(key)

    @classmethod
    def pd_indexing_setter_func(cls, obj: tp.SeriesFrame, key: tp.Any, value: tp.Any) -> None:
        obj.loc.__setitem__(key, value)


PandasIndexerT = tp.TypeVar("PandasIndexerT", bound="PandasIndexer")


class PandasIndexer(IndexingBase):
    """Implements indexing using `iloc`, `loc`, `xs` and `__getitem__`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.indexing import PandasIndexer

        >>> class C(PandasIndexer):
        ...     def __init__(self, df1, df2):
        ...         self.df1 = df1
        ...         self.df2 = df2
        ...         super().__init__()
        ...
        ...     def indexing_func(self, pd_indexing_func):
        ...         return type(self)(
        ...             pd_indexing_func(self.df1),
        ...             pd_indexing_func(self.df2)
        ...         )

        >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        >>> c = C(df1, df2)

        >>> c.iloc[:, 0]
        <__main__.C object at 0x1a1cacbbe0>

        >>> c.iloc[:, 0].df1
        0    1
        1    2
        Name: a, dtype: int64

        >>> c.iloc[:, 0].df2
        0    5
        1    6
        Name: a, dtype: int64
        ```
    """

    def __init__(self, **kwargs) -> None:
        self._iloc = iLoc(self.indexing_func, indexing_setter_func=self.indexing_setter_func, **kwargs)
        self._loc = Loc(self.indexing_func, indexing_setter_func=self.indexing_setter_func, **kwargs)
        self._indexing_kwargs = kwargs

    @property
    def indexing_kwargs(self) -> dict:
        """Indexing keyword arguments."""
        return self._indexing_kwargs

    @property
    def iloc(self) -> iLoc:
        """Purely integer-location based indexing for selection by position."""
        return self._iloc

    iloc.__doc__ = iLoc.__doc__

    @property
    def loc(self) -> Loc:
        """Purely label-location based indexer for selection by label."""
        return self._loc

    loc.__doc__ = Loc.__doc__

    def xs(self: PandasIndexerT, *args, **kwargs) -> PandasIndexerT:
        """Forwards `pd.Series.xs`/`pd.DataFrame.xs`
        operation to each Series/DataFrame and returns a new class instance."""
        return self.indexing_func(lambda x: x.xs(*args, **kwargs), **self.indexing_kwargs)

    def __getitem__(self: PandasIndexerT, key: tp.Any) -> PandasIndexerT:
        def __getitem__func(x, _key=key):
            return x.__getitem__(_key)

        return self.indexing_func(__getitem__func, **self.indexing_kwargs)

    def __setitem__(self, key: tp.Any, value: tp.Any) -> None:
        def __setitem__func(x, _key=key, _value=value):
            return x.__setitem__(_key, _value)

        self.indexing_setter_func(__setitem__func, **self.indexing_kwargs)

    def __iter__(self):
        raise TypeError(f"'{type(self).__name__}' object is not iterable")


class xLoc(iLoc):
    """Subclass of `iLoc` that transforms an `Idxr`-based operation with
    `get_idxs` to an `iLoc` operation."""

    @classmethod
    def pd_indexing_func(cls, obj: tp.SeriesFrame, key: tp.Any) -> tp.MaybeSeriesFrame:
        from vectorbtpro.base.indexes import get_index

        if isinstance(key, tuple):
            key = Idxr(*key)
        index = get_index(obj, 0)
        columns = get_index(obj, 1)
        freq = dt.infer_index_freq(index)
        row_idxs, col_idxs = get_idxs(key, index=index, columns=columns, freq=freq)
        if isinstance(row_idxs, np.ndarray) and row_idxs.ndim == 2:
            row_idxs = normalize_idxs(row_idxs, target_len=len(index))
        if isinstance(col_idxs, np.ndarray) and col_idxs.ndim == 2:
            col_idxs = normalize_idxs(col_idxs, target_len=len(columns))
        if isinstance(obj, pd.Series):
            if not isinstance(col_idxs, (slice, hslice)) or (
                col_idxs.start is not None or col_idxs.stop is not None or col_idxs.step is not None
            ):
                raise IndexingError("Too many indexers")
            return obj.iloc.__getitem__(row_idxs)
        return obj.iloc.__getitem__((row_idxs, col_idxs))

    @classmethod
    def pd_indexing_setter_func(cls, obj: tp.SeriesFrame, key: tp.Any, value: tp.Any) -> None:
        IdxSetter([(key, value)]).set_pd(obj)


class ExtPandasIndexer(PandasIndexer):
    """Extension of `PandasIndexer` that also implements indexing using `xLoc`."""

    def __init__(self, **kwargs) -> None:
        self._xloc = xLoc(self.indexing_func, indexing_setter_func=self.indexing_setter_func, **kwargs)
        PandasIndexer.__init__(self, **kwargs)

    @property
    def xloc(self) -> xLoc:
        """`Idxr`-based indexing."""
        return self._xloc

    xloc.__doc__ = xLoc.__doc__


class ParamLoc(LocBase):
    """Access a group of columns by parameter using `pd.Series.loc`.

    Uses `mapper` to establish link between columns and parameter values."""

    def __init__(
        self,
        mapper: tp.Series,
        indexing_func: tp.Callable,
        indexing_setter_func: tp.Optional[tp.Callable] = None,
        level_name: tp.Level = None,
        **kwargs,
    ) -> None:
        checks.assert_instance_of(mapper, pd.Series)

        if mapper.dtype == "O":
            # If params are objects, we must cast them to string first
            # The original mapper isn't touched
            mapper = mapper.astype(str)
        self._mapper = mapper
        self._level_name = level_name

        LocBase.__init__(self, indexing_func, indexing_setter_func=indexing_setter_func, **kwargs)

    @property
    def mapper(self) -> tp.Series:
        """Mapper."""
        return self._mapper

    @property
    def level_name(self) -> tp.Level:
        """Level name."""
        return self._level_name

    def get_idxs(self, key: tp.Any) -> tp.Array1d:
        """Get array of indices affected by this key."""
        if self.mapper.dtype == "O":
            # We must also cast the key to string
            if isinstance(key, (slice, hslice)):
                start = str(key.start) if key.start is not None else None
                stop = str(key.stop) if key.stop is not None else None
                key = slice(start, stop, key.step)
            elif isinstance(key, (list, np.ndarray)):
                key = list(map(str, key))
            else:
                # Tuples, objects, etc.
                key = str(key)
        # Use pandas to perform indexing
        mapper = pd.Series(np.arange(len(self.mapper.index)), index=self.mapper.values)
        idxs = mapper.loc.__getitem__(key)
        if isinstance(idxs, pd.Series):
            idxs = idxs.values
        return idxs

    def __getitem__(self, key: tp.Any) -> tp.Any:
        idxs = self.get_idxs(key)
        is_multiple = isinstance(key, (slice, hslice, list, np.ndarray))

        def pd_indexing_func(obj: tp.SeriesFrame) -> tp.MaybeSeriesFrame:
            from vectorbtpro.base.indexes import drop_levels

            new_obj = obj.iloc[:, idxs]
            if not is_multiple:
                # If we selected only one param, then remove its columns levels to keep it clean
                if self.level_name is not None:
                    if checks.is_frame(new_obj):
                        if isinstance(new_obj.columns, pd.MultiIndex):
                            new_obj.columns = drop_levels(new_obj.columns, self.level_name)
            return new_obj

        return self.indexing_func(pd_indexing_func, **self.indexing_kwargs)

    def __setitem__(self, key: tp.Any, value: tp.Any) -> None:
        idxs = self.get_idxs(key)

        def pd_indexing_setter_func(obj: tp.SeriesFrame) -> None:
            obj.iloc[:, idxs] = value

        return self.indexing_setter_func(pd_indexing_setter_func, **self.indexing_kwargs)


def indexing_on_mapper(
    mapper: tp.Series,
    ref_obj: tp.SeriesFrame,
    pd_indexing_func: tp.Callable,
) -> tp.Optional[tp.Series]:
    """Broadcast `mapper` Series to `ref_obj` and perform pandas indexing using `pd_indexing_func`."""
    from vectorbtpro.base.reshaping import broadcast_to

    checks.assert_instance_of(mapper, pd.Series)
    checks.assert_instance_of(ref_obj, (pd.Series, pd.DataFrame))

    if isinstance(ref_obj, pd.Series):
        range_mapper = broadcast_to(0, ref_obj)
    else:
        range_mapper = broadcast_to(np.arange(len(mapper.index))[None], ref_obj)
    loced_range_mapper = pd_indexing_func(range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)
    return None


def build_param_indexer(
    param_names: tp.Sequence[str],
    class_name: str = "ParamIndexer",
    module_name: tp.Optional[str] = None,
) -> tp.Type[IndexingBase]:
    """A factory to create a class with parameter indexing.

    Parameter indexer enables accessing a group of rows and columns by a parameter array (similar to `loc`).
    This way, one can query index/columns by another Series called a parameter mapper, which is just a
    `pd.Series` that maps columns (its index) to params (its values).

    Parameter indexing is important, since querying by column/index labels alone is not always the best option.
    For example, `pandas` doesn't let you query by list at a specific index/column level.

    Args:
        param_names (list of str): Names of the parameters.
        class_name (str): Name of the generated class.
        module_name (str): Name of the module to which the class should be bound.

    Usage:
        ```pycon
        >>> from vectorbtpro import *
        >>> from vectorbtpro.base.indexing import build_param_indexer, indexing_on_mapper

        >>> MyParamIndexer = build_param_indexer(['my_param'])
        >>> class C(MyParamIndexer):
        ...     def __init__(self, df, param_mapper):
        ...         self.df = df
        ...         self._my_param_mapper = param_mapper
        ...         super().__init__([param_mapper])
        ...
        ...     def indexing_func(self, pd_indexing_func):
        ...         return type(self)(
        ...             pd_indexing_func(self.df),
        ...             indexing_on_mapper(self._my_param_mapper, self.df, pd_indexing_func)
        ...         )

        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> param_mapper = pd.Series(['First', 'Second'], index=['a', 'b'])
        >>> c = C(df, param_mapper)

        >>> c.my_param_loc['First'].df
        0    1
        1    2
        Name: a, dtype: int64

        >>> c.my_param_loc['Second'].df
        0    3
        1    4
        Name: b, dtype: int64

        >>> c.my_param_loc[['First', 'First', 'Second', 'Second']].df
              a     b
        0  1  1  3  3
        1  2  2  4  4
        ```
    """

    class ParamIndexer(IndexingBase):
        def __init__(
            self,
            param_mappers: tp.Sequence[tp.Series],
            level_names: tp.Optional[tp.LevelSequence] = None,
            **kwargs,
        ) -> None:
            checks.assert_len_equal(param_names, param_mappers)

            for i, param_name in enumerate(param_names):
                level_name = level_names[i] if level_names is not None else None
                _param_loc = ParamLoc(param_mappers[i], self.indexing_func, level_name=level_name, **kwargs)
                setattr(self, f"_{param_name}_loc", _param_loc)

    for i, param_name in enumerate(param_names):

        def param_loc(self, _param_name=param_name) -> ParamLoc:
            return getattr(self, f"_{_param_name}_loc")

        param_loc.__doc__ = f"""Access a group of columns by parameter `{param_name}` using `pd.Series.loc`.
        
        Forwards this operation to each Series/DataFrame and returns a new class instance.
        """

        setattr(ParamIndexer, param_name + "_loc", property(param_loc))

    ParamIndexer.__name__ = class_name
    ParamIndexer.__qualname__ = ParamIndexer.__name__
    if module_name is not None:
        ParamIndexer.__module__ = module_name

    return ParamIndexer


hsliceT = tp.TypeVar("hsliceT", bound="hslice")


@define
class hslice(DefineMixin):
    """Hashable slice."""

    start: object = define.field()
    """Start."""

    stop: object = define.field()
    """Stop."""

    step: object = define.field()
    """Step."""

    def __init__(self, start: object = MISSING, stop: object = MISSING, step: object = MISSING) -> None:
        if start is not MISSING and stop is MISSING and step is MISSING:
            stop = start
            start, step = None, None
        else:
            if start is MISSING:
                start = None
            if stop is MISSING:
                stop = None
            if step is MISSING:
                step = None
        DefineMixin.__init__(self, start=start, stop=stop, step=step)

    @classmethod
    def from_slice(cls: tp.Type[hsliceT], slice_: slice) -> hsliceT:
        """Construct from a slice."""
        return cls(slice_.start, slice_.stop, slice_.step)

    def to_slice(self) -> slice:
        """Convert to a slice."""
        return slice(self.start, self.stop, self.step)


class IdxrBase:
    """Abstract class for resolving indices."""

    def get(self, *args, **kwargs) -> tp.Any:
        """Get indices."""
        raise NotImplementedError

    @classmethod
    def slice_indexer(
        cls,
        index: tp.Index,
        slice_: tp.Slice,
        closed_start: bool = True,
        closed_end: bool = False,
    ) -> slice:
        """Compute the slice indexer for input labels and step."""
        start = slice_.start
        end = slice_.stop
        if start is not None:
            left_start = index.get_slice_bound(start, side="left")
            right_start = index.get_slice_bound(start, side="right")
            if left_start == right_start or not closed_start:
                start = right_start
            else:
                start = left_start
        if end is not None:
            left_end = index.get_slice_bound(end, side="left")
            right_end = index.get_slice_bound(end, side="right")
            if left_end == right_end or closed_end:
                end = right_end
            else:
                end = left_end
        return slice(start, end, slice_.step)

    def check_idxs(self, idxs: tp.MaybeIndexArray, check_minus_one: bool = False) -> None:
        """Check indices after resolving them."""
        if isinstance(idxs, slice):
            if idxs.start is not None and not checks.is_int(idxs.start):
                raise TypeError("Start of a returned index slice must be an integer or None")
            if idxs.stop is not None and not checks.is_int(idxs.stop):
                raise TypeError("Stop of a returned index slice must be an integer or None")
            if idxs.step is not None and not checks.is_int(idxs.step):
                raise TypeError("Step of a returned index slice must be an integer or None")
            if check_minus_one and idxs.start == -1:
                raise ValueError("Range start index couldn't be matched")
            elif check_minus_one and idxs.stop == -1:
                raise ValueError("Range end index couldn't be matched")
        elif checks.is_int(idxs):
            if check_minus_one and idxs == -1:
                raise ValueError("Index couldn't be matched")
        elif checks.is_sequence(idxs) and not np.isscalar(idxs):
            if len(idxs) == 0:
                raise ValueError("No indices could be matched")
            if not isinstance(idxs, np.ndarray):
                raise ValueError(f"Indices must be a NumPy array, not {type(idxs)}")
            if not np.issubdtype(idxs.dtype, np.integer) or np.issubdtype(idxs.dtype, np.bool_):
                raise ValueError(f"Indices must be of integer data type, not {idxs.dtype}")
            if check_minus_one and -1 in idxs:
                raise ValueError("Some indices couldn't be matched")
            if idxs.ndim not in (1, 2):
                raise ValueError("Indices array must have either 1 or 2 dimensions")
            if idxs.ndim == 2 and idxs.shape[1] != 2:
                raise ValueError("Indices array provided as ranges must have exactly two columns")
        else:
            raise TypeError(
                f"Indices must be an integer, a slice, a NumPy array, or a tuple of two NumPy arrays, not {type(idxs)}"
            )


def normalize_idxs(idxs: tp.MaybeIndexArray, target_len: int) -> tp.Array1d:
    """Normalize indexes into a 1-dim integer array."""
    if isinstance(idxs, hslice):
        idxs = idxs.to_slice()
    if isinstance(idxs, slice):
        idxs = np.arange(target_len)[idxs]
    if checks.is_int(idxs):
        idxs = np.array([idxs])
    if idxs.ndim == 2:
        from vectorbtpro.base.merging import concat_arrays

        idxs = concat_arrays(tuple(map(lambda x: np.arange(x[0], x[1]), idxs)))
    if (idxs < 0).any():
        idxs = np.where(idxs >= 0, idxs, target_len + idxs)
    return idxs


class UniIdxr(IdxrBase):
    """Abstract class for resolving indices based on a single index."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        raise NotImplementedError

    def __invert__(self):
        def _op_func(x, index=None, freq=None):
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            idxs = np.setdiff1d(np.arange(len(index)), x)
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self)

    def __and__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            y = normalize_idxs(y, len(index))
            idxs = np.intersect1d(x, y)
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)

    def __or__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            y = normalize_idxs(y, len(index))
            idxs = np.union1d(x, y)
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)

    def __sub__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            y = normalize_idxs(y, len(index))
            idxs = np.setdiff1d(x, y)
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)

    def __xor__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            y = normalize_idxs(y, len(index))
            idxs = np.setxor1d(x, y)
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)

    def __lshift__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if not checks.is_int(y):
                raise TypeError("Second operand in __lshift__ must be an integer")
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            shifted = x - y
            idxs = shifted[shifted >= 0]
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)

    def __rshift__(self, other):
        def _op_func(x, y, index=None, freq=None):
            if not checks.is_int(y):
                raise TypeError("Second operand in __rshift__ must be an integer")
            if index is None:
                raise ValueError("Index is required")
            x = normalize_idxs(x, len(index))
            shifted = x + y
            idxs = shifted[shifted >= 0]
            self.check_idxs(idxs)
            return idxs

        return UniIdxrOp(_op_func, self, other)


@define
class UniIdxrOp(UniIdxr, DefineMixin):
    """Class for applying an operation to one or more indexers.

    Produces a single set of indices."""

    op_func: tp.Callable = define.field()
    """Operation function that takes the indices of each indexer (as `*args`), `index` (keyword argument), 
    and `freq` (keyword argument), and returns new indices."""

    idxrs: tp.Tuple[object, ...] = define.field()
    """A tuple of one or more indexers."""

    def __init__(self, op_func: tp.Callable, *idxrs) -> None:
        if len(idxrs) == 1 and checks.is_iterable(idxrs[0]):
            idxrs = idxrs[0]
        DefineMixin.__init__(self, op_func=op_func, idxrs=idxrs)

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        idxr_indices = []
        for idxr in self.idxrs:
            if isinstance(idxr, IdxrBase):
                checks.assert_instance_of(idxr, UniIdxr)
                idxr_indices.append(idxr.get(index=index, freq=freq))
            else:
                idxr_indices.append(idxr)
        return self.op_func(*idxr_indices, index=index, freq=freq)


@define
class PosIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices provided as integer positions."""

    value: tp.Union[None, tp.MaybeSequence[tp.MaybeSequence[int]], tp.Slice] = define.field()
    """One or more integer positions."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        idxs = self.value
        if checks.is_sequence(idxs) and not np.isscalar(idxs):
            idxs = np.asarray(idxs)
        if isinstance(idxs, hslice):
            idxs = idxs.to_slice()
        self.check_idxs(idxs)
        return idxs


@define
class MaskIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices provided as a mask."""

    value: tp.Union[None, tp.Sequence[bool]] = define.field()
    """Mask."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        idxs = np.flatnonzero(self.value)
        self.check_idxs(idxs)
        return idxs


@define
class LabelIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices provided as labels."""

    value: tp.Union[None, tp.MaybeSequence[tp.Label], tp.Slice] = define.field()
    """One or more labels."""

    closed_start: bool = define.field(default=True)
    """Whether slice start should be inclusive."""

    closed_end: bool = define.field(default=True)
    """Whether slice end should be inclusive."""

    level: tp.MaybeLevelSequence = define.field(default=None)
    """One or more levels."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        if index is None:
            raise ValueError("Index is required")
        if self.level is not None:
            from vectorbtpro.base.indexes import select_levels

            index = select_levels(index, self.level)

        if isinstance(self.value, (slice, hslice)):
            idxs = self.slice_indexer(
                index,
                self.value,
                closed_start=self.closed_start,
                closed_end=self.closed_end,
            )
        elif (checks.is_sequence(self.value) and not np.isscalar(self.value)) and (
            not isinstance(index, pd.MultiIndex)
            or (isinstance(index, pd.MultiIndex) and isinstance(self.value[0], tuple))
        ):
            idxs = index.get_indexer_for(self.value)
        else:
            idxs = index.get_loc(self.value)
            if isinstance(idxs, np.ndarray) and np.issubdtype(idxs.dtype, np.bool_):
                idxs = np.flatnonzero(idxs)
        self.check_idxs(idxs, check_minus_one=True)
        return idxs


@define
class DatetimeIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices provided as datetime-like objects."""

    value: tp.Union[None, tp.MaybeSequence[tp.DatetimeLike], tp.Slice] = define.field()
    """One or more datetime-like objects."""

    closed_start: bool = define.field(default=True)
    """Whether slice start should be inclusive."""

    closed_end: bool = define.field(default=False)
    """Whether slice end should be inclusive."""

    indexer_method: tp.Optional[str] = define.field(default="bfill")
    """Method for `pd.Index.get_indexer`.
    
    Allows two additional values: "before" and "after"."""

    below_to_zero: bool = define.field(default=False)
    """Whether to place 0 instead of -1 if `DatetimeIdxr.value` is below the first index."""

    above_to_len: bool = define.field(default=False)
    """Whether to place `len(index)` instead of -1 if `DatetimeIdxr.value` is above the last index."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        if index is None:
            raise ValueError("Index is required")
        index = dt.prepare_dt_index(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if not index.is_unique:
            raise ValueError("Datetime index must be unique")
        if not index.is_monotonic_increasing:
            raise ValueError("Datetime index must be monotonically increasing")

        if isinstance(self.value, (slice, hslice)):
            start = dt.try_align_dt_to_index(self.value.start, index)
            stop = dt.try_align_dt_to_index(self.value.stop, index)
            new_value = slice(start, stop, self.value.step)
            idxs = self.slice_indexer(index, new_value, closed_start=self.closed_start, closed_end=self.closed_end)
        elif checks.is_sequence(self.value) and not np.isscalar(self.value):
            new_value = dt.try_align_to_dt_index(self.value, index)
            idxs = index.get_indexer(new_value, method=self.indexer_method)
            if self.below_to_zero:
                idxs = np.where(new_value < index[0], 0, idxs)
            if self.above_to_len:
                idxs = np.where(new_value > index[-1], len(index), idxs)
        else:
            new_value = dt.try_align_dt_to_index(self.value, index)
            if new_value < index[0] and self.below_to_zero:
                idxs = 0
            elif new_value > index[-1] and self.above_to_len:
                idxs = len(index)
            else:
                if self.indexer_method is None or new_value in index:
                    idxs = index.get_loc(new_value)
                    if isinstance(idxs, np.ndarray) and np.issubdtype(idxs.dtype, np.bool_):
                        idxs = np.flatnonzero(idxs)
                else:
                    indexer_method = self.indexer_method
                    if indexer_method is not None:
                        indexer_method = indexer_method.lower()
                        if indexer_method == "before":
                            new_value = new_value - pd.Timedelta(1, "ns")
                            indexer_method = "ffill"
                        elif indexer_method == "after":
                            new_value = new_value + pd.Timedelta(1, "ns")
                            indexer_method = "bfill"
                    idxs = index.get_indexer([new_value], method=indexer_method)[0]
        self.check_idxs(idxs, check_minus_one=True)
        return idxs


@define
class DTCIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices provided as datetime-like components."""

    value: tp.Union[None, tp.MaybeSequence[tp.DTCLike], tp.Slice] = define.field()
    """One or more datetime-like components."""

    parse_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `vectorbtpro.utils.datetime_.DTC.parse`."""

    closed_start: bool = define.field(default=True)
    """Whether slice start should be inclusive."""

    closed_end: bool = define.field(default=False)
    """Whether slice end should be inclusive."""

    jitted: tp.JittedOption = define.field(default=None)
    """Jitting option passed to `vectorbtpro.utils.datetime_nb.index_matches_dtc_nb`
    and `vectorbtpro.utils.datetime_nb.index_within_dtc_range_nb`."""

    @staticmethod
    def get_dtc_namedtuple(value: tp.Optional[tp.DTCLike] = None, **parse_kwargs) -> dt.DTCNT:
        """Convert a value to a `vectorbtpro.utils.datetime_.DTCNT` instance."""
        if value is None:
            return dt.DTC().to_namedtuple()
        if isinstance(value, dt.DTC):
            return value.to_namedtuple()
        if isinstance(value, dt.DTCNT):
            return value
        return dt.DTC.parse(value, **parse_kwargs).to_namedtuple()

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        parse_kwargs = self.parse_kwargs
        if parse_kwargs is None:
            parse_kwargs = {}
        if index is None:
            raise ValueError("Index is required")
        index = dt.prepare_dt_index(index)
        ns_index = dt.to_ns(index)
        checks.assert_instance_of(index, pd.DatetimeIndex)
        if not index.is_unique:
            raise ValueError("Datetime index must be unique")
        if not index.is_monotonic_increasing:
            raise ValueError("Datetime index must be monotonically increasing")

        if isinstance(self.value, (slice, hslice)):
            if self.value.step is not None:
                raise ValueError("Step must be None")
            if self.value.start is None and self.value.step is None:
                return slice(None, None, None)
            start_dtc = self.get_dtc_namedtuple(self.value.start, **parse_kwargs)
            end_dtc = self.get_dtc_namedtuple(self.value.stop, **parse_kwargs)
            func = jit_reg.resolve_option(dt_nb.index_within_dtc_range_nb, self.jitted)
            mask = func(ns_index, start_dtc, end_dtc, closed_start=self.closed_start, closed_end=self.closed_end)
        elif checks.is_sequence(self.value) and not np.isscalar(self.value):
            func = jit_reg.resolve_option(dt_nb.index_matches_dtc_nb, self.jitted)
            dtcs = map(lambda x: self.get_dtc_namedtuple(x, **parse_kwargs), self.value)
            masks = map(lambda x: func(ns_index, x), dtcs)
            mask = functools.reduce(np.logical_or, masks)
        else:
            dtc = self.get_dtc_namedtuple(self.value, **parse_kwargs)
            func = jit_reg.resolve_option(dt_nb.index_matches_dtc_nb, self.jitted)
            mask = func(ns_index, dtc)
        return MaskIdxr(mask).get(index=index, freq=freq)


@define
class PointIdxr(UniIdxr, DefineMixin):
    """Class for resolving index points."""

    every: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Frequency either as an integer or timedelta.
    
    Gets translated into `on` array by creating a range. If integer, an index sequence from `start` to `end` 
    (exclusive) is created and 'indices' as `kind` is used. If timedelta-like, a date sequence from 
    `start` to `end` (inclusive) is created and 'labels' as `kind` is used.
    
    If `at_time` is not None and `every` and `on` are None, `every` defaults to one day."""

    normalize_every: bool = define.field(default=False)
    """Normalize start/end dates to midnight before generating date range."""

    at_time: tp.Optional[tp.TimeLike] = define.field(default=None)
    """Time of the day either as a (human-readable) string or `datetime.time`. 
    
    Every datetime in `on` gets floored to the daily frequency, while `at_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_delta`. 
    Index must be datetime-like."""

    start: tp.Optional[tp.Union[int, tp.DatetimeLike]] = define.field(default=None)
    """Start index/date.
    
    If (human-readable) string, gets converted into a datetime.
    
    If `every` is None, gets used to filter the final index array."""

    end: tp.Optional[tp.Union[int, tp.DatetimeLike]] = define.field(default=None)
    """End index/date.
    
    If (human-readable) string, gets converted into a datetime.
    
    If `every` is None, gets used to filter the final index array."""

    exact_start: bool = define.field(default=False)
    """Whether the first index should be exactly `start`.
    
    Depending on `every`, the first index picked by `pd.date_range` may happen after `start`.
    In such a case, `start` gets injected before the first index generated by `pd.date_range`."""

    on: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = define.field(default=None)
    """Index/label or a sequence of such.
    
    Gets converted into datetime format whenever possible."""

    add_delta: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Offset to be added to each in `on`.
    
    Gets converted to a proper offset/timedelta using `vectorbtpro.utils.datetime_.to_freq`."""

    kind: tp.Optional[str] = define.field(default=None)
    """Kind of data in `on`: indices or labels.
    
    If None, gets assigned to `indices` if `on` contains integer data, otherwise to `labels`.
    
    If `kind` is 'labels', `on` gets converted into indices using `pd.Index.get_indexer`. 
    Prior to this, gets its timezone aligned to the timezone of the index. If `kind` is 'indices', 
    `on` gets wrapped with NumPy."""

    indexer_method: str = define.field(default="bfill")
    """Method for `pd.Index.get_indexer`.
    
    Allows two additional values: "before" and "after"."""

    indexer_tolerance: tp.Optional[tp.Union[int, tp.TimedeltaLike, tp.IndexLike]] = define.field(default=None)
    """Tolerance for `pd.Index.get_indexer`.
    
    If `at_time` is set and `indexer_method` is neither exact nor nearest, `indexer_tolerance` 
    becomes such that the next element must be within the current day."""

    skip_not_found: bool = define.field(default=True)
    """Whether to drop indices that are -1 (not found)."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if index is None:
            raise ValueError("Index is required")
        idxs = get_index_points(index, **self.asdict())
        self.check_idxs(idxs, check_minus_one=True)
        return idxs


point_idxr_defaults = {a.name: a.default for a in PointIdxr.fields}


def get_index_points(
    index: tp.Index,
    every: tp.Optional[tp.FrequencyLike] = point_idxr_defaults["every"],
    normalize_every: bool = point_idxr_defaults["normalize_every"],
    at_time: tp.Optional[tp.TimeLike] = point_idxr_defaults["at_time"],
    start: tp.Optional[tp.Union[int, tp.DatetimeLike]] = point_idxr_defaults["start"],
    end: tp.Optional[tp.Union[int, tp.DatetimeLike]] = point_idxr_defaults["end"],
    exact_start: bool = point_idxr_defaults["exact_start"],
    on: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = point_idxr_defaults["on"],
    add_delta: tp.Optional[tp.FrequencyLike] = point_idxr_defaults["add_delta"],
    kind: tp.Optional[str] = point_idxr_defaults["kind"],
    indexer_method: str = point_idxr_defaults["indexer_method"],
    indexer_tolerance: str = point_idxr_defaults["indexer_tolerance"],
    skip_not_found: bool = point_idxr_defaults["skip_not_found"],
) -> tp.Array1d:
    """Translate indices or labels into index points.

    See `PointIdxr` for arguments.

    Usage:
        * Provide nothing to generate at the beginning:

        ```pycon
        >>> from vectorbtpro import *

        >>> index = pd.date_range("2020-01", "2020-02", freq="1d")

        >>> vbt.get_index_points(index)
        array([0])
        ```

        * Provide `every` as an integer frequency to generate index points using NumPy:

        ```pycon
        >>> # Generate a point every five rows
        >>> vbt.get_index_points(index, every=5)
        array([ 0,  5, 10, 15, 20, 25, 30])

        >>> # Generate a point every five rows starting at 6th row
        >>> vbt.get_index_points(index, every=5, start=5)
        array([ 5, 10, 15, 20, 25, 30])

        >>> # Generate a point every five rows from 6th to 16th row
        >>> vbt.get_index_points(index, every=5, start=5, end=15)
        array([ 5, 10])
        ```

        * Provide `every` as a time delta frequency to generate index points using Pandas:

        ```pycon
        >>> # Generate a point every week
        >>> vbt.get_index_points(index, every="W")
        array([ 4, 11, 18, 25])

        >>> # Generate a point every second day of the week
        >>> vbt.get_index_points(index, every="W", add_delta="2d")
        array([ 6, 13, 20, 27])

        >>> # Generate a point every week, starting at 11th row
        >>> vbt.get_index_points(index, every="W", start=10)
        array([11, 18, 25])

        >>> # Generate a point every week, starting exactly at 11th row
        >>> vbt.get_index_points(index, every="W", start=10, exact_start=True)
        array([10, 11, 18, 25])

        >>> # Generate a point every week, starting at 2020-01-10
        >>> vbt.get_index_points(index, every="W", start="2020-01-10")
        array([11, 18, 25])
        ```

        * Instead of using `every`, provide indices explicitly:

        ```pycon
        >>> # Generate one point
        >>> vbt.get_index_points(index, on="2020-01-07")
        array([6])

        >>> # Generate multiple points
        >>> vbt.get_index_points(index, on=["2020-01-07", "2020-01-14"])
        array([ 6, 13])
        ```
    """
    index = dt.prepare_dt_index(index)
    if on is not None and isinstance(on, str):
        on = dt.try_align_dt_to_index(on, index)
    if start is not None and isinstance(start, str):
        start = dt.try_align_dt_to_index(start, index)
    if end is not None and isinstance(end, str):
        end = dt.try_align_dt_to_index(end, index)
    if every is not None and not checks.is_int(every):
        every = dt.to_freq(every)

    start_used = False
    end_used = False
    if at_time is not None and every is None and on is None:
        every = pd.Timedelta(days=1)
    if every is not None:
        start_used = True
        end_used = True
        if checks.is_int(every):
            if start is None:
                start = 0
            if end is None:
                end = len(index)
            on = np.arange(start, end, every)
            kind = "indices"
        else:
            if start is None:
                start = 0
            if checks.is_int(start):
                start_date = index[start]
            else:
                start_date = start
            if end is None:
                end = len(index) - 1
            if checks.is_int(end):
                end_date = index[end]
            else:
                end_date = end
            on = dt.date_range(
                start_date,
                end_date,
                freq=every,
                tz=index.tz,
                normalize=normalize_every,
                inclusive="both",
            )
            if exact_start and on[0] > start_date:
                on = on.insert(0, start_date)
            kind = "labels"

    if kind is None:
        if on is None:
            if start is not None:
                if checks.is_int(start):
                    kind = "indices"
                else:
                    kind = "labels"
            else:
                kind = "indices"
        else:
            on = dt.prepare_dt_index(on)
            if pd.api.types.is_integer_dtype(on):
                kind = "indices"
            else:
                kind = "labels"
    checks.assert_in(kind, ("indices", "labels"))
    if on is None:
        if start is not None:
            on = start
            start_used = True
        else:
            if kind.lower() in ("labels",):
                on = index
            else:
                on = np.arange(len(index))
    on = dt.prepare_dt_index(on)

    if at_time is not None:
        checks.assert_instance_of(on, pd.DatetimeIndex)
        on = on.floor("D")
        add_time_delta = dt.time_to_timedelta(at_time)
        if indexer_tolerance is None:
            indexer_method = indexer_method.lower()
            if indexer_method in ("pad", "ffill"):
                indexer_tolerance = add_time_delta
            elif indexer_method in ("backfill", "bfill"):
                indexer_tolerance = pd.Timedelta(days=1) - pd.Timedelta(1, "ns") - add_time_delta
        if add_delta is None:
            add_delta = add_time_delta
        else:
            add_delta += add_time_delta

    if add_delta is not None:
        on += dt.to_freq(add_delta)

    if kind.lower() == "labels":
        on = dt.try_align_to_dt_index(on, index)
        indexer_method = indexer_method.lower()
        if indexer_method == "before":
            on = on - pd.Timedelta(1, "ns")
            indexer_method = "ffill"
        elif indexer_method == "after":
            on = on + pd.Timedelta(1, "ns")
            indexer_method = "bfill"
        index_points = index.get_indexer(on, method=indexer_method, tolerance=indexer_tolerance)
    else:
        index_points = np.asarray(on)

    if start is not None and not start_used:
        if not checks.is_int(start):
            start = index.get_indexer([start], method="bfill").item(0)
        index_points = index_points[index_points >= start]
    if end is not None and not end_used:
        if not checks.is_int(end):
            end = index.get_indexer([end], method="ffill").item(0)
            index_points = index_points[index_points <= end]
        else:
            index_points = index_points[index_points < end]

    if skip_not_found:
        index_points = index_points[index_points != -1]

    return index_points


@define
class RangeIdxr(UniIdxr, DefineMixin):
    """Class for resolving index ranges."""

    every: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Frequency either as an integer or timedelta.

    Gets translated into `start` and `end` arrays by creating a range. If integer, an index sequence from `start` 
    to `end` (exclusive) is created and 'indices' as `kind` is used. If timedelta-like, a date sequence 
    from `start` to `end` (inclusive) is created and 'bounds' as `kind` is used. 

    If `start_time` and `end_time` are not None and `every`, `start`, and `end` are None, 
    `every` defaults to one day."""

    normalize_every: bool = define.field(default=False)
    """Normalize start/end dates to midnight before generating date range."""

    split_every: bool = define.field(default=True)
    """Whether to split the sequence generated using `every` into `start` and `end` arrays.

    After creation, and if `split_every` is True, an index range is created from each pair of elements in 
    the generated sequence. Otherwise, the entire sequence is assigned to `start` and `end`, and only time 
    and delta instructions can be used to further differentiate between them.

    Forced to False if `every`, `start_time`, and `end_time` are not None and `fixed_start` is False."""

    start_time: tp.Optional[tp.TimeLike] = define.field(default=None)
    """Start time of the day either as a (human-readable) string or `datetime.time`. 

    Every datetime in `start` gets floored to the daily frequency, while `start_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_start_delta`. 
    Index must be datetime-like."""

    end_time: tp.Optional[tp.TimeLike] = define.field(default=None)
    """End time of the day either as a (human-readable) string or `datetime.time`. 

    Every datetime in `end` gets floored to the daily frequency, while `end_time` gets converted into 
    a timedelta using `vectorbtpro.utils.datetime_.time_to_timedelta` and added to `add_end_delta`. 
    Index must be datetime-like."""

    lookback_period: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Lookback period either as an integer or offset.

    If `lookback_period` is set, `start` becomes `end-lookback_period`. If `every` is not None, 
    the sequence is generated from `start+lookback_period` to `end` and then assigned to `end`.

    If string, gets converted to a proper offset/timedelta using `vectorbtpro.utils.datetime_.to_freq`.
    If integer, gets multiplied by the frequency of the index if the index is not integer."""

    start: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = define.field(default=None)
    """Start index/label or a sequence of such.

    Gets converted into datetime format whenever possible.

    Gets broadcasted together with `end`."""

    end: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = define.field(default=None)
    """End index/label or a sequence of such.

    Gets converted into datetime format whenever possible.

    Gets broadcasted together with `start`."""

    exact_start: bool = define.field(default=False)
    """Whether the first index in the `start` array should be exactly `start`.

    Depending on `every`, the first index picked by `pd.date_range` may happen after `start`.
    In such a case, `start` gets injected before the first index generated by `pd.date_range`.

    Cannot be used together with `lookback_period`."""

    fixed_start: bool = define.field(default=False)
    """Whether all indices in the `start` array should be exactly `start`.

    Works only together with `every`.

    Cannot be used together with `lookback_period`."""

    closed_start: bool = define.field(default=True)
    """Whether `start` should be inclusive."""

    closed_end: bool = define.field(default=False)
    """Whether `end` should be inclusive."""

    add_start_delta: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Offset to be added to each in `start`.

    If string, gets converted to a proper offset/timedelta using `vectorbtpro.utils.datetime_.to_freq`."""

    add_end_delta: tp.Optional[tp.FrequencyLike] = define.field(default=None)
    """Offset to be added to each in `end`.

    If string, gets converted to a proper offset/timedelta using `vectorbtpro.utils.datetime_.to_freq`."""

    kind: tp.Optional[str] = define.field(default=None)
    """Kind of data in `on`: indices, labels or bounds.

    If None, gets assigned to `indices` if `start` and `end` contain integer data, to `bounds`
    if `start`, `end`, and index are datetime-like, otherwise to `labels`.

    If `kind` is 'labels', `start` and `end` get converted into indices using `pd.Index.get_indexer`. 
    Prior to this, get their timezone aligned to the timezone of the index. If `kind` is 'indices', 
    `start` and `end` get wrapped with NumPy. If kind` is 'bounds', 
    `vectorbtpro.base.resampling.base.Resampler.map_bounds_to_source_ranges` is used."""

    skip_not_found: bool = define.field(default=True)
    """Whether to drop indices that are -1 (not found)."""

    jitted: tp.JittedOption = define.field(default=None)
    """Jitting option passed to `vectorbtpro.base.resampling.base.Resampler.map_bounds_to_source_ranges`."""

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if index is None:
            raise ValueError("Index is required")

        from vectorbtpro.base.merging import column_stack_arrays

        start_idxs, end_idxs = get_index_ranges(index, index_freq=freq, **self.asdict())
        idxs = column_stack_arrays((start_idxs, end_idxs))
        self.check_idxs(idxs, check_minus_one=True)
        return idxs


range_idxr_defaults = {a.name: a.default for a in RangeIdxr.fields}


def get_index_ranges(
    index: tp.Index,
    index_freq: tp.Optional[tp.FrequencyLike] = None,
    every: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["every"],
    normalize_every: bool = range_idxr_defaults["normalize_every"],
    split_every: bool = range_idxr_defaults["split_every"],
    start_time: tp.Optional[tp.TimeLike] = range_idxr_defaults["start_time"],
    end_time: tp.Optional[tp.TimeLike] = range_idxr_defaults["end_time"],
    lookback_period: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["lookback_period"],
    start: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = range_idxr_defaults["start"],
    end: tp.Optional[tp.Union[int, tp.DatetimeLike, tp.IndexLike]] = range_idxr_defaults["end"],
    exact_start: bool = range_idxr_defaults["exact_start"],
    fixed_start: bool = range_idxr_defaults["fixed_start"],
    closed_start: bool = range_idxr_defaults["closed_start"],
    closed_end: bool = range_idxr_defaults["closed_end"],
    add_start_delta: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["add_start_delta"],
    add_end_delta: tp.Optional[tp.FrequencyLike] = range_idxr_defaults["add_end_delta"],
    kind: tp.Optional[str] = range_idxr_defaults["kind"],
    skip_not_found: bool = range_idxr_defaults["skip_not_found"],
    jitted: tp.JittedOption = range_idxr_defaults["jitted"],
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Translate indices, labels, or bounds into index ranges.

    See `RangeIdxr` for arguments.

    Usage:
        * Provide nothing to generate one largest index range:

        ```pycon
        >>> from vectorbtpro import *

        >>> index = pd.date_range("2020-01", "2020-02", freq="1d")

        >>> np.column_stack(vbt.get_index_ranges(index))
        array([[ 0, 32]])
        ```

        * Provide `every` as an integer frequency to generate index ranges using NumPy:

        ```pycon
        >>> # Generate a range every five rows
        >>> np.column_stack(vbt.get_index_ranges(index, every=5))
        array([[ 0,  5],
               [ 5, 10],
               [10, 15],
               [15, 20],
               [20, 25],
               [25, 30]])

        >>> # Generate a range every five rows, starting at 6th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every=5,
        ...     start=5
        ... ))
        array([[ 5, 10],
               [10, 15],
               [15, 20],
               [20, 25],
               [25, 30]])

        >>> # Generate a range every five rows from 6th to 16th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every=5,
        ...     start=5,
        ...     end=15
        ... ))
        array([[ 5, 10],
               [10, 15]])
        ```

        * Provide `every` as a time delta frequency to generate index ranges using Pandas:

        ```pycon
        >>> # Generate a range every week
        >>> np.column_stack(vbt.get_index_ranges(index, every="W"))
        array([[ 4, 11],
               [11, 18],
               [18, 25]])

        >>> # Generate a range every second day of the week
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     add_start_delta="2d"
        ... ))
        array([[ 6, 11],
               [13, 18],
               [20, 25]])

        >>> # Generate a range every week, starting at 11th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start=10
        ... ))
        array([[11, 18],
               [18, 25]])

        >>> # Generate a range every week, starting exactly at 11th row
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start=10,
        ...     exact_start=True
        ... ))
        array([[10, 11],
               [11, 18],
               [18, 25]])

        >>> # Generate a range every week, starting at 2020-01-10
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start="2020-01-10"
        ... ))
        array([[11, 18],
               [18, 25]])

        >>> # Generate a range every week, each starting at 2020-01-10
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start="2020-01-10",
        ...     fixed_start=True
        ... ))
        array([[11, 18],
               [11, 25]])

        >>> # Generate an expanding range that increments by week
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     start=0,
        ...     exact_start=True,
        ...     fixed_start=True
        ... ))
        array([[ 0,  4],
               [ 0, 11],
               [ 0, 18],
               [ 0, 25]])
        ```

        * Use a look-back period (instead of an end index):

        ```pycon
        >>> # Generate a range every week, looking 5 days back
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     lookback_period=5
        ... ))
        array([[ 6, 11],
               [13, 18],
               [20, 25]])

        >>> # Generate a range every week, looking 2 weeks back
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     every="W",
        ...     lookback_period="2W"
        ... ))
        array([[ 0, 11],
               [ 4, 18],
               [11, 25]])
        ```

        * Instead of using `every`, provide start and end indices explicitly:

        ```pycon
        >>> # Generate one range
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start="2020-01-01",
        ...     end="2020-01-07"
        ... ))
        array([[0, 6]])

        >>> # Generate ranges between multiple dates
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start=["2020-01-01", "2020-01-07"],
        ...     end=["2020-01-07", "2020-01-14"]
        ... ))
        array([[ 0,  6],
               [ 6, 13]])

        >>> # Generate ranges with a fixed start
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start="2020-01-01",
        ...     end=["2020-01-07", "2020-01-14"]
        ... ))
        array([[ 0,  6],
               [ 0, 13]])
        ```

        * Use `closed_start` and `closed_end` to exclude any of the bounds:

        ```pycon
        >>> # Generate ranges between multiple dates
        >>> # by excluding the start date and including the end date
        >>> np.column_stack(vbt.get_index_ranges(
        ...     index,
        ...     start=["2020-01-01", "2020-01-07"],
        ...     end=["2020-01-07", "2020-01-14"],
        ...     closed_start=False,
        ...     closed_end=True
        ... ))
        array([[ 1,  7],
               [ 7, 14]])
        ```
    """
    from vectorbtpro.base.indexes import repeat_index
    from vectorbtpro.base.resampling.base import Resampler

    index = dt.prepare_dt_index(index)
    if isinstance(index, pd.DatetimeIndex):
        if start is not None:
            start = dt.try_align_to_dt_index(start, index)
            if isinstance(start, pd.DatetimeIndex):
                start = start.tz_localize(None)
        if end is not None:
            end = dt.try_align_to_dt_index(end, index)
            if isinstance(end, pd.DatetimeIndex):
                end = end.tz_localize(None)
        naive_index = index.tz_localize(None)
    else:
        if start is not None:
            if not isinstance(start, pd.Index):
                try:
                    start = pd.Index(start)
                except Exception as e:
                    start = pd.Index([start])
        if end is not None:
            if not isinstance(end, pd.Index):
                try:
                    end = pd.Index(end)
                except Exception as e:
                    end = pd.Index([end])
        naive_index = index
    if every is not None and not checks.is_int(every):
        every = dt.to_freq(every)
    if lookback_period is not None and not checks.is_int(lookback_period):
        lookback_period = dt.to_freq(lookback_period)
    if fixed_start and lookback_period is not None:
        raise ValueError("Cannot use fixed_start and lookback_period together")
    if exact_start and lookback_period is not None:
        raise ValueError("Cannot use exact_start and lookback_period together")

    if start_time is not None or end_time is not None:
        if every is None and start is None and end is None:
            every = pd.Timedelta(days=1)
    if every is not None:
        if not fixed_start:
            if start_time is None and end_time is not None:
                start_time = time(0, 0, 0, 0)
                closed_start = True
            if start_time is not None and end_time is None:
                end_time = time(0, 0, 0, 0)
                closed_end = False
        if start_time is not None and end_time is not None and not fixed_start:
            split_every = False

        if checks.is_int(every):
            if start is None:
                start = 0
            else:
                start = start[0]
            if end is None:
                end = len(naive_index)
            else:
                end = end[-1]
            if closed_end:
                end -= 1
            if lookback_period is None:
                new_index = np.arange(start, end + 1, every)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = np.full(len(new_index) - 1, new_index[0])
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]
            else:
                end = np.arange(start + lookback_period, end + 1, every)
                start = end - lookback_period
            kind = "indices"
            lookback_period = None
        else:
            if start is None:
                start = 0
            else:
                start = start[0]
            if checks.is_int(start):
                start_date = naive_index[start]
            else:
                start_date = start
            if end is None:
                end = len(naive_index) - 1
            else:
                end = end[-1]
            if checks.is_int(end):
                end_date = naive_index[end]
            else:
                end_date = end
            if lookback_period is None:
                new_index = dt.date_range(
                    start_date,
                    end_date,
                    freq=every,
                    normalize=normalize_every,
                    inclusive="both",
                )
                if exact_start and new_index[0] > start_date:
                    new_index = new_index.insert(0, start_date)
                if not split_every:
                    start = end = new_index
                else:
                    if fixed_start:
                        start = repeat_index(new_index[[0]], len(new_index) - 1)
                    else:
                        start = new_index[:-1]
                    end = new_index[1:]
            else:
                if checks.is_int(lookback_period):
                    lookback_period *= dt.infer_index_freq(naive_index, freq=index_freq)
                if isinstance(lookback_period, BaseOffset):
                    end = dt.date_range(
                        start_date,
                        end_date,
                        freq=every,
                        normalize=normalize_every,
                        inclusive="both",
                    )
                    start = end - lookback_period
                    start_mask = start >= start_date
                    start = start[start_mask]
                    end = end[start_mask]
                else:
                    end = dt.date_range(
                        start_date + lookback_period,
                        end_date,
                        freq=every,
                        normalize=normalize_every,
                        inclusive="both",
                    )
                    start = end - lookback_period
            kind = "bounds"
            lookback_period = None

    if kind is None:
        if start is None and end is None:
            kind = "indices"
        else:
            if start is not None:
                ref_index = start
            if end is not None:
                ref_index = end
            if pd.api.types.is_integer_dtype(ref_index):
                kind = "indices"
            elif isinstance(ref_index, pd.DatetimeIndex) and isinstance(naive_index, pd.DatetimeIndex):
                kind = "bounds"
            else:
                kind = "labels"
    checks.assert_in(kind, ("indices", "labels", "bounds"))
    if end is None:
        if kind.lower() in ("labels", "bounds"):
            end = pd.Index([naive_index[-1]])
        else:
            end = pd.Index([len(naive_index)])
    if start is not None and lookback_period is not None:
        raise ValueError("Cannot use start and lookback_period together")
    if start is None:
        if lookback_period is None:
            if kind.lower() in ("labels", "bounds"):
                start = pd.Index([naive_index[0]])
            else:
                start = pd.Index([0])
        else:
            if checks.is_int(lookback_period) and not pd.api.types.is_integer_dtype(end):
                lookback_period *= dt.infer_index_freq(naive_index, freq=index_freq)
            start = end - lookback_period
    if len(start) == 1 and len(end) > 1:
        start = repeat_index(start, len(end))
    elif len(start) > 1 and len(end) == 1:
        end = repeat_index(end, len(start))
    checks.assert_len_equal(start, end)

    if start_time is not None:
        checks.assert_instance_of(start, pd.DatetimeIndex)
        start = start.floor("D")
        add_start_time_delta = dt.time_to_timedelta(start_time)
        if add_start_delta is None:
            add_start_delta = add_start_time_delta
        else:
            add_start_delta += add_start_time_delta
    else:
        add_start_time_delta = None
    if end_time is not None:
        checks.assert_instance_of(end, pd.DatetimeIndex)
        end = end.floor("D")
        add_end_time_delta = dt.time_to_timedelta(end_time)
        if add_start_time_delta is not None:
            if add_end_time_delta < add_start_delta:
                add_end_time_delta += pd.Timedelta(days=1)
        if add_end_delta is None:
            add_end_delta = add_end_time_delta
        else:
            add_end_delta += add_end_time_delta

    if add_start_delta is not None:
        start += dt.to_freq(add_start_delta)
    if add_end_delta is not None:
        end += dt.to_freq(add_end_delta)

    if kind.lower() == "bounds":
        range_starts, range_ends = Resampler.map_bounds_to_source_ranges(
            source_index=naive_index.values,
            target_lbound_index=start.values,
            target_rbound_index=end.values,
            closed_lbound=closed_start,
            closed_rbound=closed_end,
            skip_not_found=skip_not_found,
            jitted=jitted,
        )
    else:
        if kind.lower() == "labels":
            range_starts = np.empty(len(start), dtype=np.int_)
            range_ends = np.empty(len(end), dtype=np.int_)
            range_index = pd.Series(np.arange(len(naive_index)), index=naive_index)
            for i in range(len(range_starts)):
                selected_range = range_index[start[i] : end[i]]
                if len(selected_range) > 0 and not closed_start and selected_range.index[0] == start[i]:
                    selected_range = selected_range.iloc[1:]
                if len(selected_range) > 0 and not closed_end and selected_range.index[-1] == end[i]:
                    selected_range = selected_range.iloc[:-1]
                if len(selected_range) > 0:
                    range_starts[i] = selected_range.iloc[0]
                    range_ends[i] = selected_range.iloc[-1]
                else:
                    range_starts[i] = -1
                    range_ends[i] = -1
        else:
            if not closed_start:
                start = start + 1
            if closed_end:
                end = end + 1
            range_starts = np.asarray(start)
            range_ends = np.asarray(end)
        if skip_not_found:
            valid_mask = (range_starts != -1) & (range_ends != -1)
            range_starts = range_starts[valid_mask]
            range_ends = range_ends[valid_mask]

    if np.any(range_starts >= range_ends):
        raise ValueError("Some start indices are equal to or higher than end indices")

    return range_starts, range_ends


@define
class AutoIdxr(UniIdxr, DefineMixin):
    """Class for resolving indices, datetime-like objects, frequency-like objects, and labels for one axis."""

    value: tp.Union[
        None,
        tp.PosSel,
        tp.LabelSel,
        tp.MaybeSequence[tp.MaybeSequence[int]],
        tp.MaybeSequence[tp.Label],
        tp.MaybeSequence[tp.DatetimeLike],
        tp.MaybeSequence[tp.DTCLike],
        tp.FrequencyLike,
        tp.Slice,
    ] = define.field()
    """One or more integer indices, datetime-like objects, frequency-like objects, or labels.
    
    Can also be an instance of `vectorbtpro.utils.selection.PosSel` holding position(s)
    and `vectorbtpro.utils.selection.LabelSel` holding label(s)."""

    closed_start: bool = define.optional_field()
    """Whether slice start should be inclusive."""

    closed_end: bool = define.optional_field()
    """Whether slice end should be inclusive."""

    indexer_method: tp.Optional[str] = define.optional_field()
    """Method for `pd.Index.get_indexer`."""

    below_to_zero: bool = define.optional_field()
    """Whether to place 0 instead of -1 if `AutoIdxr.value` is below the first index."""

    above_to_len: bool = define.optional_field()
    """Whether to place `len(index)` instead of -1 if `AutoIdxr.value` is above the last index."""

    level: tp.MaybeLevelSequence = define.field(default=None)
    """One or more levels.

    If `level` is not None and `kind` is None, `kind` becomes "labels"."""

    kind: tp.Optional[str] = define.field(default=None)
    """Kind of value.

    Allowed are

    * "position(s)" for `PosIdxr`
    * "mask" for `MaskIdxr`
    * "label(s)" for `LabelIdxr`
    * "datetime" for `DatetimeIdxr`
    * "dtc": for `DTCIdxr`
    * "frequency" for `PointIdxr`

    If None, will (try to) determine automatically based on the type of indices."""

    idxr_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to the selected indexer."""

    def __init__(self, *args, **kwargs) -> None:
        idxr_kwargs = kwargs.pop("idxr_kwargs", None)
        if idxr_kwargs is None:
            idxr_kwargs = {}
        else:
            idxr_kwargs = dict(idxr_kwargs)
        builtin_keys = {a.name for a in self.fields}
        for k in list(kwargs.keys()):
            if k not in builtin_keys:
                idxr_kwargs[k] = kwargs.pop(k)
        DefineMixin.__init__(self, *args, idxr_kwargs=idxr_kwargs, **kwargs)

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
    ) -> tp.MaybeIndexArray:
        if self.value is None:
            return slice(None, None, None)
        value = self.value
        kind = self.kind
        if self.level is not None:
            from vectorbtpro.base.indexes import select_levels

            if index is None:
                raise ValueError("Index is required")
            index = select_levels(index, self.level)
            if kind is None:
                kind = "labels"
        if self.idxr_kwargs is None:
            idxr_kwargs = self.idxr_kwargs
        else:
            idxr_kwargs = None
        if idxr_kwargs is None:
            idxr_kwargs = {}

        def _dtc_check_func(dtc):
            return (
                not dtc.has_full_datetime()
                and self.indexer_method in (MISSING, None)
                and self.below_to_zero is MISSING
                and self.above_to_len is MISSING
            )

        if kind is None:
            if isinstance(value, PosSel):
                kind = "positions"
                value = value.value
            elif isinstance(value, LabelSel):
                kind = "labels"
                value = value.value
            elif isinstance(value, (slice, hslice)):
                if checks.is_int(value.start) or checks.is_int(value.stop):
                    kind = "positions"
                elif value.start is None and value.stop is None:
                    kind = "positions"
                else:
                    if index is None:
                        raise ValueError("Index is required")
                    if isinstance(index, pd.DatetimeIndex):
                        if dt.DTC.is_parsable(value.start, check_func=_dtc_check_func) or dt.DTC.is_parsable(
                            value.stop, check_func=_dtc_check_func
                        ):
                            kind = "dtc"
                        else:
                            kind = "datetime"
                    else:
                        kind = "labels"
            elif (checks.is_sequence(value) and not np.isscalar(value)) and (
                index is None
                or (
                    not isinstance(index, pd.MultiIndex)
                    or (isinstance(index, pd.MultiIndex) and isinstance(value[0], tuple))
                )
            ):
                if checks.is_bool(value[0]):
                    kind = "mask"
                elif checks.is_int(value[0]):
                    kind = "positions"
                elif (
                    (index is None or not isinstance(index, pd.MultiIndex) or not isinstance(value[0], tuple))
                    and checks.is_sequence(value[0])
                    and len(value[0]) == 2
                    and checks.is_int(value[0][0])
                    and checks.is_int(value[0][1])
                ):
                    kind = "positions"
                else:
                    if index is None:
                        raise ValueError("Index is required")
                    elif isinstance(index, pd.DatetimeIndex):
                        if dt.DTC.is_parsable(value[0], check_func=_dtc_check_func):
                            kind = "dtc"
                        else:
                            kind = "datetime"
                    else:
                        kind = "labels"
            else:
                if checks.is_bool(value):
                    kind = "mask"
                elif checks.is_int(value):
                    kind = "positions"
                else:
                    if index is None:
                        raise ValueError("Index is required")
                    if isinstance(index, pd.DatetimeIndex):
                        if dt.DTC.is_parsable(value, check_func=_dtc_check_func):
                            kind = "dtc"
                        elif isinstance(value, str):
                            try:
                                if not value.isupper() and not value.islower():
                                    raise Exception  # "2020" shouldn't be a frequency
                                _ = dt.to_freq(value)
                                kind = "frequency"
                            except Exception as e:
                                try:
                                    _ = dt.to_timestamp(value)
                                    kind = "datetime"
                                except Exception as e:
                                    raise ValueError(f"'{value}' is neither a frequency nor a datetime")
                        elif checks.is_frequency(value):
                            kind = "frequency"
                        else:
                            kind = "datetime"
                    else:
                        kind = "labels"

        def _expand_target_kwargs(target_cls, **target_kwargs):
            source_arg_names = {a.name for a in self.fields if a.default is MISSING}
            target_arg_names = {a.name for a in target_cls.fields}
            for arg_name in source_arg_names:
                if arg_name in target_arg_names:
                    arg_value = getattr(self, arg_name)
                    if arg_value is not MISSING:
                        target_kwargs[arg_name] = arg_value
            return target_kwargs

        if kind.lower() in ("position", "positions"):
            idx = PosIdxr(value, **_expand_target_kwargs(PosIdxr, **idxr_kwargs))
        elif kind.lower() == "mask":
            idx = MaskIdxr(value, **_expand_target_kwargs(MaskIdxr, **idxr_kwargs))
        elif kind.lower() in ("label", "labels"):
            idx = LabelIdxr(value, **_expand_target_kwargs(LabelIdxr, **idxr_kwargs))
        elif kind.lower() == "datetime":
            idx = DatetimeIdxr(value, **_expand_target_kwargs(DatetimeIdxr, **idxr_kwargs))
        elif kind.lower() == "dtc":
            idx = DTCIdxr(value, **_expand_target_kwargs(DTCIdxr, **idxr_kwargs))
        elif kind.lower() == "frequency":
            idx = PointIdxr(every=value, **_expand_target_kwargs(PointIdxr, **idxr_kwargs))
        else:
            raise ValueError(f"Invalid option kind='{kind}'")
        return idx.get(index=index, freq=freq)


@define
class RowIdxr(IdxrBase, DefineMixin):
    """Class for resolving row indices."""

    idxr: object = define.field()
    """Indexer.
    
    Can be an instance of `UniIdxr`, a custom template, or a value to be wrapped with `AutoIdxr`."""

    idxr_kwargs: tp.KwargsLike = define.field()
    """Keyword arguments passed to `AutoIdxr`."""

    def __init__(self, idxr: object, **idxr_kwargs) -> None:
        DefineMixin.__init__(self, idxr=idxr, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.MaybeIndexArray:
        idxr = self.idxr
        if isinstance(idxr, CustomTemplate):
            _template_context = merge_dicts(dict(index=index, freq=freq), template_context)
            idxr = idxr.substitute(_template_context, eval_id="idxr")
        if not isinstance(idxr, UniIdxr):
            if isinstance(idxr, IdxrBase):
                raise TypeError(f"Indexer of {type(self)} must be an instance of UniIdxr")
            idxr = AutoIdxr(idxr, **self.idxr_kwargs)
        return idxr.get(index=index, freq=freq)


@define
class ColIdxr(IdxrBase, DefineMixin):
    """Class for resolving column indices."""

    idxr: object = define.field()
    """Indexer.
        
    Can be an instance of `UniIdxr`, a custom template, or a value to be wrapped with `AutoIdxr`."""

    idxr_kwargs: tp.KwargsLike = define.field()
    """Keyword arguments passed to `AutoIdxr`."""

    def __init__(self, idxr: object, **idxr_kwargs) -> None:
        DefineMixin.__init__(self, idxr=idxr, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        columns: tp.Optional[tp.Index] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.MaybeIndexArray:
        idxr = self.idxr
        if isinstance(idxr, CustomTemplate):
            _template_context = merge_dicts(dict(columns=columns), template_context)
            idxr = idxr.substitute(_template_context, eval_id="idxr")
        if not isinstance(idxr, UniIdxr):
            if isinstance(idxr, IdxrBase):
                raise TypeError(f"Indexer of {type(self)} must be an instance of UniIdxr")
            idxr = AutoIdxr(idxr, **self.idxr_kwargs)
        return idxr.get(index=columns)


@define
class Idxr(IdxrBase, DefineMixin):
    """Class for resolving indices."""

    idxrs: tp.Tuple[object, ...] = define.field()
    """A tuple of one or more indexers.
    
    If one indexer is provided, can be an instance of `RowIdxr` or `ColIdxr`, 
    a custom template, or a value to wrapped with `RowIdxr`.
    
    If two indexers are provided, can be an instance of `RowIdxr` and `ColIdxr` respectively,
    or a value to wrapped with `RowIdxr` and `ColIdxr` respectively."""

    idxr_kwargs: tp.KwargsLike = define.field()
    """Keyword arguments passed to `RowIdxr` and `ColIdxr`."""

    def __init__(self, *idxrs: object, **idxr_kwargs) -> None:
        DefineMixin.__init__(self, idxrs=idxrs, idxr_kwargs=hdict(idxr_kwargs))

    def get(
        self,
        index: tp.Optional[tp.Index] = None,
        columns: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Tuple[tp.MaybeIndexArray, tp.MaybeIndexArray]:
        if len(self.idxrs) == 0:
            raise ValueError("Must provide at least one indexer")
        elif len(self.idxrs) == 1:
            idxr = self.idxrs[0]
            if isinstance(idxr, CustomTemplate):
                _template_context = merge_dicts(dict(index=index, columns=columns, freq=freq), template_context)
                idxr = idxr.substitute(_template_context, eval_id="idxr")
                if isinstance(idxr, tuple):
                    return type(self)(*idxr).get(
                        index=index,
                        columns=columns,
                        freq=freq,
                        template_context=template_context,
                    )
                return type(self)(idxr).get(
                    index=index,
                    columns=columns,
                    freq=freq,
                    template_context=template_context,
                )
            if isinstance(idxr, ColIdxr):
                row_idxr = None
                col_idxr = idxr
            else:
                row_idxr = idxr
                col_idxr = None
        elif len(self.idxrs) == 2:
            row_idxr = self.idxrs[0]
            col_idxr = self.idxrs[1]
        else:
            raise ValueError("Must provide at most two indexers")
        if not isinstance(row_idxr, RowIdxr):
            if isinstance(row_idxr, (ColIdxr, Idxr)):
                raise TypeError(f"Indexer {type(row_idxr)} not supported as a row indexer")
            row_idxr = RowIdxr(row_idxr, **self.idxr_kwargs)
        row_idxs = row_idxr.get(index=index, freq=freq, template_context=template_context)
        if not isinstance(col_idxr, ColIdxr):
            if isinstance(col_idxr, (RowIdxr, Idxr)):
                raise TypeError(f"Indexer {type(col_idxr)} not supported as a column indexer")
            col_idxr = ColIdxr(col_idxr, **self.idxr_kwargs)
        col_idxs = col_idxr.get(columns=columns, template_context=template_context)
        return row_idxs, col_idxs


def get_idxs(
    idxr: object,
    index: tp.Optional[tp.Index] = None,
    columns: tp.Optional[tp.Index] = None,
    freq: tp.Optional[tp.FrequencyLike] = None,
    template_context: tp.KwargsLike = None,
    **kwargs,
) -> tp.Tuple[tp.MaybeIndexArray, tp.MaybeIndexArray]:
    """Translate indexer to row and column indices.

    If `idxr` is not an indexer class, wraps it with `Idxr`.

    Keyword arguments are passed when constructing a new `Idxr`."""
    if not isinstance(idxr, Idxr):
        idxr = Idxr(idxr, **kwargs)
    return idxr.get(index=index, columns=columns, freq=freq, template_context=template_context)


class index_dict(pdict):
    """Dict that contains indexer objects as keys and values to be set as values.

    Each indexer object must be hashable. To make a slice hashable, use `hslice`.
    To make an array hashable, convert it into a tuple.

    To set a default value, use the `_def` key (case-sensitive!)."""

    pass


IdxSetterT = tp.TypeVar("IdxSetterT", bound="IdxSetter")


@define
class IdxSetter(DefineMixin):
    """Class for setting values based on indexing."""

    idx_items: tp.List[tp.Tuple[object, tp.ArrayLike]] = define.field()
    """Items where the first element is an indexer and the second element is a value to be set."""

    @classmethod
    def set_row_idxs(cls, arr: tp.Array, idxs: tp.MaybeIndexArray, v: tp.Any) -> None:
        """Set row indices in an array."""
        from vectorbtpro.base.reshaping import broadcast_array_to

        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        single_v = v.size == 1 or (v.ndim == 2 and v.shape[0] == 1)
        if arr.ndim == 2:
            single_row = not isinstance(idxs, slice) and (np.isscalar(idxs) or idxs.size == 1)
            if not single_row:
                if v.ndim == 1 and v.size > 1:
                    v = v[:, None]

        if isinstance(idxs, np.ndarray) and idxs.ndim == 2:
            if not single_v:
                if arr.ndim == 2:
                    v = broadcast_array_to(v, (len(idxs), arr.shape[1]))
                else:
                    v = broadcast_array_to(v, (len(idxs),))
            for i in range(len(idxs)):
                _slice = slice(idxs[i, 0], idxs[i, 1])
                if not single_v:
                    cls.set_row_idxs(arr, _slice, v[[i]])
                else:
                    cls.set_row_idxs(arr, _slice, v)
        else:
            arr[idxs] = v

    @classmethod
    def set_col_idxs(cls, arr: tp.Array, idxs: tp.MaybeIndexArray, v: tp.Any) -> None:
        """Set column indices in an array."""
        from vectorbtpro.base.reshaping import broadcast_array_to

        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        single_v = v.size == 1 or (v.ndim == 2 and v.shape[1] == 1)

        if isinstance(idxs, np.ndarray) and idxs.ndim == 2:
            if not single_v:
                v = broadcast_array_to(v, (arr.shape[0], len(idxs)))
            for j in range(len(idxs)):
                _slice = slice(idxs[j, 0], idxs[j, 1])
                if not single_v:
                    cls.set_col_idxs(arr, _slice, v[:, [j]])
                else:
                    cls.set_col_idxs(arr, _slice, v)
        else:
            arr[:, idxs] = v

    @classmethod
    def set_row_and_col_idxs(
        cls,
        arr: tp.Array,
        row_idxs: tp.MaybeIndexArray,
        col_idxs: tp.MaybeIndexArray,
        v: tp.Any,
    ) -> None:
        """Set row and column indices in an array."""
        from vectorbtpro.base.reshaping import broadcast_array_to

        if not isinstance(v, np.ndarray):
            v = np.asarray(v)
        single_v = v.size == 1
        if (
            isinstance(row_idxs, np.ndarray)
            and row_idxs.ndim == 2
            and isinstance(col_idxs, np.ndarray)
            and col_idxs.ndim == 2
        ):
            if not single_v:
                v = broadcast_array_to(v, (len(row_idxs), len(col_idxs)))
            for i in range(len(row_idxs)):
                for j in range(len(col_idxs)):
                    row_slice = slice(row_idxs[i, 0], row_idxs[i, 1])
                    col_slice = slice(col_idxs[j, 0], col_idxs[j, 1])
                    if not single_v:
                        cls.set_row_and_col_idxs(arr, row_slice, col_slice, v[i, j])
                    else:
                        cls.set_row_and_col_idxs(arr, row_slice, col_slice, v)
        elif isinstance(row_idxs, np.ndarray) and row_idxs.ndim == 2:
            if not single_v:
                if isinstance(col_idxs, slice):
                    col_idxs = np.arange(arr.shape[1])[col_idxs]
                v = broadcast_array_to(v, (len(row_idxs), len(col_idxs)))
            for i in range(len(row_idxs)):
                row_slice = slice(row_idxs[i, 0], row_idxs[i, 1])
                if not single_v:
                    cls.set_row_and_col_idxs(arr, row_slice, col_idxs, v[[i]])
                else:
                    cls.set_row_and_col_idxs(arr, row_slice, col_idxs, v)
        elif isinstance(col_idxs, np.ndarray) and col_idxs.ndim == 2:
            if not single_v:
                if isinstance(row_idxs, slice):
                    row_idxs = np.arange(arr.shape[0])[row_idxs]
                v = broadcast_array_to(v, (len(row_idxs), len(col_idxs)))
            for j in range(len(col_idxs)):
                col_slice = slice(col_idxs[j, 0], col_idxs[j, 1])
                if not single_v:
                    cls.set_row_and_col_idxs(arr, row_idxs, col_slice, v[:, [j]])
                else:
                    cls.set_row_and_col_idxs(arr, row_idxs, col_slice, v)
        else:
            if np.isscalar(row_idxs) or np.isscalar(col_idxs):
                arr[row_idxs, col_idxs] = v
            elif np.isscalar(v) and (isinstance(row_idxs, slice) or isinstance(col_idxs, slice)):
                arr[row_idxs, col_idxs] = v
            elif np.isscalar(v):
                arr[np.ix_(row_idxs, col_idxs)] = v
            else:
                if isinstance(row_idxs, slice):
                    row_idxs = np.arange(arr.shape[0])[row_idxs]
                if isinstance(col_idxs, slice):
                    col_idxs = np.arange(arr.shape[1])[col_idxs]
                v = broadcast_array_to(v, (len(row_idxs), len(col_idxs)))
                arr[np.ix_(row_idxs, col_idxs)] = v

    def get_set_meta(
        self,
        shape: tp.ShapeLike,
        index: tp.Optional[tp.Index] = None,
        columns: tp.Optional[tp.Index] = None,
        freq: tp.Optional[tp.FrequencyLike] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Kwargs:
        """Get meta of setting operations in `IdxSetter.idx_items`."""
        from vectorbtpro.base.reshaping import to_tuple_shape

        shape = to_tuple_shape(shape)
        rows_changed = False
        cols_changed = False
        set_funcs = []
        default = None

        for idxr, v in self.idx_items:
            if isinstance(idxr, str) and idxr == "_def":
                if default is None:
                    default = v
                continue
            row_idxs, col_idxs = get_idxs(
                idxr,
                index=index,
                columns=columns,
                freq=freq,
                template_context=template_context,
            )
            if isinstance(v, CustomTemplate):
                _template_context = merge_dicts(
                    dict(
                        idxr=idxr,
                        row_idxs=row_idxs,
                        col_idxs=col_idxs,
                    ),
                    template_context,
                )
                v = v.substitute(_template_context, eval_id="set")
            if not isinstance(v, np.ndarray):
                v = np.asarray(v)

            def _check_use_idxs(idxs):
                use_idxs = True
                if isinstance(idxs, slice):
                    if idxs.start is None and idxs.stop is None and idxs.step is None:
                        use_idxs = False
                if isinstance(idxs, np.ndarray):
                    if idxs.size == 0:
                        use_idxs = False
                return use_idxs

            use_row_idxs = _check_use_idxs(row_idxs)
            use_col_idxs = _check_use_idxs(col_idxs)

            if use_row_idxs and use_col_idxs:
                set_funcs.append(partial(self.set_row_and_col_idxs, row_idxs=row_idxs, col_idxs=col_idxs, v=v))
                rows_changed = True
                cols_changed = True
            elif use_col_idxs:
                set_funcs.append(partial(self.set_col_idxs, idxs=col_idxs, v=v))
                if checks.is_int(col_idxs):
                    if v.size > 1:
                        rows_changed = True
                else:
                    if v.ndim == 2:
                        if v.shape[0] > 1:
                            rows_changed = True
                cols_changed = True
            else:
                set_funcs.append(partial(self.set_row_idxs, idxs=row_idxs, v=v))
                if use_row_idxs:
                    rows_changed = True
                if len(shape) == 2:
                    if checks.is_int(row_idxs):
                        if v.size > 1:
                            cols_changed = True
                    else:
                        if v.ndim == 2:
                            if v.shape[1] > 1:
                                cols_changed = True
        return dict(
            default=default,
            set_funcs=set_funcs,
            rows_changed=rows_changed,
            cols_changed=cols_changed,
        )

    def set(self, arr: tp.Array, set_funcs: tp.Optional[tp.Sequence[tp.Callable]] = None, **kwargs) -> None:
        """Set values of a NumPy array based on `IdxSetter.get_set_meta`."""
        if set_funcs is None:
            set_meta = self.get_set_meta(arr.shape, **kwargs)
            set_funcs = set_meta["set_funcs"]
        for set_op in set_funcs:
            set_op(arr)

    def set_pd(self, pd_arr: tp.SeriesFrame, **kwargs) -> None:
        """Set values of a Pandas array based on `IdxSetter.get_set_meta`."""
        from vectorbtpro.base.indexes import get_index

        index = get_index(pd_arr, 0)
        columns = get_index(pd_arr, 1)
        freq = dt.infer_index_freq(index)
        self.set(pd_arr.values, index=index, columns=columns, freq=freq, **kwargs)

    def fill_and_set(
        self,
        shape: tp.ShapeLike,
        keep_flex: bool = False,
        fill_value: tp.Scalar = np.nan,
        **kwargs,
    ) -> tp.Array:
        """Fill a new array and set its values based on `IdxSetter.get_set_meta`.

        If `keep_flex` is True, will return the most memory-efficient array representation
        capable of flexible indexing.

        If `fill_value` is None, will search for the `_def` key in `IdxSetter.idx_items`.
        If there's none, will be set to NaN."""
        set_meta = self.get_set_meta(shape, **kwargs)
        if set_meta["default"] is not None:
            fill_value = set_meta["default"]
        if isinstance(fill_value, str):
            dtype = object
        else:
            dtype = None
        if keep_flex and not set_meta["cols_changed"] and not set_meta["rows_changed"]:
            arr = np.full((1,) if len(shape) == 1 else (1, 1), fill_value, dtype=dtype)
        elif keep_flex and not set_meta["cols_changed"]:
            arr = np.full(shape if len(shape) == 1 else (shape[0], 1), fill_value, dtype=dtype)
        elif keep_flex and not set_meta["rows_changed"]:
            arr = np.full((1, shape[1]), fill_value, dtype=dtype)
        else:
            arr = np.full(shape, fill_value, dtype=dtype)
        self.set(arr, set_funcs=set_meta["set_funcs"])
        return arr


class IdxSetterFactory:
    """Class for building index setters."""

    def get(self) -> tp.Union[IdxSetter, tp.Dict[tp.Label, IdxSetter]]:
        """Get an instance of `IdxSetter` or a dict of such instances - one per array name."""
        raise NotImplementedError


@define
class IdxDict(IdxSetterFactory, DefineMixin):
    """Class for building an index setter from a dict."""

    index_dct: dict = define.field()
    """Dict that contains indexer objects as keys and values to be set as values."""

    def get(self) -> tp.Union[IdxSetter, tp.Dict[tp.Label, IdxSetter]]:
        return IdxSetter(list(self.index_dct.items()))


@define
class IdxSeries(IdxSetterFactory, DefineMixin):
    """Class for building an index setter from a Series."""

    sr: tp.AnyArray1d = define.field()
    """Series or any array-like object to create the Series from."""

    split: bool = define.field(default=False)
    """Whether to split the setting operation.
        
    If False, will set all values using a single operation.
    Otherwise, will do one operation per element."""

    idx_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `idx` if the indexer isn't an instance of `Idxr`."""

    def get(self) -> tp.Union[IdxSetter, tp.Dict[tp.Label, IdxSetter]]:
        sr = self.sr
        split = self.split
        idx_kwargs = self.idx_kwargs

        if idx_kwargs is None:
            idx_kwargs = {}
        if not isinstance(sr, pd.Series):
            sr = pd.Series(sr)
        if split:
            idx_items = list(sr.items())
        else:
            idx_items = [(sr.index, sr.values)]
        new_idx_items = []
        for idxr, v in idx_items:
            if idxr is None:
                raise ValueError("Indexer cannot be None")
            if not isinstance(idxr, Idxr):
                idxr = idx(idxr, **idx_kwargs)
            new_idx_items.append((idxr, v))
        return IdxSetter(new_idx_items)


@define
class IdxFrame(IdxSetterFactory, DefineMixin):
    """Class for building an index setter from a DataFrame."""

    df: tp.AnyArray2d = define.field()
    """DataFrame or any array-like object to create the DataFrame from."""

    split: tp.Union[bool, str] = define.field(default=False)
    """Whether to split the setting operation.
    
    If False, will set all values using a single operation.
    Otherwise, the following options are supported:

    * 'columns': one operation per column
    * 'rows': one operation per row
    * True or 'elements': one operation per element"""

    rowidx_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `rowidx` if the indexer isn't an instance of `RowIdxr`."""

    colidx_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `colidx` if the indexer isn't an instance of `ColIdxr`."""

    def get(self) -> tp.Union[IdxSetter, tp.Dict[tp.Label, IdxSetter]]:
        df = self.df
        split = self.split
        rowidx_kwargs = self.rowidx_kwargs
        colidx_kwargs = self.colidx_kwargs

        if rowidx_kwargs is None:
            rowidx_kwargs = {}
        if colidx_kwargs is None:
            colidx_kwargs = {}
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df)
        if isinstance(split, bool):
            if split:
                split = "elements"
            else:
                split = None
        if split is not None:
            if split.lower() == "columns":
                idx_items = []
                for col, sr in df.items():
                    idx_items.append((sr.index, col, sr.values))
            elif split.lower() == "rows":
                idx_items = []
                for row, sr in df.iterrows():
                    idx_items.append((row, df.columns, sr.values))
            elif split.lower() == "elements":
                idx_items = []
                for col, sr in df.items():
                    for row, v in sr.items():
                        idx_items.append((row, col, v))
            else:
                raise ValueError(f"Invalid option split='{split}'")
        else:
            idx_items = [(df.index, df.columns, df.values)]
        new_idx_items = []
        for row_idxr, col_idxr, v in idx_items:
            if row_idxr is None:
                raise ValueError("Row indexer cannot be None")
            if col_idxr is None:
                raise ValueError("Column indexer cannot be None")
            if row_idxr is not None and not isinstance(row_idxr, RowIdxr):
                row_idxr = rowidx(row_idxr, **rowidx_kwargs)
            if col_idxr is not None and not isinstance(col_idxr, ColIdxr):
                col_idxr = colidx(col_idxr, **colidx_kwargs)
            new_idx_items.append((idx(row_idxr, col_idxr), v))
        return IdxSetter(new_idx_items)


@define
class IdxRecords(IdxSetterFactory, DefineMixin):
    """Class for building index setters from records - one per field."""

    records: tp.RecordsLike = define.field()
    """Series, DataFrame, or any sequence of mapping-like objects.
    
    If a Series or DataFrame and the index is not a default range, the index will become a row field.
    If a custom row field is provided, the index will be ignored."""

    row_field: tp.Union[None, bool, tp.Label] = define.field(default=None)
    """Row field.
    
    If None or True, will search for "row", "index", "open time", and "date" (case-insensitive).
    If `IdxRecords.records` is a Series or DataFrame, will also include the index name
    if the index is not a default range.
    
    If a record doesn't have a row field, all rows will be set.
    If there's no row and column field, the field value will become the default of the entire array."""

    col_field: tp.Union[None, bool, tp.Label] = define.field(default=None)
    """Column field.

    If None or True, will search for "col", "column", and "symbol" (case-insensitive).
    
    If a record doesn't have a column field, all columns will be set.
    If there's no row and column field, the field value will become the default of the entire array."""

    rowidx_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `rowidx` if the indexer isn't an instance of `RowIdxr`."""

    colidx_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `colidx` if the indexer isn't an instance of `ColIdxr`."""

    def get(self) -> tp.Union[IdxSetter, tp.Dict[tp.Label, IdxSetter]]:
        records = self.records
        row_field = self.row_field
        col_field = self.col_field
        rowidx_kwargs = self.rowidx_kwargs
        colidx_kwargs = self.colidx_kwargs

        if rowidx_kwargs is None:
            rowidx_kwargs = {}
        if colidx_kwargs is None:
            colidx_kwargs = {}
        default_index = False
        index_field = None
        if isinstance(records, pd.Series):
            records = records.to_frame()
        if isinstance(records, pd.DataFrame):
            records = records
            if checks.is_default_index(records.index):
                default_index = True
            records = records.reset_index(drop=default_index)
            if not default_index:
                index_field = records.columns[0]
            records = records.itertuples(index=False)

        def _resolve_field_meta(fields):
            _row_field = row_field
            _row_kind = None
            _col_field = col_field
            _col_kind = None
            row_fields = set()
            col_fields = set()
            for field in fields:
                if isinstance(field, str) and index_field is not None and field == index_field:
                    row_fields.add((field, None))
                if isinstance(field, str) and field.lower() in ("row", "index"):
                    row_fields.add((field, None))
                if isinstance(field, str) and field.lower() in ("open time", "date", "datetime"):
                    if (field, None) in row_fields:
                        row_fields.remove((field, None))
                    row_fields.add((field, "datetime"))
                if isinstance(field, str) and field.lower() in ("col", "column"):
                    col_fields.add((field, None))
                if isinstance(field, str) and field.lower() == "symbol":
                    if (field, None) in col_fields:
                        col_fields.remove((field, None))
                    col_fields.add((field, "labels"))
            if _row_field in (None, True):
                if len(row_fields) == 0:
                    if _row_field is True:
                        raise ValueError("Cannot find row field")
                    _row_field = None
                elif len(row_fields) == 1:
                    _row_field, _row_kind = row_fields.pop()
                else:
                    raise ValueError("Multiple row field candidates")
            elif _row_field is False:
                _row_field = None
            if _col_field in (None, True):
                if len(col_fields) == 0:
                    if _col_field is True:
                        raise ValueError("Cannot find column field")
                    _col_field = None
                elif len(col_fields) == 1:
                    _col_field, _col_kind = col_fields.pop()
                else:
                    raise ValueError("Multiple column field candidates")
            elif _col_field is False:
                _col_field = None
            field_meta = dict()
            field_meta["row_field"] = _row_field
            field_meta["row_kind"] = _row_kind
            field_meta["col_field"] = _col_field
            field_meta["col_kind"] = _col_kind
            return field_meta

        idx_items = dict()
        for r in records:
            r = to_field_mapping(r)
            field_meta = _resolve_field_meta(r.keys())
            if field_meta["row_field"] is None:
                row_idxr = None
            else:
                row_idxr = r.get(field_meta["row_field"], None)
            if row_idxr == "_def":
                row_idxr = None
            if row_idxr is not None and not isinstance(row_idxr, RowIdxr):
                _rowidx_kwargs = dict(rowidx_kwargs)
                if field_meta["row_kind"] is not None and "kind" not in _rowidx_kwargs:
                    _rowidx_kwargs["kind"] = field_meta["row_kind"]
                row_idxr = rowidx(row_idxr, **_rowidx_kwargs)
            if field_meta["col_field"] is None:
                col_idxr = None
            else:
                col_idxr = r.get(field_meta["col_field"], None)
            if col_idxr is not None and not isinstance(col_idxr, ColIdxr):
                _colidx_kwargs = dict(colidx_kwargs)
                if field_meta["col_kind"] is not None and "kind" not in _colidx_kwargs:
                    _colidx_kwargs["kind"] = field_meta["col_kind"]
                col_idxr = colidx(col_idxr, **_colidx_kwargs)
            if isinstance(col_idxr, str) and col_idxr == "_def":
                col_idxr = None
            item_produced = False
            for k, v in r.items():
                if index_field is not None and k == index_field:
                    continue
                if field_meta["row_field"] is not None and k == field_meta["row_field"]:
                    continue
                if field_meta["col_field"] is not None and k == field_meta["col_field"]:
                    continue
                if k not in idx_items:
                    idx_items[k] = []
                if row_idxr is None and col_idxr is None:
                    idx_items[k].append(("_def", v))
                else:
                    idx_items[k].append((idx(row_idxr, col_idxr), v))
                item_produced = True
            if not item_produced:
                raise ValueError(f"Record {r} has no fields to set")

        idx_setters = dict()
        for k, v in idx_items.items():
            idx_setters[k] = IdxSetter(v)
        return idx_setters


posidx = PosIdxr
"""Shortcut for `PosIdxr`."""

__pdoc__["posidx"] = False

maskidx = MaskIdxr
"""Shortcut for `MaskIdxr`."""

__pdoc__["maskidx"] = False

lbidx = LabelIdxr
"""Shortcut for `LabelIdxr`."""

__pdoc__["lbidx"] = False

dtidx = DatetimeIdxr
"""Shortcut for `DatetimeIdxr`."""

__pdoc__["dtidx"] = False

dtcidx = DTCIdxr
"""Shortcut for `DTCIdxr`."""

__pdoc__["dtcidx"] = False

pointidx = PointIdxr
"""Shortcut for `PointIdxr`."""

__pdoc__["pointidx"] = False

rangeidx = RangeIdxr
"""Shortcut for `RangeIdxr`."""

__pdoc__["rangeidx"] = False

autoidx = AutoIdxr
"""Shortcut for `AutoIdxr`."""

__pdoc__["autoidx"] = False

rowidx = RowIdxr
"""Shortcut for `RowIdxr`."""

__pdoc__["rowidx"] = False

colidx = ColIdxr
"""Shortcut for `ColIdxr`."""

__pdoc__["colidx"] = False

idx = Idxr
"""Shortcut for `Idxr`."""

__pdoc__["idx"] = False
