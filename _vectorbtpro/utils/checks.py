# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for validation during runtime."""

import datetime
import traceback
import warnings
from collections.abc import Hashable, Mapping
from inspect import signature, getmro
from keyword import iskeyword

import attr
import numba
import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp

__all__ = [
    "is_numba_enabled",
    "is_deep_equal",
]


class Comparable:
    """Class representing an object that can be compared to another object."""

    def equals(self, other: tp.Any, *args, **kwargs) -> bool:
        """Check two objects for (deep) equality.

        Should accept the keyword arguments accepted by `is_deep_equal`."""
        raise NotImplementedError

    def __eq__(self, other: tp.Any) -> bool:
        return self.equals(other)


# ############# Checks ############# #


def is_bool(arg: tp.Any) -> bool:
    """Check whether the argument is a bool."""
    return isinstance(arg, (bool, np.bool_))


def is_int(arg: tp.Any) -> bool:
    """Check whether the argument is an integer (and not a timedelta, for example)."""
    return isinstance(arg, (int, np.integer)) and not isinstance(arg, np.timedelta64)


def is_float(arg: tp.Any) -> bool:
    """Check whether the argument is a float."""
    return isinstance(arg, (float, np.floating))


def is_number(arg: tp.Any) -> bool:
    """Check whether the argument is a number."""
    return is_int(arg) or is_float(arg)


def is_np_scalar(arg: tp.Any) -> bool:
    """Check whether the argument is a NumPy scalar."""
    return isinstance(arg, np.generic)


def is_td(arg: tp.Any) -> bool:
    """Check whether the argument is a timedelta object."""
    return isinstance(arg, (pd.Timedelta, datetime.timedelta, np.timedelta64))


def is_td_like(arg: tp.Any) -> bool:
    """Check whether the argument is a timedelta-like object."""
    return is_td(arg) or is_number(arg) or isinstance(arg, str)


def is_frequency(arg: tp.Any) -> bool:
    """Check whether the argument is a frequency object."""
    return is_td(arg) or isinstance(arg, pd.DateOffset)


def is_frequency_like(arg: tp.Any) -> bool:
    """Check whether the argument is a frequency-like object."""
    return is_frequency(arg) or is_number(arg) or isinstance(arg, str)


def is_dt(arg: tp.Any) -> bool:
    """Check whether the argument is a datetime object."""
    return isinstance(arg, (pd.Timestamp, datetime.datetime, np.datetime64))


def is_dt_like(arg: tp.Any) -> bool:
    """Check whether the argument is a datetime-like object."""
    return is_dt(arg) or is_number(arg) or isinstance(arg, str)


def is_time(arg: tp.Any) -> bool:
    """Check whether the argument is a time object."""
    return isinstance(arg, datetime.time)


def is_time_like(arg: tp.Any) -> bool:
    """Check whether the argument is a time-like object."""
    return is_time(arg) or isinstance(arg, str)


def is_np_array(arg: tp.Any) -> bool:
    """Check whether the argument is a NumPy array."""
    return isinstance(arg, np.ndarray)


def is_record_array(arg: tp.Any) -> bool:
    """Check whether the argument is a structured NumPy array."""
    return is_np_array(arg) and arg.dtype.fields is not None


def is_series(arg: tp.Any) -> bool:
    """Check whether the argument is `pd.Series`."""
    return isinstance(arg, pd.Series)


def is_index(arg: tp.Any) -> bool:
    """Check whether the argument is `pd.Index`."""
    return isinstance(arg, pd.Index)


def is_multi_index(arg: tp.Any) -> bool:
    """Check whether the argument is `pd.MultiIndex`."""
    return isinstance(arg, pd.MultiIndex)


def is_frame(arg: tp.Any) -> bool:
    """Check whether the argument is `pd.DataFrame`."""
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg: tp.Any) -> bool:
    """Check whether the argument is `pd.Series`, `pd.Index`, or `pd.DataFrame`."""
    return is_series(arg) or is_index(arg) or is_frame(arg)


def is_any_array(arg: tp.Any) -> bool:
    """Check whether the argument is a NumPy array or a Pandas object."""
    return is_pandas(arg) or isinstance(arg, np.ndarray)


def _to_any_array(arg: tp.ArrayLike) -> tp.AnyArray:
    """Convert any array-like object to an array.

    Pandas objects are kept as-is."""
    if is_any_array(arg):
        return arg
    return np.asarray(arg)


def is_sequence(arg: tp.Any) -> bool:
    """Check whether the argument is a sequence."""
    try:
        len(arg)
        arg[0:0]
        return True
    except (TypeError, KeyError):
        return False


def is_complex_sequence(arg: tp.Any) -> bool:
    """Check whether the argument is a sequence but not a string or bytes object."""
    if isinstance(arg, (str, bytes, bytearray)):
        return False
    return is_sequence(arg)


def is_iterable(arg: tp.Any) -> bool:
    """Check whether the argument is iterable."""
    try:
        _ = iter(arg)
        return True
    except TypeError:
        return False


def is_complex_iterable(arg: tp.Any) -> bool:
    """Check whether the argument is iterable but not a string or bytes object."""
    if isinstance(arg, (str, bytes, bytearray)):
        return False
    return is_iterable(arg)


def is_numba_enabled() -> bool:
    """Check whether Numba is enabled globally."""
    return numba.config.DISABLE_JIT != 1


def is_numba_func(arg: tp.Any) -> bool:
    """Check whether the argument is a Numba-compiled function."""
    from vectorbtpro._settings import settings

    numba_cfg = settings["numba"]

    if not numba_cfg["check_func_type"]:
        return True
    if not is_numba_enabled():
        if numba_cfg["check_func_suffix"]:
            if arg.__name__.endswith("_nb"):
                return True
            return False
        return False
    return isinstance(arg, CPUDispatcher)


def is_hashable(arg: tp.Any) -> bool:
    """Check whether the argument can be hashed."""
    if not isinstance(arg, Hashable):
        return False
    # Having __hash__() method does not mean that it's hashable
    try:
        hash(arg)
    except TypeError:
        return False
    return True


def is_index_equal(arg1: tp.Any, arg2: tp.Any, check_names: bool = True) -> bool:
    """Check whether indexes are equal.

    If `check_names` is True, checks whether names are equal on top of `pd.Index.equals`."""
    if not check_names:
        return pd.Index.equals(arg1, arg2)
    if isinstance(arg1, pd.MultiIndex) and isinstance(arg2, pd.MultiIndex):
        if arg1.names != arg2.names:
            return False
    elif isinstance(arg1, pd.MultiIndex) or isinstance(arg2, pd.MultiIndex):
        return False
    else:
        if arg1.name != arg2.name:
            return False
    return pd.Index.equals(arg1, arg2)


def is_default_index(arg: tp.Any, check_names: bool = True) -> bool:
    """Check whether index is a basic range."""
    return is_index_equal(arg, pd.RangeIndex(start=0, stop=len(arg), step=1), check_names=check_names)


def is_namedtuple(arg: tp.Any) -> bool:
    """Check whether object is an instance of namedtuple."""
    if not isinstance(arg, type):
        arg = type(arg)
    bases = arg.__bases__
    if len(bases) != 1 or bases[0] != tuple:
        return False
    fields = getattr(arg, "_fields", None)
    if not isinstance(fields, tuple):
        return False
    return all(type(field) == str for field in fields)


def is_record(arg: tp.Any) -> bool:
    """Check whether object is a NumPy record."""
    return isinstance(arg, (np.void, np.record)) and hasattr(arg.dtype, "names") and arg.dtype.names is not None


def func_accepts_arg(func: tp.Callable, arg_name: str, arg_kind: tp.Optional[tp.MaybeTuple[int]] = None) -> bool:
    """Check whether `func` accepts a positional or keyword argument with name `arg_name`."""
    sig = signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    if arg_kind is None:
        if arg_name.startswith("**"):
            return arg_name[2:] in [p.name for p in sig.parameters.values() if p.kind == p.VAR_KEYWORD]
        if arg_name.startswith("*"):
            return arg_name[1:] in [p.name for p in sig.parameters.values() if p.kind == p.VAR_POSITIONAL]
        return arg_name in [
            p.name for p in sig.parameters.values() if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
        ]
    return arg_name in [p.name for p in sig.parameters.values() if p.kind in arg_kind]


def is_equal(
    arg1: tp.Any,
    arg2: tp.Any,
    equality_func: tp.Callable[[tp.Any, tp.Any], bool] = lambda x, y: x == y,
) -> bool:
    """Check whether two objects are equal."""
    try:
        return equality_func(arg1, arg2)
    except:
        pass
    return False


def is_deep_equal(
    arg1: tp.Any,
    arg2: tp.Any,
    check_exact: bool = False,
    debug: bool = False,
    _key: tp.Optional[str] = None,
    only_types: bool = False,
    **kwargs,
) -> bool:
    """Check whether two objects are equal (deep check)."""

    def _select_kwargs(_method, _kwargs):
        __kwargs = dict()
        if len(kwargs) > 0:
            for k, v in _kwargs.items():
                if func_accepts_arg(_method, k):
                    __kwargs[k] = v
        return __kwargs

    def _check_array(assert_method):
        __kwargs = _select_kwargs(assert_method, kwargs)
        if arg1.dtype != arg2.dtype:
            raise AssertionError(f"Dtypes {arg1.dtype} and {arg2.dtype} do not match")
        if arg1.dtype.fields is not None:
            for field in arg1.dtype.names:
                try:
                    assert_method(arg1[field], arg2[field], **__kwargs)
                except Exception as e:
                    raise AssertionError(f"Dtype field '{field}'") from e
        else:
            assert_method(arg1, arg2, **__kwargs)

    try:
        if only_types:
            if type(arg1) != type(arg2):
                raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")
            return True
        if id(arg1) == id(arg2):
            return True
        if isinstance(arg1, Comparable):
            return arg1.equals(arg2, check_exact=check_exact, debug=debug, _key=_key, **kwargs)
        if type(arg1) != type(arg2):
            raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")
        if attr.has(type(arg1)):
            return is_deep_equal(
                attr.asdict(arg1),
                attr.asdict(arg2),
                check_exact=check_exact,
                debug=debug,
                _key=_key,
                **kwargs,
            )
        if isinstance(arg1, pd.Series):
            _kwargs = _select_kwargs(pd.testing.assert_series_equal, kwargs)
            pd.testing.assert_series_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.DataFrame):
            _kwargs = _select_kwargs(pd.testing.assert_frame_equal, kwargs)
            pd.testing.assert_frame_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, pd.Index):
            _kwargs = _select_kwargs(pd.testing.assert_index_equal, kwargs)
            pd.testing.assert_index_equal(arg1, arg2, check_exact=check_exact, **_kwargs)
        elif isinstance(arg1, np.ndarray):
            try:
                _check_array(np.testing.assert_array_equal)
            except Exception as e:
                if check_exact:
                    raise e
                _check_array(np.testing.assert_allclose)
        elif isinstance(arg1, (tuple, list)):
            for i in range(len(arg1)):
                if not is_deep_equal(
                    arg1[i],
                    arg2[i],
                    check_exact=check_exact,
                    debug=debug,
                    _key=f"[{i}]" if _key is None else _key + f"[{i}]",
                    **kwargs,
                ):
                    return False
        elif isinstance(arg1, dict):
            for k in arg1.keys():
                if not is_deep_equal(
                    arg1[k],
                    arg2[k],
                    check_exact=check_exact,
                    debug=debug,
                    _key=f"['{k}']" if _key is None else _key + f"['{k}']",
                    **kwargs,
                ):
                    return False
        else:
            try:
                if arg1 == arg2:
                    return True
            except:
                pass
            try:
                import dill

                _kwargs = _select_kwargs(dill.dumps, kwargs)
                if dill.dumps(arg1, **_kwargs) == dill.dumps(arg2, **_kwargs):
                    return True
            except:
                pass
            if debug:
                warnings.warn(
                    f"\n############### {_key} ###############\nObjects do not match",
                    stacklevel=2,
                )
            return False
    except Exception as e:
        if debug:
            if _key is None:
                warnings.warn(traceback.format_exc(), stacklevel=2)
            else:
                warnings.warn(
                    f"\n############### {_key} ###############\n" + traceback.format_exc(),
                    stacklevel=2,
                )
        return False
    return True


def is_class(arg: type, types: tp.TypeLike) -> bool:
    """Check whether the argument is `types`.

    `types` can be one or multiple types, strings, or patterns of type `vectorbtpro.utils.parsing.Regex`."""
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, str):
        return str(arg) == types or arg.__name__ == types
    if isinstance(types, Regex):
        return types.matches(str(arg)) or types.matches(arg.__name__)
    if isinstance(types, tuple):
        for t in types:
            if is_class(arg, t):
                return True
        return False
    return arg is types


def is_subclass_of(arg: tp.Any, types: tp.TypeLike) -> bool:
    """Check whether the argument is a subclass of `types`.

    `types` can be one or multiple types, strings, or patterns of type `vectorbtpro.utils.parsing.Regex`."""
    from vectorbtpro.utils.parsing import Regex

    if isinstance(types, type):
        return issubclass(arg, types)
    if isinstance(types, str):
        for base_t in getmro(arg):
            if str(base_t) == types or base_t.__name__ == types:
                return True
    if isinstance(types, Regex):
        for base_t in getmro(arg):
            if types.matches(str(base_t)) or types.matches(base_t.__name__):
                return True
    if isinstance(types, tuple):
        for t in types:
            if is_subclass_of(arg, t):
                return True
    return False


def is_instance_of(arg: tp.Any, types: tp.TypeLike) -> bool:
    """Check whether the argument is an instance of `types`.

    `types` can be one or multiple types or strings."""
    return is_subclass_of(type(arg), types)


def is_mapping(arg: tp.Any) -> bool:
    """Check whether the arguments is a mapping."""
    return isinstance(arg, Mapping)


def is_mapping_like(arg: tp.Any) -> bool:
    """Check whether the arguments is a mapping-like object."""
    return is_mapping(arg) or is_series(arg) or is_index(arg) or is_namedtuple(arg)


def is_valid_variable_name(arg: str) -> bool:
    """Check whether the argument is a valid variable name."""
    return arg.isidentifier() and not iskeyword(arg)


def is_notebook() -> bool:
    """Check whether the code runs in a notebook.

    Credit: https://stackoverflow.com/a/39662359"""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True
        elif shell == "TerminalInteractiveShell":
            return False
        else:
            return False
    except NameError:
        return False


# ############# Asserts ############# #


def safe_assert(arg: bool, msg: tp.Optional[str] = None) -> None:
    if not arg:
        raise AssertionError(msg)


def assert_in(arg1: tp.Any, arg2: tp.Sequence, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the first argument is not in the second argument."""
    if arg_name is None:
        x = ""
    else:
        x = f"for '{arg_name}'"
    if arg1 not in arg2:
        raise AssertionError(f"{arg1} not found in {arg2}{x}")


def assert_numba_func(func: tp.Callable) -> None:
    """Raise exception if `func` is not Numba-compiled."""
    if not is_numba_func(func):
        raise AssertionError(f"Function {func} must be Numba compiled")


def assert_not_none(arg: tp.Any, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is None."""
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if arg is None:
        raise AssertionError(f"{x} cannot be None")


def assert_instance_of(arg: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is none of types `types`."""
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_instance_of(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} must be of one of types {types}, not {type(arg)}")
        else:
            raise AssertionError(f"{x} must be of type {types}, not {type(arg)}")


def assert_not_instance_of(arg: tp.Any, types: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is one of types `types`."""
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_instance_of(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"{x} cannot be of one of types {types}")
        else:
            raise AssertionError(f"{x} cannot be of type {types}")


def assert_subclass_of(arg: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is not a subclass of classes `classes`."""
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if not is_subclass_of(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} must be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} must be a subclass of type {classes}")


def assert_not_subclass_of(arg: tp.Type, classes: tp.TypeLike, arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is a subclass of classes `classes`."""
    if arg_name is None:
        x = "Argument"
    else:
        x = f"Argument '{arg_name}'"
    if is_subclass_of(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"{x} cannot be a subclass of one of types {classes}")
        else:
            raise AssertionError(f"{x} cannot be a subclass of type {classes}")


def assert_type_equal(arg1: tp.Any, arg2: tp.Any) -> None:
    """Raise exception if the first argument and the second argument have different types."""
    if type(arg1) != type(arg2):
        raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")


def assert_dtype(arg: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is not of data type `dtype`."""
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    arg = _to_any_array(arg)
    if isinstance(dtype, tuple):
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not any([col_dtype == _dtype for _dtype in dtype]):
                    raise AssertionError(f"{x} (column {i}) must be one of {dtype}, not {col_dtype}")
        else:
            if not any([arg.dtype == _dtype for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {arg.dtype}")
    else:
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if col_dtype != dtype:
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if arg.dtype != dtype:
                raise AssertionError(f"{x} must be {dtype}, not {arg.dtype}")


def assert_subdtype(arg: tp.ArrayLike, dtype: tp.MaybeTuple[tp.DTypeLike], arg_name: tp.Optional[str] = None) -> None:
    """Raise exception if the argument is not a sub data type of `dtype`."""
    if arg_name is None:
        x = "Data type"
    else:
        x = f"Data type of '{arg_name}'"
    arg = _to_any_array(arg)
    if isinstance(dtype, tuple):
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not any([np.issubdtype(col_dtype, _dtype) for _dtype in dtype]):
                    raise AssertionError(f"{x} (column {i}) must be one of {dtype}, not {col_dtype}")
        else:
            if not any([np.issubdtype(arg.dtype, _dtype) for _dtype in dtype]):
                raise AssertionError(f"{x} must be one of {dtype}, not {arg.dtype}")
    else:
        if isinstance(arg, pd.DataFrame):
            for i, col_dtype in enumerate(arg.dtypes):
                if not np.issubdtype(col_dtype, dtype):
                    raise AssertionError(f"{x} (column {i}) must be {dtype}, not {col_dtype}")
        else:
            if not np.issubdtype(arg.dtype, dtype):
                raise AssertionError(f"{x} must be {dtype}, not {arg.dtype}")


def assert_dtype_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """Raise exception if the first argument and the second argument have different data types."""
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if isinstance(arg1, pd.DataFrame):
        dtypes1 = arg1.dtypes.to_numpy()
    else:
        dtypes1 = np.array([arg1.dtype])
    if isinstance(arg2, pd.DataFrame):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.array([arg2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if np.all(np.unique(dtypes1) == np.unique(dtypes2)):
            return
    raise AssertionError(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(arg: tp.ArrayLike, ndims: tp.MaybeTuple[int]) -> None:
    """Raise exception if the argument has a different number of dimensions than `ndims`."""
    arg = _to_any_array(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise AssertionError(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise AssertionError(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def assert_len_equal(arg1: tp.Sized, arg2: tp.Sized) -> None:
    """Raise exception if the first argument and the second argument have different length.

    Does not transform arguments to NumPy arrays."""
    if len(arg1) != len(arg2):
        raise AssertionError(f"Lengths of {arg1} and {arg2} do not match")


def assert_shape_equal(
    arg1: tp.ArrayLike,
    arg2: tp.ArrayLike,
    axis: tp.Optional[tp.Union[int, tp.Tuple[int, int]]] = None,
) -> None:
    """Raise exception if the first argument and the second argument have different shapes along `axis`."""
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    if axis is None:
        if arg1.shape != arg2.shape:
            raise AssertionError(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if axis[0] >= arg1.ndim and axis[1] >= arg2.ndim:
                return
            if arg1.shape[axis[0]] != arg2.shape[axis[1]]:
                raise AssertionError(f"Axis {axis[0]} of {arg1.shape} and axis {axis[1]} of {arg2.shape} do not match")
        else:
            if axis >= arg1.ndim and axis >= arg2.ndim:
                return
            if arg1.shape[axis] != arg2.shape[axis]:
                raise AssertionError(f"Axis {axis} of {arg1.shape} and {arg2.shape} do not match")


def assert_index_equal(arg1: tp.Index, arg2: tp.Index, check_names: bool = True) -> None:
    """Raise exception if the first argument and the second argument have different index."""
    if not is_index_equal(arg1, arg2, check_names=check_names):
        raise AssertionError(f"Indexes {arg1} and {arg2} do not match")


def assert_columns_equal(arg1: tp.Index, arg2: tp.Index, check_names: bool = True) -> None:
    """Raise exception if the first argument and the second argument have different columns."""
    if not is_index_equal(arg1, arg2, check_names=check_names):
        raise AssertionError(f"Columns {arg1} and {arg2} do not match")


def assert_meta_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike, axis: tp.Optional[int] = None) -> None:
    """Raise exception if the first argument and the second argument have different metadata."""
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_type_equal(arg1, arg2)
    if axis is not None:
        assert_shape_equal(arg1, arg2, axis=axis)
    else:
        assert_shape_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        if axis is None or axis == 0:
            assert_index_equal(arg1.index, arg2.index)
        if axis is None or axis == 1:
            if is_series(arg1) and is_series(arg2):
                assert_columns_equal(pd.Index([arg1.name]), pd.Index([arg2.name]))
            else:
                assert_columns_equal(arg1.columns, arg2.columns)


def assert_array_equal(arg1: tp.ArrayLike, arg2: tp.ArrayLike) -> None:
    """Raise exception if the first argument and the second argument have different metadata or values."""
    arg1 = _to_any_array(arg1)
    arg2 = _to_any_array(arg2)
    assert_meta_equal(arg1, arg2)
    if is_pandas(arg1) and is_pandas(arg2):
        if arg1.equals(arg2):
            return
    elif not is_pandas(arg1) and not is_pandas(arg2):
        if np.array_equal(arg1, arg2):
            return
    raise AssertionError(f"Arrays {arg1} and {arg2} do not match")


def assert_level_not_exists(arg: tp.Index, level_name: str) -> None:
    """Raise exception if index the argument has level `level_name`."""
    if isinstance(arg, pd.MultiIndex):
        names = arg.names
    else:
        names = [arg.name]
    if level_name in names:
        raise AssertionError(f"Level {level_name} already exists in {names}")


def assert_equal(arg1: tp.Any, arg2: tp.Any, deep: bool = False) -> None:
    """Raise exception if the first argument and the second argument are different."""
    if deep:
        if not is_deep_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match (deep check)")
    else:
        if not is_equal(arg1, arg2):
            raise AssertionError(f"{arg1} and {arg2} do not match")


def assert_dict_valid(arg: tp.DictLike, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """Raise exception if dict the argument has keys that are not in `lvl_keys`.

    `lvl_keys` must be a list of lists, each corresponding to a level in the dict."""
    if arg is None:
        arg = {}
    if len(lvl_keys) == 0:
        return
    if isinstance(lvl_keys[0], str):
        lvl_keys = [lvl_keys]
    set1 = set(arg.keys())
    set2 = set(lvl_keys[0])
    if not set1.issubset(set2):
        raise AssertionError(f"Invalid keys {list(set1.difference(set2))}, possible keys are {list(set2)}")
    for k, v in arg.items():
        if isinstance(v, dict):
            assert_dict_valid(v, lvl_keys[1:])


def assert_dict_sequence_valid(arg: tp.DictLikeSequence, lvl_keys: tp.Sequence[tp.MaybeSequence[str]]) -> None:
    """Raise exception if a dict or any dict in a sequence of dicts has keys that are not in `lvl_keys`."""
    if arg is None:
        arg = {}
    if isinstance(arg, dict):
        assert_dict_valid(arg, lvl_keys)
    else:
        for _arg in arg:
            assert_dict_valid(_arg, lvl_keys)


def assert_sequence(arg: tp.Any) -> None:
    """Raise exception if the argument is not a sequence."""
    if not is_sequence(arg):
        raise ValueError(f"{arg} must be a sequence")


def assert_iterable(arg: tp.Any) -> None:
    """Raise exception if the argument is not an iterable."""
    if not is_iterable(arg):
        raise ValueError(f"{arg} must be an iterable")
