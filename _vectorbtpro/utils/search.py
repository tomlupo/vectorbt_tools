# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for searching."""

from copy import copy

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import set_dict_item


def find_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> dict:
    """Find matches in an object in a recursive manner.

    Traverses dicts, tuples, lists and (frozen-)sets. Does not look for matches in keys.

    If `excl_types` is not None, uses `vectorbtpro.utils.checks.is_instance_of` to check whether
    the object is one of the types that are blacklisted. If so, the object is simply returned.
    Same for `incl_types` for whitelisting, but it has a priority over `excl_types`.

    If `max_len` is not None, processes any object only if it's shorter than the specified length.

    If `max_depth` is not None, processes any object only up to a certain recursion level.
    Level of 0 means dicts and other iterables are not processed, only matches are expected.

    Returns a map of keys (multiple levels get represented by a tuple) to their respective values.

    For defaults, see `vectorbtpro._settings.search`."""
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if match_func(_key, obj, **kwargs):
        return {_key: obj}
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return {}
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                match_dct = {}
                for k, v in obj.items():
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
                    match_dct.update(
                        find_in_obj(
                            v,
                            match_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                return match_dct
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                match_dct = {}
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    match_dct.update(
                        find_in_obj(
                            o,
                            match_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                return match_dct
    return {}


def replace_in_obj(obj: tp.Any, match_dct: dict, _key: tp.Optional[tp.Hashable] = None) -> tp.Any:
    """Replace matches in an object in a recursive manner.

    See `find_in_obj` for `match_dct` (returned value)."""
    if len(match_dct) == 0:
        return obj
    if _key in match_dct:
        return match_dct[_key]
    match_dct = dict(match_dct)

    if isinstance(obj, dict):
        new_obj = {}
        for k in obj:
            if k in match_dct:
                new_obj[k] = match_dct.pop(k)
            else:
                new_match_dct = {}
                for k2 in list(match_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == k:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_match_dct[new_k2] = match_dct.pop(k2)
                if len(new_match_dct) == 0:
                    new_obj[k] = obj[k]
                else:
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
                    new_obj[k] = replace_in_obj(obj[k], new_match_dct, _key=new_key)
        return new_obj
    if isinstance(obj, (tuple, list, set, frozenset)):
        new_obj = []
        for i in range(len(obj)):
            if i in match_dct:
                new_obj.append(match_dct.pop(i))
            else:
                new_match_dct = {}
                for k2 in list(match_dct.keys()):
                    if isinstance(k2, tuple) and k2[0] == i:
                        new_k2 = k2[1:] if len(k2) > 2 else k2[1]
                        new_match_dct[new_k2] = match_dct.pop(k2)
                if len(new_match_dct) == 0:
                    new_obj.append(obj[i])
                else:
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    new_obj.append(replace_in_obj(obj[i], new_match_dct, _key=new_key))
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)
    return obj


def any_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> bool:
    """Return whether there is any match in an object in a recursive manner.

    See `find_in_obj` for arguments."""
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if match_func(_key, obj, **kwargs):
        return True
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return False
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                for k, v in obj.items():
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
                    if find_in_obj(
                        v,
                        match_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    ):
                        return True
        if isinstance(obj, (tuple, list, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    if find_in_obj(
                        o,
                        match_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    ):
                        return True
    return False


def find_and_replace_in_obj(
    obj: tp.Any,
    match_func: tp.Callable,
    replace_func: tp.Callable,
    excl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    incl_types: tp.Union[None, bool, tp.Sequence[type]] = None,
    max_len: tp.Optional[int] = None,
    max_depth: tp.Optional[int] = None,
    make_copy: bool = True,
    check_any_first: bool = True,
    _key: tp.Optional[tp.Hashable] = None,
    _depth: int = 0,
    **kwargs,
) -> tp.Any:
    """Find and replace matches in an object in a recursive manner.

    See `find_in_obj` for arguments.

    !!! note
        If the object is deep (such as a dict or a list), creates a copy of it if any match found inside,
        thus losing the reference to the original. Make sure to do a deep or hybrid copy of the object
        before proceeding for consistent behavior, or disable `make_copy` to override the original in place.
    """
    from vectorbtpro._settings import settings

    search_cfg = settings["search"]

    if excl_types is None:
        excl_types = search_cfg["excl_types"]
    if isinstance(excl_types, bool) and excl_types:
        raise ValueError("Argument excl_types cannot be True")
    if incl_types is None:
        incl_types = search_cfg["incl_types"]
    if isinstance(incl_types, bool) and not incl_types:
        raise ValueError("Argument incl_types cannot be False")
    if max_len is None:
        max_len = search_cfg["max_len"]
    if max_depth is None:
        max_depth = search_cfg["max_depth"]

    if check_any_first and not any_in_obj(
        obj,
        match_func,
        excl_types=excl_types,
        incl_types=incl_types,
        max_len=max_len,
        max_depth=max_depth,
        _key=_key,
        _depth=_depth,
        **kwargs,
    ):
        return obj

    if match_func(_key, obj, **kwargs):
        return replace_func(_key, obj, **kwargs)
    if max_depth is None or _depth < max_depth:
        if excl_types not in (None, False) and checks.is_instance_of(obj, excl_types):
            if incl_types is None or not (incl_types is True or checks.is_instance_of(obj, incl_types)):
                return obj
        if isinstance(obj, dict):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for k, v in obj.items():
                    new_key = k if _key is None else (*_key, k) if isinstance(_key, tuple) else (_key, k)
                    set_dict_item(
                        obj,
                        k,
                        find_and_replace_in_obj(
                            v,
                            match_func,
                            replace_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            make_copy=make_copy,
                            check_any_first=False,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        ),
                        force=True,
                    )
                return obj
        if isinstance(obj, list):
            if max_len is None or len(obj) <= max_len:
                if make_copy:
                    obj = copy(obj)
                for i in range(len(obj)):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    obj[i] = find_and_replace_in_obj(
                        obj[i],
                        match_func,
                        replace_func,
                        excl_types=excl_types,
                        incl_types=incl_types,
                        max_len=max_len,
                        max_depth=max_depth,
                        make_copy=make_copy,
                        check_any_first=False,
                        _key=new_key,
                        _depth=_depth + 1,
                        **kwargs,
                    )
                return obj
        if isinstance(obj, (tuple, set, frozenset)):
            if max_len is None or len(obj) <= max_len:
                result = []
                for i, o in enumerate(obj):
                    new_key = i if _key is None else (*_key, i) if isinstance(_key, tuple) else (_key, i)
                    result.append(
                        find_and_replace_in_obj(
                            o,
                            match_func,
                            replace_func,
                            excl_types=excl_types,
                            incl_types=incl_types,
                            max_len=max_len,
                            max_depth=max_depth,
                            make_copy=make_copy,
                            check_any_first=False,
                            _key=new_key,
                            _depth=_depth + 1,
                            **kwargs,
                        )
                    )
                if checks.is_namedtuple(obj):
                    return type(obj)(*result)
                return type(obj)(result)
    return obj
