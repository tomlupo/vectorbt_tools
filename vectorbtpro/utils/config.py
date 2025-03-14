# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for configuration."""

import inspect
import warnings
from copy import copy, deepcopy
from pathlib import Path

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.caching import Cacheable
from vectorbtpro.utils.chaining import Chainable
from vectorbtpro.utils.checks import Comparable, is_deep_equal, assert_in, assert_instance_of
from vectorbtpro.utils.decorators import class_or_instancemethod
from vectorbtpro.utils.formatting import Prettified, prettify_dict, prettify_inited
from vectorbtpro.utils.pickling import RecState, Pickleable, pdict

__all__ = [
    "hdict",
    "atomic_dict",
    "unsetkey",
    "merge_dicts",
    "child_dict",
    "Config",
    "FrozenConfig",
    "ReadonlyConfig",
    "HybridConfig",
    "Configured",
    "AtomicConfig",
]


class hdict(dict):
    """Hashable dict."""

    def __hash__(self):
        return hash(frozenset(self.items()))


def resolve_dict(dct: tp.DictLikeSequence, i: tp.Optional[int] = None) -> dict:
    """Select keyword arguments."""
    if dct is None:
        dct = {}
    if isinstance(dct, dict):
        return dict(dct)
    if i is not None:
        _dct = dct[i]
        if _dct is None:
            _dct = {}
        return dict(_dct)
    raise ValueError("Cannot resolve dict")


class atomic_dict(pdict):
    """Dict that behaves like a single value when merging."""

    pass


InConfigLikeT = tp.Union[None, dict, "ConfigT"]
OutConfigLikeT = tp.Union[dict, "ConfigT"]


def convert_to_dict(dct: InConfigLikeT, nested: bool = True) -> dict:
    """Convert any config to `dict`.

    Set `nested` to True to convert all child dicts in recursive manner.

    If a config is an instance of `AtomicConfig`, will convert it to `atomic_dict`."""
    if dct is None:
        dct = {}
    if isinstance(dct, Config):
        if isinstance(dct, AtomicConfig):
            dct = atomic_dict(dct)
        else:
            dct = dict(dct)
    else:
        dct = type(dct)(dct)
    if not nested:
        return dct
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = convert_to_dict(v, nested=nested)
        else:
            dct[k] = v
    return dct


def resolve_pathlike_key(k: tp.PathLikeKey) -> tp.Tuple[tp.Hashable, ...]:
    """Convert a path-like key into a tuple."""
    if isinstance(k, Path):
        k = k.parts
    if isinstance(k, str) and "." in k:
        k = tuple(k.split("."))
    if isinstance(k, tuple):
        return k
    return (k,)


def combine_pathlike_keys(k1: tp.PathLikeKey, k2: tp.PathLikeKey) -> tp.PathLikeKey:
    """Combine two path-like keys."""
    if isinstance(k1, Path) and isinstance(k2, Path):
        return k1 / k2
    if isinstance(k1, str) and isinstance(k2, str):
        return k1 + "." + k2
    k1 = resolve_pathlike_key(k1)
    k2 = resolve_pathlike_key(k2)
    return k1 + k2


def get_dict_item(dct: dict, k: tp.PathLikeKey, populate: bool = False) -> tp.Any:
    """Get dict item under the key `k`.

    The key can be nested using the dot notation, `pathlib.Path`, or a tuple, and must be hashable."""
    if k in dct:
        return dct[k]
    k = resolve_pathlike_key(k)
    if len(k) == 1:
        k = k[0]
    else:
        return get_dict_item(get_dict_item(dct, k[0], populate=populate), k[1:], populate=populate)
    if k not in dct and populate:
        dct[k] = dict()
    return dct[k]


def set_dict_item(dct: dict, k: tp.Any, v: tp.Any, force: bool = False) -> None:
    """Set dict item.

    If the dict is of the type `Config`, also passes `force` keyword to override blocking flags."""
    if isinstance(dct, Config):
        dct.__setitem__(k, v, force=force)
    else:
        dct[k] = v


def del_dict_item(dct: dict, k: tp.Any, force: bool = False) -> None:
    """Delete dict item.

    If the dict is of the type `Config`, also passes `force` keyword to override blocking flags."""
    if isinstance(dct, Config):
        dct.__delitem__(k, force=force)
    else:
        del dct[k]


def copy_dict(dct: InConfigLikeT, copy_mode: str = "shallow", nested: bool = True) -> OutConfigLikeT:
    """Copy dict based on a copy mode.

    The following modes are supported:

    * 'none': Does not copy
    * 'shallow': Copies keys only
    * 'hybrid': Copies keys and values using `copy.copy`
    * 'deep': Copies the whole thing using `copy.deepcopy`

    Set `nested` to True to copy all child dicts in recursive manner."""
    if dct is None:
        return {}
    copy_mode = copy_mode.lower()
    if copy_mode not in {"none", "shallow", "hybrid", "deep"}:
        raise ValueError(f"Copy mode '{copy_mode}' is not supported")

    if copy_mode == "none":
        return dct
    if copy_mode == "deep":
        return deepcopy(dct)
    if isinstance(dct, Config):
        return dct.copy(copy_mode=copy_mode, nested=nested)
    dct_copy = copy(dct)  # copy structure using shallow copy
    for k, v in dct_copy.items():
        if nested and isinstance(v, dict):
            _v = copy_dict(v, copy_mode=copy_mode, nested=nested)
        else:
            if copy_mode == "hybrid":
                _v = copy(v)  # copy values using shallow copy
            else:
                _v = v
        set_dict_item(dct_copy, k, _v, force=True)
    return dct_copy


def update_dict(
    x: InConfigLikeT,
    y: InConfigLikeT,
    nested: bool = True,
    force: bool = False,
    same_keys: bool = False,
) -> None:
    """Update dict with keys and values from other dict.

    Set `nested` to True to update all child dicts in recursive manner.

    For `force`, see `set_dict_item`.

    If you want to treat any dict as a single value, wrap it with `atomic_dict`.

    If `nested` is True, a value in `x` is an instance of `Configured`, and the corresponding
    value in `y` is a dictionary, calls `Configured.replace`.

    !!! note
        If the child dict is not atomic, it will copy only its values, not its meta."""
    if x is None:
        return
    if y is None:
        return
    assert_instance_of(x, dict)
    assert_instance_of(y, dict)

    for k, v in y.items():
        if (
            nested
            and k in x
            and isinstance(x[k], (dict, Configured))
            and isinstance(v, dict)
            and not isinstance(v, atomic_dict)
        ):
            if isinstance(x[k], Configured):
                set_dict_item(x, k, x[k].replace(**v), force=force)
            else:
                update_dict(x[k], v, force=force)
        else:
            if same_keys and k not in x:
                continue
            set_dict_item(x, k, v, force=force)


class _unsetkey:
    pass


unsetkey = _unsetkey()
"""When passed as a value, the corresponding key will be unset.

It can still be overridden by another dict."""


def unset_keys(
    dct: InConfigLikeT,
    nested: bool = True,
    force: bool = False,
) -> None:
    """Unset the keys that have the value `unsetkey`."""
    if dct is None:
        return
    assert_instance_of(dct, dict)

    for k, v in list(dct.items()):
        if isinstance(v, _unsetkey):
            del_dict_item(dct, k, force=force)
        elif nested and isinstance(v, dict) and not isinstance(v, atomic_dict):
            unset_keys(v, nested=nested, force=force)


def merge_dicts(
    *dicts: InConfigLikeT,
    to_dict: bool = True,
    copy_mode: str = "shallow",
    nested: tp.Optional[bool] = None,
    same_keys: bool = False,
) -> OutConfigLikeT:
    """Merge dicts.

    Args:
        *dicts (dict): Dicts.
        to_dict (bool): Whether to call `convert_to_dict` on each dict prior to copying.
        copy_mode (str): Mode for `copy_dict` to copy each dict prior to merging.
        nested (bool): Whether to merge all child dicts in recursive manner.

            If None, checks whether any dict is nested.
        same_keys (bool): Whether to merge on the overlapping keys only."""
    if len(dicts) == 1:
        dicts = (None, dicts[0])

    # Shortcut when both dicts are None
    if dicts[0] is None and dicts[1] is None:
        if len(dicts) > 2:
            return merge_dicts(
                None,
                *dicts[2:],
                to_dict=to_dict,
                copy_mode=copy_mode,
                nested=nested,
                same_keys=same_keys,
            )
        return {}

    # Check whether any dict is nested
    if nested is None:
        for dct in dicts:
            if dct is not None:
                for v in dct.values():
                    if isinstance(v, dict) and not isinstance(v, atomic_dict):
                        nested = True
                        break
            if nested:
                break

    # Convert dict-like objects to regular dicts
    if to_dict:
        # Shortcut when all dicts are already regular
        if not nested and not same_keys and copy_mode in {"none", "shallow"}:
            out = {}
            for dct in dicts:
                if dct is not None:
                    out.update(dct)
            for k, v in list(out.items()):
                if isinstance(v, _unsetkey):
                    del out[k]
            return out
        dicts = tuple([convert_to_dict(dct, nested=True) for dct in dicts])

    # Copy all dicts
    if not to_dict or copy_mode not in {"none", "shallow"}:
        # to_dict already does a shallow copy
        dicts = tuple([copy_dict(dct, copy_mode=copy_mode, nested=nested) for dct in dicts])

    # Merge both dicts
    x, y = dicts[0], dicts[1]
    should_update = True
    if type(x) is dict and type(y) is dict and len(x) == 0:
        x = y
        should_update = False
    if isinstance(x, atomic_dict) or isinstance(y, atomic_dict):
        x = y
        should_update = False
    if should_update:
        update_dict(x, y, nested=nested, force=True, same_keys=same_keys)

    # Unset keys
    unset_keys(x, nested=nested, force=True)

    # Merge resulting dict with remaining dicts
    if len(dicts) > 2:
        return merge_dicts(
            x,
            *dicts[2:],
            to_dict=False,  # executed only once
            copy_mode="none",  # executed only once
            nested=nested,
            same_keys=same_keys,
        )
    return x


class child_dict(pdict):
    """Subclass of `dict` acting as a child dict."""

    pass


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(pdict):
    """Extends pickleable dict with config features such as nested updates, freezing, and resetting.

    Args:
        *args: Arguments to construct the dict from.
        options_ (dict): Config options (see below).
        **kwargs: Keyword arguments to construct the dict from.

    Options can have the following keys:

    Attributes:
        copy_kwargs (dict): Keyword arguments passed to `copy_dict` for copying main dict and `reset_dct`.

            Copy mode defaults to 'none'.
        reset_dct (dict): Dict to fall back to in case of resetting.

            Defaults to None. If None, copies main dict using `reset_dct_copy_kwargs`.

            !!! note
                Defaults to main dict in case it's None and `readonly` is True.
        reset_dct_copy_kwargs (dict): Keyword arguments that override `copy_kwargs` for `reset_dct`.

            Copy mode defaults to 'none' if `readonly` is True, else to 'hybrid'.
        pickle_reset_dct (bool): Whether to pickle `reset_dct`.
        frozen_keys (bool): Whether to deny updates to the keys of the config.

            Defaults to False.
        readonly (bool): Whether to deny updates to the keys and values of the config.

            Defaults to False.
        nested (bool): Whether to do operations recursively on each child dict.

            Such operations include copy, update, and merge.
            Disable to treat each child dict as a single value. Defaults to True.
        convert_children (bool or type): Whether to convert child dicts of type `child_dict` to configs
            with the same configuration.

            This will trigger a waterfall reaction across all child dicts. Won't convert dicts that
            are already configs. Apart from boolean, you can set it to any subclass of `Config` to use
            it for construction. Requires `nested` to be True. Defaults to False.
        as_attrs (bool): Whether to enable accessing dict keys via the dot notation.

            Enables autocompletion (but only during runtime!). Raises error in case of naming conflicts.
            Defaults to True if `frozen_keys` or `readonly`, otherwise False.

            To make nested dictionaries also accessible via the dot notation, wrap
            them with `child_dict` and set `convert_children` and `nested` to True.

    Defaults can be overridden with settings under `vectorbtpro._settings.config`.

    If another config is passed, its properties are copied over, but they can still be overridden
    with the arguments passed to the initializer.

    !!! note
        All arguments are applied only once during initialization.
    """

    def __init__(self, *args, options_: tp.KwargsLike = None, **kwargs) -> None:
        try:
            from vectorbtpro._settings import settings

            options_cfg = settings["config"]["options"]
        except ImportError:
            options_cfg = {}

        # Build dict
        if len(args) > 0 and isinstance(args[0], Config):
            cfg = args[0]
        else:
            cfg = None
        dct = dict(*args, **kwargs)
        if "options_" in dct:
            raise ValueError("options_ is an argument reserved for configs")
        if options_ is None:
            options_ = dict()
        else:
            options_ = dict(options_)

        # Resolve settings
        def _resolve_setting(pname: str, default: tp.Any, merge: bool = False) -> tp.Any:
            cfg_default = options_cfg.get(pname, None)
            if cfg is None:
                dct_p = None
            else:
                dct_p = cfg.get_option(pname)
            option = options_.pop(pname, None)

            if merge and isinstance(default, dict):
                return merge_dicts(default, cfg_default, dct_p, option)
            if option is not None:
                return option
            if dct_p is not None:
                return dct_p
            if cfg_default is not None:
                return cfg_default
            return default

        options_["reset_dct_copy_kwargs"] = merge_dicts(
            options_.get("copy_kwargs", None),
            options_.get("reset_dct_copy_kwargs", None),
        )
        reset_dct = _resolve_setting("reset_dct", None)
        pickle_reset_dct = _resolve_setting("pickle_reset_dct", False)
        frozen_keys = _resolve_setting("frozen_keys", False)
        readonly = _resolve_setting("readonly", False)
        nested = _resolve_setting("nested", True)
        convert_children = _resolve_setting("convert_children", False)
        as_attrs = _resolve_setting("as_attrs", frozen_keys or readonly)
        copy_kwargs = _resolve_setting(
            "copy_kwargs",
            dict(copy_mode="none", nested=nested),
            merge=True,
        )
        reset_dct_copy_kwargs = _resolve_setting(
            "reset_dct_copy_kwargs",
            dict(copy_mode="none" if readonly else "hybrid", nested=nested),
            merge=True,
        )
        if len(options_) > 0:
            raise ValueError(f"Unexpected config options: {options_}")

        # Copy dict
        dct = copy_dict(dict(dct), **copy_kwargs)

        # Convert child dicts
        if convert_children and nested:
            for k, v in dct.items():
                if isinstance(v, child_dict):
                    if isinstance(convert_children, bool):
                        config_cls = type(self)
                    elif issubclass(convert_children, Config):
                        config_cls = convert_children
                    else:
                        raise TypeError("Option 'convert_children' must be either boolean or a subclass of Config")
                    dct[k] = config_cls(
                        v,
                        options_=dict(
                            copy_kwargs=copy_kwargs,
                            reset_dct=None,
                            reset_dct_copy_kwargs=reset_dct_copy_kwargs,
                            pickle_reset_dct=pickle_reset_dct,
                            frozen_keys=frozen_keys,
                            readonly=readonly,
                            nested=nested,
                            convert_children=convert_children,
                            as_attrs=as_attrs,
                        ),
                    )

        # Copy initial config
        if reset_dct is None:
            reset_dct = dct
        reset_dct = copy_dict(dict(reset_dct), **reset_dct_copy_kwargs)

        dict.__init__(self, dct)

        self._options_ = dict(
            copy_kwargs=copy_kwargs,
            reset_dct=reset_dct,
            reset_dct_copy_kwargs=reset_dct_copy_kwargs,
            pickle_reset_dct=pickle_reset_dct,
            frozen_keys=frozen_keys,
            readonly=readonly,
            nested=nested,
            convert_children=convert_children,
            as_attrs=as_attrs,
        )

        # Set keys as attributes for autocomplete
        if as_attrs:
            self_dir = set(self.__dir__())
            for k, v in self.items():
                if k in self_dir:
                    raise ValueError(f"Key '{k}' shadows an attribute of the config. Disable option 'as_attrs'.")

    @property
    def options_(self) -> dict:
        """Config options."""
        return self._options_

    def get_option(self, k: str) -> tp.Any:
        """Get an option."""
        return self._options_[k]

    def set_option(self, k: str, v: tp.Any) -> None:
        """Set an option."""
        self._options_[k] = v

    def __getattr__(self, k: str) -> tp.Any:
        try:
            as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]
        except AttributeError:
            return object.__getattribute__(self, k)
        if as_attrs:
            try:
                return self.__getitem__(k)
            except KeyError:
                raise AttributeError
        return object.__getattribute__(self, k)

    def __setattr__(self, k: str, v: tp.Any, force: bool = False) -> None:
        try:
            as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]
        except AttributeError:
            return object.__setattr__(self, k, v)
        if as_attrs:
            return self.__setitem__(k, v, force=force)
        return object.__setattr__(self, k, v)

    def __delattr__(self, k: str, force: bool = False) -> None:
        try:
            as_attrs = object.__getattribute__(self, "_options_")["as_attrs"]
        except AttributeError:
            return object.__delattr__(self, k)
        if as_attrs:
            return self.__delitem__(k, force=force)
        return object.__delattr__(self, k)

    def __setitem__(self, k: str, v: tp.Any, force: bool = False) -> None:
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            if k not in self:
                raise KeyError(f"Config keys are frozen: key '{k}' not found")
        dict.__setitem__(self, k, v)

    def __delitem__(self, k: str, force: bool = False) -> None:
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError(f"Config keys are frozen")
        dict.__delitem__(self, k)

    def pop(self, k: str, v: tp.Any = MISSING, force: bool = False) -> tp.Any:
        """Remove and return the pair by the key."""
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError(f"Config keys are frozen")
        if v is MISSING:
            result = dict.pop(self, k)
        else:
            result = dict.pop(self, k, v)
        return result

    def popitem(self, force: bool = False) -> tp.Tuple[tp.Any, tp.Any]:
        """Remove and return some pair."""
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError(f"Config keys are frozen")
        result = dict.popitem(self)
        return result

    def clear(self, force: bool = False) -> None:
        """Remove all items."""
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        if not force and self.get_option("frozen_keys"):
            raise KeyError(f"Config keys are frozen")
        dict.clear(self)

    def update(self, *args, nested: tp.Optional[bool] = None, force: bool = False, **kwargs) -> None:
        """Update the config.

        See `update_dict`."""
        other = dict(*args, **kwargs)
        if nested is None:
            nested = self.get_option("nested")
        update_dict(self, other, nested=nested, force=force)

    def __copy__(self: ConfigT) -> ConfigT:
        """Shallow operation, primarily used by `copy.copy`.

        Does not take into account copy settings."""
        cls = type(self)
        self_copy = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = copy(v)
        self_copy.clear(force=True)
        self_copy.update(copy(dict(self)), nested=False, force=True)
        return self_copy

    def __deepcopy__(self: ConfigT, memo: tp.DictLike = None) -> ConfigT:
        """Deep operation, primarily used by `copy.deepcopy`.

        Does not take into account copy settings."""
        if memo is None:
            memo = {}
        cls = type(self)
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = deepcopy(v, memo)
        self_copy.clear(force=True)
        self_copy.update(deepcopy(dict(self), memo), nested=False, force=True)
        return self_copy

    def copy(
        self: ConfigT,
        reset_dct_copy_kwargs: tp.KwargsLike = None,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
    ) -> ConfigT:
        """Copy the instance.

        By default, copies in the same way as during the initialization."""
        if copy_mode is None:
            copy_mode = self.get_option("copy_kwargs")["copy_mode"]
            reset_dct_copy_mode = self.get_option("reset_dct_copy_kwargs")["copy_mode"]
        else:
            reset_dct_copy_mode = copy_mode
        if nested is None:
            nested = self.get_option("copy_kwargs")["nested"]
            reset_dct_nested = self.get_option("reset_dct_copy_kwargs")["nested"]
        else:
            reset_dct_nested = nested
        reset_dct_copy_kwargs = resolve_dict(reset_dct_copy_kwargs)
        if "copy_mode" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["copy_mode"] is not None:
                reset_dct_copy_mode = reset_dct_copy_kwargs["copy_mode"]
        if "nested" in reset_dct_copy_kwargs:
            if reset_dct_copy_kwargs["nested"] is not None:
                reset_dct_nested = reset_dct_copy_kwargs["nested"]

        self_copy = self.__copy__()
        reset_dct = copy_dict(
            dict(self_copy.get_option("reset_dct")),
            copy_mode=reset_dct_copy_mode,
            nested=reset_dct_nested,
        )
        self_copy.set_option("reset_dct", reset_dct)
        dct = copy_dict(dict(self_copy), copy_mode=copy_mode, nested=nested)
        self_copy.update(dct, nested=False, force=True)
        return self_copy

    def merge_with(
        self: ConfigT,
        other: InConfigLikeT,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> OutConfigLikeT:
        """Merge with another dict into one single dict.

        See `merge_dicts`."""
        if copy_mode is None:
            copy_mode = "shallow"
        if nested is None:
            nested = self.get_option("nested")
        return merge_dicts(self, other, copy_mode=copy_mode, nested=nested, **kwargs)

    def to_dict(self, nested: tp.Optional[bool] = None) -> dict:
        """Convert to dict."""
        return convert_to_dict(self, nested=nested)

    def reset(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Clears the config and updates it with the initial config.

        `reset_dct_copy_kwargs` override `reset_dct_copy_kwargs`."""
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.get_option("reset_dct_copy_kwargs"), reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self.get_option("reset_dct")), **reset_dct_copy_kwargs)
        self.clear(force=True)
        self.update(self.get_option("reset_dct"), nested=False, force=True)
        self.set_option("reset_dct", reset_dct)

    def make_checkpoint(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Replace `reset_dct` by the current state.

        `reset_dct_copy_kwargs` override `reset_dct_copy_kwargs`."""
        if not force and self.get_option("readonly"):
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.get_option("reset_dct_copy_kwargs"), reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self), **reset_dct_copy_kwargs)
        self.set_option("reset_dct", reset_dct)

    def load_update(
        self,
        path: tp.Optional[tp.PathLike] = None,
        clear: bool = False,
        update_options: bool = False,
        nested: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        """Load dumps from a file and update this instance in-place."""
        loaded = self.load(path=path, **kwargs)
        if clear:
            self.clear(force=True)
            if update_options:
                self.__dict__.clear()
        if nested is None:
            nested = self.get_option("nested")
        self.update(loaded, nested=nested, force=True)
        if update_options:
            self.__dict__.update(loaded.__dict__)

    def prettify(
        self,
        with_options: bool = False,
        replace: tp.DictLike = None,
        path: tp.PathLikeKey = None,
        htchar: str = "    ",
        lfchar: str = "\n",
        indent: int = 0,
    ) -> str:
        dct = dict(self)
        if with_options:
            dct["options_"] = self.options_
        if all([isinstance(k, str) and k.isidentifier() for k in dct]):
            return prettify_inited(
                type(self),
                dct,
                replace=replace,
                path=path,
                htchar=htchar,
                lfchar=lfchar,
                indent=indent,
            )
        return prettify_dict(self, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        check_options: bool = False,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        if _key is None:
            _key = type(self).__name__
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        if check_options and not is_deep_equal(
            self.options_,
            other.options_,
            _key=_key + ".options_",
            **kwargs,
        ):
            return False
        return is_deep_equal(
            dict(self),
            dict(other),
            _key=_key,
            **kwargs,
        )

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        init_kwargs = dict(self)
        init_kwargs["options_"] = dict(self.options_)
        if not self.get_option("pickle_reset_dct"):
            init_kwargs["options_"]["reset_dct"] = None
        return RecState(init_kwargs=init_kwargs)


class AtomicConfig(Config, atomic_dict):
    """Config that behaves like a single value when merging."""

    pass


class FrozenConfig(Config):
    """`Config` with `frozen_keys` flag set to True."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        options_["frozen_keys"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)


class ReadonlyConfig(Config):
    """`Config` with `readonly` flag set to True."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        options_["readonly"] = True
        Config.__init__(self, *args, options_=options_, **kwargs)


class HybridConfig(Config):
    """`Config` with `copy_kwargs` set to `copy_mode='hybrid'`."""

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        options_ = kwargs.pop("options_", None)
        if options_ is None:
            options_ = {}
        copy_kwargs = options_.pop("copy_kwargs", None)
        if copy_kwargs is None:
            copy_kwargs = {}
        copy_kwargs["copy_mode"] = "hybrid"
        options_["copy_kwargs"] = copy_kwargs
        Config.__init__(self, *args, options_=options_, **kwargs)


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class SettingsNotFoundError(KeyError):
    """Gets raised if settings could not be found."""

    pass


class SettingNotFoundError(KeyError):
    """Gets raised if a setting could not be found."""

    pass


HasSettingsT = tp.TypeVar("HasSettingsT", bound="HasSettings")


class HasSettings:
    """Class that has settings in `vectorbtpro._settings`."""

    _settings_path: tp.SettingsPath = None
    """Path(s) corresponding to this class in `vectorbtpro._settings`.

    Must be either string (one path) or a dictionary of path ids and paths.

    Lookup is done using `get_dict_item`."""

    @classmethod
    def get_path_settings(
        cls,
        path: tp.PathLikeKey,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> dict:
        """Get the settings under a path."""
        from vectorbtpro._settings import settings

        sub_path_settings = None
        if sub_path is not None:
            sub_path = combine_pathlike_keys(path, sub_path)
            try:
                sub_path_settings = cls.get_path_settings(sub_path)
            except SettingsNotFoundError as e:
                if sub_path_only:
                    raise SettingsNotFoundError(f"Found no settings under the path '{sub_path}'")
        try:
            path_settings = get_dict_item(settings, path)
        except KeyError as e:
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        if sub_path_settings is not None:
            return merge_dicts(path_settings, sub_path_settings)
        return path_settings

    @classmethod
    def resolve_settings_paths(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        super_first: bool = True,
    ) -> tp.List[tp.Tuple[tp.Type[HasSettingsT], str]]:
        """Resolve the settings paths associated with this class and its superclasses (if `inherit` is True)."""
        paths = []
        if inherit:
            if super_first:
                classes = cls.__mro__[::-1]
            else:
                classes = cls.__mro__
        else:
            classes = [cls]
        for i, cls_ in enumerate(classes):
            if issubclass(cls_, HasSettings):
                path = getattr(cls_, "_settings_path")
                if path is not None:
                    if isinstance(path, dict):
                        if path_id is None:
                            raise ValueError("Must specify path id")
                        if path_id not in path:
                            continue
                        path = path[path_id]
                        if path is None:
                            continue
                    paths.append((cls_, path))
        return paths

    @classmethod
    def get_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> dict:
        """Get the settings associated with this class and its superclasses (if `inherit` is True)."""
        paths = cls.resolve_settings_paths(path_id=path_id, inherit=inherit)
        if len(paths) == 0:
            if path_id is not None:
                raise SettingsNotFoundError(f"Found no settings associated with the path id '{path_id}'")
            else:
                raise SettingsNotFoundError(f"Found no settings associated with the class {cls.__name__}")
        setting_dicts = []
        for cls_, path in paths:
            path_settings = cls_.get_path_settings(path, sub_path=sub_path, sub_path_only=sub_path_only)
            setting_dicts.append(path_settings)
        if len(setting_dicts) == 1:
            return setting_dicts[0]
        return merge_dicts(*setting_dicts)

    @classmethod
    def has_path_settings(
        cls,
        path: tp.PathLikeKey,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether the settings under a path exist."""
        try:
            cls.get_path_settings(path, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except SettingsNotFoundError as e:
            return False

    @classmethod
    def has_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether there the settings associated with this class and its superclasses
        (if `inherit` is True) exist."""
        try:
            cls.get_settings(path_id=path_id, inherit=inherit, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except SettingsNotFoundError as e:
            return False

    @classmethod
    def get_path_setting(
        cls,
        path: tp.PathLikeKey,
        key: str,
        default: tp.Any = MISSING,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> tp.Any:
        """Get a value from the settings under a path."""
        from vectorbtpro._settings import settings

        if sub_path is not None:
            sub_path = combine_pathlike_keys(path, sub_path)
            try:
                sub_path_settings = cls.get_path_settings(sub_path)
                try:
                    return sub_path_settings[key]
                except KeyError as e:
                    if sub_path_only:
                        raise SettingNotFoundError(f"Found no key '{key}' in the settings under the path '{sub_path}'")
            except SettingsNotFoundError as e:
                if sub_path_only:
                    raise SettingsNotFoundError(f"Found no settings under the path '{sub_path}'")
        try:
            path_settings = get_dict_item(settings, path)
        except KeyError as e:
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        try:
            return path_settings[key]
        except KeyError as e:
            if default is MISSING:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' in the settings under the paths '{path}' and '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(f"Found no key '{key}' in the settings under the path '{path}'")
        return default

    @classmethod
    def get_setting(
        cls,
        key: str,
        default: tp.Any = MISSING,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> tp.Any:
        """Get a value under the settings associated with this class and its superclasses
        (if `inherit` is True)."""
        paths = cls.resolve_settings_paths(path_id=path_id, inherit=inherit, super_first=False)
        if len(paths) == 0:
            if path_id is not None:
                raise SettingsNotFoundError(f"Found no settings associated with the path id '{path_id}'")
            else:
                raise SettingsNotFoundError(f"Found no settings associated with the class {cls.__name__}")
        for cls_, path in paths:
            try:
                return cls_.get_path_setting(path, key, sub_path=sub_path, sub_path_only=sub_path_only)
            except (SettingsNotFoundError, SettingNotFoundError) as e:
                continue
        if default is MISSING:
            if path_id is not None:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the path id '{path_id}' "
                        f"and sub-path '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the path id '{path_id}'"
                    )
            else:
                if sub_path is not None:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the class {cls.__name__} "
                        f"and sub-path '{sub_path}'"
                    )
                else:
                    raise SettingNotFoundError(
                        f"Found no key '{key}' under the settings associated with the class {cls.__name__}"
                    )
        return default

    @classmethod
    def has_path_setting(
        cls,
        path: tp.PathLikeKey,
        key: str,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether the setting under a path exists."""
        try:
            cls.get_path_setting(path, key, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except (SettingsNotFoundError, SettingNotFoundError) as e:
            return False

    @classmethod
    def has_setting(
        cls,
        key: str,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> bool:
        """Return whether the settings associated with this class and its superclasses
        (if `inherit` is True) exists."""
        try:
            cls.get_setting(key, path_id=path_id, inherit=inherit, sub_path=sub_path, sub_path_only=sub_path_only)
            return True
        except (SettingsNotFoundError, SettingNotFoundError) as e:
            return False

    @classmethod
    def resolve_setting(
        cls,
        value: tp.Optional[tp.Any],
        key: str,
        merge: bool = False,
        default: tp.Any = MISSING,
        path_id: tp.Optional[tp.Hashable] = None,
        inherit: bool = True,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        sub_path_only: bool = False,
    ) -> tp.Any:
        """Resolve a value that has a key under the settings in `vectorbtpro._settings`
        associated with this class.

        If the provided value is None, returns the setting, otherwise the value.
        If the value is a dict and `merge` is True, merges it over the corresponding dict in the settings.

        If `sub_path` is provided, appends it to the resolved path and gives it more priority.
        If only the `sub_path` should be considered, set `sub_path_only` to True."""
        if merge:
            setting = cls.get_setting(
                key,
                default=default,
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
            )
            if setting is None or isinstance(setting, dict):
                if value is None or isinstance(value, dict):
                    return merge_dicts(setting, value)
            return value
        if value is None:
            return cls.get_setting(
                key,
                default=default,
                path_id=path_id,
                inherit=inherit,
                sub_path=sub_path,
                sub_path_only=sub_path_only,
            )
        return value

    @classmethod
    def set_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
        populate_: bool = False,
        **kwargs,
    ) -> None:
        """Set the settings in `vectorbtpro._settings` associated with this class.

        If the settings do not exist yet, pass `populate_=True`."""
        from vectorbtpro._settings import settings

        if isinstance(cls._settings_path, dict):
            if path_id is None:
                raise ValueError("Must specify path id")
            if path_id not in cls._settings_path:
                raise SettingsNotFoundError(f"Found no settings associated with the path id '{path_id}'")
            path = cls._settings_path[path_id]
        else:
            path = cls._settings_path
        if path is None:
            raise SettingsNotFoundError(f"Found no settings associated with the class {cls.__name__}")
        if sub_path is not None:
            path = combine_pathlike_keys(path, sub_path)
        cls_cfg = get_dict_item(settings, path, populate=populate_)
        for k, v in kwargs.items():
            if populate_:
                cls_cfg[k] = v
            else:
                if k not in cls_cfg:
                    raise SettingNotFoundError(f"Found no key '{k}' in the settings under the path '{path}'")
                if isinstance(cls_cfg[k], dict) and isinstance(v, dict):
                    cls_cfg[k] = merge_dicts(cls_cfg[k], v)
                else:
                    cls_cfg[k] = v

    @classmethod
    def reset_settings(
        cls,
        path_id: tp.Optional[tp.Hashable] = None,
        sub_path: tp.Optional[tp.PathLikeKey] = None,
    ) -> None:
        """Reset the settings in `vectorbtpro._settings` associated with this class."""
        from vectorbtpro._settings import settings

        if isinstance(cls._settings_path, dict):
            if path_id is None:
                raise ValueError("Must specify path id")
            if path_id not in cls._settings_path:
                raise SettingsNotFoundError(f"Found no settings associated with the path id '{path_id}'")
            path = cls._settings_path[path_id]
        else:
            path = cls._settings_path
        if path is None:
            raise SettingsNotFoundError(f"Found no settings associated with the class {cls.__name__}")
        if sub_path is not None:
            path = combine_pathlike_keys(path, sub_path)
        if not cls.has_path_settings(path):
            raise SettingsNotFoundError(f"Found no settings under the path '{path}'")
        cls_cfg = get_dict_item(settings, path)
        cls_cfg.reset(force=True)


class Configured(HasSettings, Cacheable, Comparable, Pickleable, Prettified, Chainable):
    """Class with an initialization config.

    All subclasses of `Configured` are initialized using `Config`, which makes it easier to pickle.

    Settings are defined under `vectorbtpro._settings.configured`.

    !!! warning
        If any attribute has been overwritten that isn't listed in `Configured._writeable_attrs`,
        or if any `Configured.__init__` argument depends upon global defaults,
        their values won't be copied over. Make sure to pass them explicitly to
        make that the saved & loaded / copied instance is resilient to any changes in globals."""

    _expected_keys: tp.ExpectedKeys = None
    """Set of expected keys."""

    _writeable_attrs: tp.WriteableAttrs = None
    """Set of writeable attributes that will be saved/copied along with the config."""

    def __init__(self, **config) -> None:
        from vectorbtpro._settings import settings

        configured_cfg = settings["configured"]

        check_expected_keys_ = config.get("check_expected_keys_", None)
        if self._expected_keys is None:
            check_expected_keys_ = False
        if check_expected_keys_ is None:
            check_expected_keys_ = configured_cfg["check_expected_keys_"]
        if check_expected_keys_:
            if isinstance(check_expected_keys_, bool):
                check_expected_keys_ = "raise"
            keys_diff = list(set(config.keys()).difference(self._expected_keys))
            if len(keys_diff) > 0:
                assert_in(check_expected_keys_, ("warn", "raise"))
                if check_expected_keys_ == "warn":
                    warnings.warn(f"{type(self).__name__} doesn't expect arguments {keys_diff}", stacklevel=2)
                else:
                    raise ValueError(f"{type(self).__name__} doesn't expect arguments {keys_diff}")

        self._config = Config(config, options_=configured_cfg["config"]["options"])

        Cacheable.__init__(self)

    @property
    def config(self) -> Config:
        """Initialization config."""
        return self._config

    @class_or_instancemethod
    def get_writeable_attrs(cls_or_self) -> tp.Optional[tp.Set[str]]:
        """Get set of attributes that are writeable by this class or by any of its base classes."""
        if isinstance(cls_or_self, type):
            cls = cls_or_self
        else:
            cls = type(cls_or_self)
        writeable_attrs = set()
        for cls in inspect.getmro(cls):
            if issubclass(cls, Configured) and cls._writeable_attrs is not None:
                writeable_attrs |= cls._writeable_attrs
        return writeable_attrs

    @classmethod
    def resolve_merge_kwargs(
        cls,
        *configs: tp.MaybeTuple[ConfigT],
        on_merge_conflict: tp.Union[str, dict] = "error",
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments for initializing `Configured` after merging."""
        if len(configs) == 1:
            configs = configs[0]
        configs = list(configs)

        all_keys = set()
        for config in configs:
            all_keys = all_keys.union(set(config.keys()))

        for k in all_keys:
            if k not in kwargs:
                v = None
                for i in range(len(configs)):
                    config = configs[i]
                    if isinstance(on_merge_conflict, dict):
                        if k in on_merge_conflict:
                            _on_merge_conflict = on_merge_conflict[k]
                        elif "_def" in on_merge_conflict:
                            _on_merge_conflict = on_merge_conflict["_def"]
                        else:
                            _on_merge_conflict = "error"
                    else:
                        _on_merge_conflict = on_merge_conflict
                    if _on_merge_conflict.lower() == "error":
                        if i == 0:
                            continue
                        same_k = True
                        try:
                            if k in config:
                                if not is_deep_equal(configs[0][k], config[k]):
                                    same_k = False
                            else:
                                same_k = False
                        except KeyError as e:
                            same_k = False
                        if not same_k:
                            raise ValueError(f"Objects to be merged must have compatible '{k}'. Pass to override.")
                        else:
                            v = config[k]
                    elif _on_merge_conflict.lower() == "first":
                        if k in config:
                            v = config[k]
                            break
                    elif _on_merge_conflict.lower() == "last":
                        if k in config:
                            v = config[k]
                    else:
                        raise ValueError(f"Invalid option on_merge_conflict='{_on_merge_conflict}'")
                kwargs[k] = v
        return kwargs

    def replace(
        self: ConfiguredT,
        copy_mode_: tp.Optional[str] = None,
        nested_: tp.Optional[bool] = None,
        cls_: tp.Optional[type] = None,
        copy_writeable_attrs_: tp.Optional[bool] = None,
        **new_config,
    ) -> ConfiguredT:
        """Create a new instance by copying and (optionally) changing the config.

        !!! warning
            This operation won't return a copy of the instance but a new instance
            initialized with the same config and writeable attributes (or their copy, depending on `copy_mode`)."""
        if cls_ is None:
            cls_ = type(self)
        if copy_writeable_attrs_ is None:
            copy_writeable_attrs_ = cls_ is type(self)
        new_config = self.config.merge_with(new_config, copy_mode=copy_mode_, nested=nested_)
        new_instance = cls_(**new_config)
        if copy_writeable_attrs_:
            for attr in self.get_writeable_attrs():
                attr_obj = getattr(self, attr)
                if isinstance(attr_obj, Config):
                    attr_obj = attr_obj.copy(copy_mode=copy_mode_, nested=nested_)
                else:
                    if copy_mode_ is not None:
                        if copy_mode_ == "hybrid":
                            attr_obj = copy(attr_obj)
                        elif copy_mode_ == "deep":
                            attr_obj = deepcopy(attr_obj)
                setattr(new_instance, attr, attr_obj)
        return new_instance

    def copy(
        self: ConfiguredT,
        copy_mode: tp.Optional[str] = None,
        nested: tp.Optional[bool] = None,
        cls: tp.Optional[type] = None,
    ) -> ConfiguredT:
        """Create a new instance by copying the config.

        See `Configured.replace`."""
        return self.replace(copy_mode_=copy_mode, nested_=nested, cls_=cls)

    def equals(
        self,
        other: tp.Any,
        check_types: bool = True,
        check_attrs: bool = True,
        check_options: bool = False,
        _key: tp.Optional[str] = None,
        **kwargs,
    ) -> bool:
        """Check two objects for equality."""
        if _key is None:
            _key = type(self).__name__
        if check_types and not is_deep_equal(
            self,
            other,
            _key=_key,
            only_types=True,
            **kwargs,
        ):
            return False
        if check_attrs:
            if not is_deep_equal(
                self.get_writeable_attrs(),
                other.get_writeable_attrs(),
                _key=_key + ".get_writeable_attrs()",
                **kwargs,
            ):
                return False
            for attr in self.get_writeable_attrs():
                if not is_deep_equal(
                    getattr(self, attr),
                    getattr(other, attr),
                    _key=_key + f".{attr}",
                    **kwargs,
                ):
                    return False
        return self.config.equals(
            other.config,
            check_types=check_types,
            check_options=check_options,
            _key=_key + ".config",
            **kwargs,
        )

    def update_config(self, *args, **kwargs) -> None:
        """Force-update the config."""
        self.config.update(*args, **kwargs, force=True)

    def prettify(self, **kwargs) -> str:
        return "%s(%s)" % (
            type(self).__name__,
            self.config.prettify(**kwargs)[len(type(self.config).__name__) + 1 : -1],
        )

    @property
    def rec_state(self) -> tp.Optional[RecState]:
        if self._writeable_attrs is not None:
            attr_dct = {k: getattr(self, k) for k in self._writeable_attrs}
        else:
            attr_dct = {}
        return RecState(init_kwargs=dict(self.config), attr_dct=attr_dct)
