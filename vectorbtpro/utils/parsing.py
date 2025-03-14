# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for parsing."""

import ast
import contextlib
import inspect
import io
import re
import sys
import warnings

from vectorbtpro import _typing as tp
from vectorbtpro.utils.annotations import get_annotations, VarArgs, VarKwargs
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "Regex",
    "PrintsSuppressed",
    "WarningsFiltered",
]


@define
class Regex(DefineMixin):
    """Class for matching a regular expression."""

    pattern: str = define.field()
    """Pattern."""

    flags: int = define.field(default=0)
    """Flags."""

    def matches(self, string: str) -> bool:
        """Return whether the string matches the regular expression pattern."""
        return re.match(self.pattern, string, self.flags) is not None


def get_func_kwargs(func: tp.Callable) -> dict:
    """Get keyword arguments with defaults of a function."""
    signature = inspect.signature(func)
    return {k: v.default for k, v in signature.parameters.items() if v.default is not inspect.Parameter.empty}


def get_func_arg_names(
    func: tp.Callable,
    arg_kind: tp.Optional[tp.MaybeTuple[int]] = None,
    req_only: bool = False,
    opt_only: bool = False,
) -> tp.List[str]:
    """Get argument names of a function."""
    signature = inspect.signature(func)
    if arg_kind is not None and isinstance(arg_kind, int):
        arg_kind = (arg_kind,)
    arg_names = []
    for p in signature.parameters.values():
        if arg_kind is None:
            if p.kind == p.VAR_POSITIONAL or p.kind == p.VAR_KEYWORD:
                continue
        else:
            if p.kind not in arg_kind:
                continue
        if req_only and p.default is not inspect.Parameter.empty:
            continue
        if opt_only and p.default is inspect.Parameter.empty:
            continue
        arg_names.append(p.name)
    return arg_names


def extend_args(func: tp.Callable, args: tp.Args, kwargs: tp.Kwargs, **with_kwargs) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Extend arguments and keyword arguments with other arguments."""
    kwargs = dict(kwargs)
    new_args = ()
    new_kwargs = dict()
    signature = inspect.signature(func)
    for p in signature.parameters.values():
        if p.kind == p.VAR_POSITIONAL:
            new_args += args
            args = ()
            continue
        if p.kind == p.VAR_KEYWORD:
            for k in list(kwargs.keys()):
                new_kwargs[k] = kwargs.pop(k)
            continue

        arg_name = p.name.lower()
        took_from_args = False
        if arg_name not in kwargs and arg_name in with_kwargs:
            arg_value = with_kwargs[arg_name]
        elif len(args) > 0:
            arg_value = args[0]
            args = args[1:]
            took_from_args = True
        elif arg_name in kwargs:
            arg_value = kwargs.pop(arg_name)
        else:
            continue
        if p.kind == p.POSITIONAL_ONLY or len(args) > 0 or took_from_args:
            new_args += (arg_value,)
        else:
            new_kwargs[arg_name] = arg_value

    return new_args + args, {**new_kwargs, **kwargs}


def annotate_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    only_passed: bool = False,
    allow_partial: bool = False,
    attach_annotations: bool = False,
    flatten: bool = False,
) -> tp.AnnArgs:
    """Annotate arguments and keyword arguments using the function's signature.

    If `allow_partial` is True, required arguments that weren't provided won't raise an error.
    But regardless of `allow_partial`, arguments that aren't in the signature will still raise an error."""
    kwargs = dict(kwargs)
    signature = inspect.signature(func)
    if not allow_partial:
        signature.bind(*args, **kwargs)
    ann_args = dict()
    if attach_annotations:
        annotations = get_annotations(func)
    else:
        annotations = dict()

    last_pos = None
    var_positional = False
    var_keyword = False
    for p in signature.parameters.values():
        if p.kind == p.POSITIONAL_ONLY:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
                last_pos = p.name
            elif not only_passed:
                if allow_partial:
                    ann_args[p.name] = dict(kind=p.kind)
                else:
                    raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.VAR_POSITIONAL:
            var_positional = True
            if len(args) > 0 or not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=args)
                args = ()
                last_pos = p.name
        elif p.kind == p.POSITIONAL_OR_KEYWORD:
            if len(args) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=args[0])
                args = args[1:]
                last_pos = p.name
            elif p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                if p.default is not p.empty:
                    ann_args[p.name] = dict(kind=p.kind, value=p.default)
                else:
                    if allow_partial:
                        ann_args[p.name] = dict(kind=p.kind)
                    else:
                        raise TypeError(f"missing a required argument: '{p.name}'")
        elif p.kind == p.KEYWORD_ONLY:
            if p.name in kwargs:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs.pop(p.name))
            elif not only_passed:
                ann_args[p.name] = dict(kind=p.kind, value=p.default)
        else:
            var_keyword = True
            if not only_passed or len(kwargs) > 0:
                ann_args[p.name] = dict(kind=p.kind, value=kwargs)
        if p.name in ann_args and p.name in annotations:
            ann_args[p.name]["annotation"] = annotations[p.name]

    if not var_positional:
        if len(args) == 1:
            raise TypeError(f"{func.__name__}() got an unexpected positional argument after '{last_pos}'")
        if len(args) > 1:
            raise TypeError(f"{func.__name__}() got {len(args)} unexpected positional arguments after '{last_pos}'")
    if not var_keyword:
        if len(kwargs) == 1:
            raise TypeError(f"{func.__name__}() got an unexpected keyword argument '{list(kwargs.keys())[0]}'")
        if len(kwargs) > 1:
            raise TypeError(f"{func.__name__}() got unexpected keyword arguments {list(kwargs.keys())}")
    if flatten:
        return flatten_ann_args(ann_args)
    return ann_args


def ann_args_to_args(ann_args: tp.AnnArgs) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Convert annotated arguments back to positional and keyword arguments."""
    args = ()
    kwargs = {}
    p = inspect.Parameter
    for k, v in ann_args.items():
        if v["kind"] in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD):
            args += (v["value"],)
        elif v["kind"] == p.VAR_POSITIONAL:
            args += v["value"]
        elif v["kind"] == p.KEYWORD_ONLY:
            kwargs[k] = v["value"]
        else:
            for _k, _v in v["value"].items():
                kwargs[_k] = _v
    return args, kwargs


def flat_ann_args_to_args(ann_args: tp.AnnArgs) -> tp.Tuple[tp.Args, tp.Kwargs]:
    """Convert flattened annotated arguments back to positional and keyword arguments."""
    return ann_args_to_args(flatten_ann_args(ann_args))


def flatten_ann_args(ann_args: tp.AnnArgs) -> tp.FlatAnnArgs:
    """Flatten annotated arguments."""
    flat_ann_args = {}
    for arg_name, ann_arg in ann_args.items():
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            for i, v in enumerate(ann_arg["value"]):
                dct = dict(var_name=arg_name, kind=ann_arg["kind"], value=v)
                if "annotation" in ann_arg:
                    if isinstance(ann_arg["annotation"], VarArgs):
                        dct["annotation"] = ann_arg["annotation"].args[i]
                        dct["var_annotation"] = ann_arg["annotation"]
                    else:
                        if isinstance(ann_arg["annotation"], VarKwargs):
                            raise TypeError("VarKwargs used for variable positional arguments")
                        dct["annotation"] = ann_arg["annotation"]
                new_arg_name = f"{arg_name}_{i}"
                if new_arg_name in flat_ann_args:
                    raise ValueError(f"Unpacked key {new_arg_name} already exists in annotated arguments")
                flat_ann_args[new_arg_name] = dct
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            for var_arg_name, var_value in ann_arg["value"].items():
                dct = dict(var_name=arg_name, kind=ann_arg["kind"], value=var_value)
                if "annotation" in ann_arg:
                    if isinstance(ann_arg["annotation"], VarKwargs):
                        dct["annotation"] = ann_arg["annotation"].kwargs[var_arg_name]
                        dct["var_annotation"] = ann_arg["annotation"]
                    else:
                        if isinstance(ann_arg["annotation"], VarArgs):
                            raise TypeError("VarArgs used for variable keyword arguments")
                        dct["annotation"] = ann_arg["annotation"]
                if var_arg_name in flat_ann_args:
                    raise ValueError(f"Unpacked key {var_arg_name} already exists in annotated arguments")
                flat_ann_args[var_arg_name] = dct
        else:
            dct = dict(kind=ann_arg["kind"])
            if "value" in ann_arg:
                dct["value"] = ann_arg["value"]
            if "annotation" in ann_arg:
                dct["annotation"] = ann_arg["annotation"]
            flat_ann_args[arg_name] = dct
    return flat_ann_args


def unflatten_ann_args(flat_ann_args: tp.FlatAnnArgs, partial_ann_args: tp.Optional[tp.AnnArgs] = None) -> tp.AnnArgs:
    """Unflatten annotated arguments."""
    ann_args = dict()
    for arg_name, ann_arg in flat_ann_args.items():
        ann_arg = dict(ann_arg)
        if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                dct = dict(value=(), kind=ann_arg["kind"])
                if "var_annotation" in ann_arg:
                    dct["annotation"] = ann_arg["var_annotation"]
                elif "annotation" in ann_arg:
                    dct["annotation"] = ann_arg["annotation"]
                ann_args[var_arg_name] = dct
            ann_args[var_arg_name]["value"] = ann_args[var_arg_name]["value"] + (ann_arg["value"],)
        elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
            var_arg_name = ann_arg.pop("var_name")
            if var_arg_name not in ann_args:
                dct = dict(value={}, kind=ann_arg["kind"])
                if "var_annotation" in ann_arg:
                    dct["annotation"] = ann_arg["var_annotation"]
                elif "annotation" in ann_arg:
                    dct["annotation"] = ann_arg["annotation"]
                ann_args[var_arg_name] = dct
            ann_args[var_arg_name]["value"][arg_name] = ann_arg["value"]
        else:
            ann_args[arg_name] = ann_arg
    if partial_ann_args is not None:
        if ann_args.keys() > partial_ann_args.keys():
            raise ValueError("Unflattened annotated arguments contain unexpected keys")
        for k, v in partial_ann_args.items():
            if k not in ann_args:
                ann_args[k] = v
        new_ann_args = dict()
        for k in partial_ann_args:
            new_ann_args[k] = ann_args[k]
        return new_ann_args
    return ann_args


def match_flat_ann_arg(
    flat_ann_args: tp.FlatAnnArgs,
    query: tp.AnnArgQuery,
    return_name: bool = False,
    return_index: bool = False,
) -> tp.Any:
    """Match an argument from flattened annotated arguments.

    A query can be an integer indicating the position of the argument, or a string containing the name
    of the argument, or a regular expression for matching the name of the argument.

    If multiple arguments were matched, returns the first one.

    The position can stretch over any variable argument."""
    if return_name and return_index:
        raise ValueError("Either return_name or return_index can be provided, not both")
    for i, (arg_name, ann_arg) in enumerate(flat_ann_args.items()):
        if (
            (isinstance(query, int) and query == i)
            or (isinstance(query, str) and query == arg_name)
            or (isinstance(query, Regex) and query.matches(arg_name))
        ):
            if return_name:
                return arg_name
            if return_index:
                return i
            return ann_arg["value"]
    raise KeyError(f"Query '{query}' could not be matched with any argument")


def match_ann_arg(
    ann_args: tp.AnnArgs,
    query: tp.AnnArgQuery,
    return_name: bool = False,
    return_index: bool = False,
) -> tp.Any:
    """Match an argument from annotated arguments.

    See `match_flat_ann_arg` for matching logic."""
    return match_flat_ann_arg(
        flatten_ann_args(ann_args),
        query,
        return_name=return_name,
        return_index=return_index,
    )


def match_and_set_flat_ann_arg(
    flat_ann_args: tp.FlatAnnArgs,
    query: tp.AnnArgQuery,
    new_value: tp.Any,
) -> None:
    """Match an argument from flattened annotated arguments and set it to a new value.

    See `match_flat_ann_arg` for matching logic."""
    matched = False
    for i, (arg_name, ann_arg) in enumerate(flat_ann_args.items()):
        if (
            (isinstance(query, int) and query == i)
            or (isinstance(query, str) and query == arg_name)
            or (isinstance(query, Regex) and query.matches(arg_name))
        ):
            ann_arg["value"] = new_value
            matched = True
    if not matched:
        raise KeyError(f"Query '{query}' could not be matched with any argument")


def ignore_flat_ann_args(flat_ann_args: tp.FlatAnnArgs, ignore_args: tp.Iterable[tp.AnnArgQuery]) -> tp.FlatAnnArgs:
    """Ignore flattened annotated arguments."""
    new_flat_ann_args = {}
    for i, (arg_name, arg) in enumerate(flat_ann_args.items()):
        arg_matched = False
        for ignore_arg in ignore_args:
            if (
                (isinstance(ignore_arg, int) and ignore_arg == i)
                or (isinstance(ignore_arg, str) and ignore_arg == arg_name)
                or (isinstance(ignore_arg, Regex) and ignore_arg.matches(arg_name))
            ):
                arg_matched = True
                break
        if not arg_matched:
            new_flat_ann_args[arg_name] = arg
    return new_flat_ann_args


class UnhashableArgsError(Exception):
    """Unhashable arguments error."""

    pass


def hash_args(
    func: tp.Callable,
    args: tp.Args,
    kwargs: tp.Kwargs,
    ignore_args: tp.Optional[tp.Iterable[tp.AnnArgQuery]] = None,
) -> int:
    """Get hash of arguments.

    Use `ignore_args` to provide a sequence of queries for arguments that should be ignored."""
    if ignore_args is None:
        ignore_args = []
    ann_args = annotate_args(func, args, kwargs, only_passed=True)
    flat_ann_args = flatten_ann_args(ann_args)
    if len(ignore_args) > 0:
        flat_ann_args = ignore_flat_ann_args(flat_ann_args, ignore_args)
    try:
        return hash(tuple(map(lambda x: (x[0], x[1]["value"]), flat_ann_args.items())))
    except TypeError:
        raise UnhashableArgsError


def get_expr_var_names(expression: str) -> tp.List[str]:
    """Get variable names listed in the expression."""
    return [node.id for node in ast.walk(ast.parse(expression)) if type(node) is ast.Name]


def get_context_vars(
    var_names: tp.Iterable[str],
    frames_back: int = 0,
    local_dict: tp.Optional[tp.Mapping] = None,
    global_dict: tp.Optional[tp.Mapping] = None,
) -> tp.List[tp.Any]:
    """Get variables from the local/global context."""
    call_frame = sys._getframe(frames_back + 1)
    clear_local_dict = False
    if local_dict is None:
        local_dict = call_frame.f_locals
        clear_local_dict = True
    try:
        frame_globals = call_frame.f_globals
        if global_dict is None:
            global_dict = frame_globals
        clear_local_dict = clear_local_dict and frame_globals is not local_dict
        args = []
        for var_name in var_names:
            try:
                a = local_dict[var_name]
            except KeyError:
                a = global_dict[var_name]
            args.append(a)
    finally:
        # See https://github.com/pydata/numexpr/issues/310
        if clear_local_dict:
            local_dict.clear()
    return args


def suppress_stdout(func: tp.Callable) -> tp.Callable:
    """Suppress output from a function."""

    def wrapper(*a, **ka):
        with contextlib.redirect_stdout(io.StringIO()):
            return func(*a, **ka)

    return wrapper


def warn_stdout(func: tp.Callable) -> tp.Callable:
    """Supress and convert to a warning output from a function."""

    def wrapper(*a, **ka):
        with contextlib.redirect_stdout(io.StringIO()) as f:
            out = func(*a, **ka)
        s = f.getvalue()
        if len(s) > 0:
            warnings.warn(s, stacklevel=2)
        return out

    return wrapper


PrintsSuppressedT = tp.TypeVar("PrintsSuppressedT", bound="PrintsSuppressed")


class PrintsSuppressed(contextlib.redirect_stdout):
    """Context manager to ignore print statements."""

    def __new__(cls, *args, **kwargs) -> PrintsSuppressedT:
        return cls(io.StringIO(), *args, **kwargs)


WarningsFilteredT = tp.TypeVar("WarningsFilteredT", bound="WarningsFiltered")


class WarningsFiltered(warnings.catch_warnings):
    """Context manager to ignore warnings."""

    def __init__(self, entries: tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]] = "ignore", **kwargs) -> None:
        warnings.catch_warnings.__init__(self, **kwargs)
        self._entries = entries

    @property
    def entries(self) -> tp.Optional[tp.MaybeSequence[tp.Union[str, tp.Kwargs]]]:
        """One or more simple entries to add into the list of warnings filters."""
        return self._entries

    def __enter__(self: WarningsFilteredT) -> WarningsFilteredT:
        warnings.catch_warnings.__enter__(self)
        if self.entries is not None:
            if isinstance(self.entries, (str, dict)):
                entry = self.entries
                if isinstance(entry, str):
                    entry = dict(action=entry)
                warnings.simplefilter(**entry)
            else:
                for entry in self.entries:
                    if isinstance(entry, str):
                        entry = dict(action=entry)
                    warnings.simplefilter(**entry)
        return self
