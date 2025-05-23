# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for formatting."""

import inspect
import re

import attr
import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp

__all__ = [
    "prettify",
    "format_func",
    "pprint",
    "ptable",
    "phelp",
    "pdir",
]


def camel_to_snake_case(camel_str: str) -> str:
    """Convert a camel case string to a snake case string."""
    snake_str = re.sub(r"(?<!^)(?<![A-Z_])([A-Z])", r"_\1", camel_str).lower()
    if snake_str.startswith('_'):
        snake_str = snake_str[1:]
    return snake_str


class Prettified:
    """Abstract class that can be prettified."""

    def prettify(self, **kwargs) -> str:
        """Prettify this object.

        !!! warning
            Calling `prettify` can lead to an infinite recursion.
            Make sure to pre-process this object."""
        raise NotImplementedError

    def __str__(self) -> str:
        try:
            return self.prettify()
        except NotImplementedError:
            return repr(self)


def prettify_inited(
    cls: type,
    kwargs: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
) -> tp.Any:
    """Prettify an instance initialized with keyword arguments."""
    items = []
    for k, v in kwargs.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + "." + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(v, replace=replace, path=new_path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
        k_repr = repr(k)
        if isinstance(k, str):
            k_repr = k_repr[1:-1]
        items.append(lfchar + htchar * (indent + 1) + k_repr + "=" + new_v)
    if len(items) == 0:
        return "%s()" % (cls.__name__,)
    return "%s(%s)" % (cls.__name__, ",".join(items) + lfchar + htchar * indent)


def prettify_dict(
    obj: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
) -> tp.Any:
    """Prettify a dictionary."""
    if all([isinstance(k, str) and k.isidentifier() for k in obj]):
        return prettify_inited(
            type(obj),
            obj,
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
        )
    items = []
    for k, v in obj.items():
        if replace is None:
            replace = {}
        if path is None:
            new_path = k
        else:
            new_path = str(path) + "." + str(k)
        if new_path in replace:
            new_v = replace[new_path]
        else:
            new_v = prettify(v, replace=replace, path=new_path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
        items.append(lfchar + htchar * (indent + 1) + repr(k) + ": " + new_v)
    if type(obj) is dict:
        if len(items) == 0:
            return "{}"
        return "{%s}" % (",".join(items) + lfchar + htchar * indent)
    if len(items) == 0:
        return "%s({})" % (type(obj).__name__,)
    return "%s({%s})" % (type(obj).__name__, ",".join(items) + lfchar + htchar * indent)


def prettify(
    obj: tp.Any,
    replace: tp.DictLike = None,
    path: str = None,
    htchar: str = "    ",
    lfchar: str = "\n",
    indent: int = 0,
) -> tp.Any:
    """Prettify an object.

    Unfolds regular Python data structures such as lists and tuples.

    If `obj` is an instance of `Prettified`, calls `Prettified.prettify`."""
    if isinstance(obj, Prettified):
        return obj.prettify(replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)
    if attr.has(type(obj)):
        return prettify_inited(
            type(obj),
            attr.asdict(obj),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
        )
    if isinstance(obj, dict):
        return prettify_dict(obj, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent)
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return prettify_inited(
            type(obj),
            obj._asdict(),
            replace=replace,
            path=path,
            htchar=htchar,
            lfchar=lfchar,
            indent=indent,
        )
    if isinstance(obj, (tuple, list, set, frozenset)):
        items = []
        for v in obj:
            new_v = prettify(v, replace=replace, path=path, htchar=htchar, lfchar=lfchar, indent=indent + 1)
            items.append(lfchar + htchar * (indent + 1) + new_v)
        if type(obj) is tuple:
            if len(items) == 0:
                return "()"
            return "(%s)" % (",".join(items) + lfchar + htchar * indent)
        if type(obj) is list:
            if len(items) == 0:
                return "[]"
            return "[%s]" % (",".join(items) + lfchar + htchar * indent)
        if type(obj) is set:
            if len(items) == 0:
                return "set()"
            return "{%s}" % (",".join(items) + lfchar + htchar * indent)
        if len(items) == 0:
            return "%s([])" % (type(obj).__name__,)
        return "%s([%s])" % (type(obj).__name__, ",".join(items) + lfchar + htchar * indent)
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        items = []
        for k, v in dict(obj.fields).items():
            items.append(lfchar + htchar * (indent + 1) + repr((k, str(v[0]))))
        return "np.dtype([%s])" % (",".join(items) + lfchar + htchar * indent)
    if hasattr(obj, "shape") and isinstance(obj.shape, tuple) and len(obj.shape) > 0:
        module = type(obj).__module__
        qualname = type(obj).__qualname__
        return "<%s.%s object at %s with shape %s>" % (module, qualname, str(hex(id(obj))), obj.shape)
    if isinstance(obj, float):
        if np.isnan(obj):
            return "np.nan"
        if np.isposinf(obj):
            return "np.inf"
        if np.isneginf(obj):
            return "-np.inf"
    return repr(obj)


def pprint(*args, **kwargs) -> None:
    """Print the output of `prettify`."""
    print(prettify(*args, **kwargs))


def format_array(array: tp.ArrayLike, tabulate: tp.Optional[bool] = None, html: bool = False, **kwargs) -> str:
    """Format an array.

    Arguments are passed to `pd.DataFrame.to_string` or `pd.DataFrame.to_html` if `tabulate` is False,
    otherwise to `tabulate.tabulate`. If `tabulate` is None, will be set to True if the `tabulate`
    library is installed and `html` is disabled."""
    from vectorbtpro.base.reshaping import to_pd_array

    pd_array = to_pd_array(array)
    if tabulate is None:
        from vectorbtpro.utils.module_ import check_installed

        tabulate = check_installed("tabulate") and not html
    if tabulate:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("tabulate")
        from tabulate import tabulate

        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        if html:
            return tabulate(pd_array, headers="keys", tablefmt="html", **kwargs)
        return tabulate(pd_array, headers="keys", **kwargs)
    if html:
        if isinstance(pd_array, pd.Series):
            pd_array = pd_array.to_frame()
        return pd_array.to_html(**kwargs)
    return pd_array.to_string(**kwargs)


def ptable(*args, display_html: tp.Optional[bool] = None, **kwargs) -> None:
    """Print the output of `format_array`.

    If `display_html` is None, checks whether the code runs in a IPython notebook, and if so, becomes True.
    If `display_html` is True, displays the table in HTML format."""
    from vectorbtpro.utils.checks import is_notebook

    if display_html is None:
        display_html = is_notebook()
    if display_html:
        from IPython.display import display, HTML

        display(HTML(format_array(*args, html=True, **kwargs)))
    else:
        print(format_array(*args, **kwargs))


def format_parameter(param: inspect.Parameter, annotate: bool = False) -> str:
    """Format a parameter of a signature."""
    kind = param.kind
    formatted = param.name

    if annotate and param.annotation is not param.empty:
        formatted = "{}: {}".format(formatted, inspect.formatannotation(param.annotation))

    if param.default is not param.empty:
        if annotate and param.annotation is not param.empty:
            formatted = "{} = {}".format(formatted, repr(param.default))
        else:
            formatted = "{}={}".format(formatted, repr(param.default))

    if kind == param.VAR_POSITIONAL:
        formatted = "*" + formatted
    elif kind == param.VAR_KEYWORD:
        formatted = "**" + formatted

    return formatted


def format_signature(
    signature: inspect.signature,
    annotate: bool = False,
    start: str = "\n    ",
    separator: str = ",\n    ",
    end: str = "\n",
) -> str:
    """Format a signature."""
    result = []
    render_pos_only_separator = False
    render_kw_only_separator = True

    for param in signature.parameters.values():
        formatted = format_parameter(param, annotate=annotate)

        kind = param.kind

        if kind == param.POSITIONAL_ONLY:
            render_pos_only_separator = True
        elif render_pos_only_separator:
            result.append("/")
            render_pos_only_separator = False

        if kind == param.VAR_POSITIONAL:
            render_kw_only_separator = False
        elif kind == param.KEYWORD_ONLY and render_kw_only_separator:
            result.append("*")
            render_kw_only_separator = False

        result.append(formatted)

    if render_pos_only_separator:
        result.append("/")

    if len(result) == 0:
        rendered = "()"
    else:
        rendered = "({})".format(start + separator.join(result) + end)

    if annotate and signature.return_annotation is not inspect._empty:
        anno = inspect.formatannotation(signature.return_annotation)
        rendered += " -> {}".format(anno)

    return rendered


def format_func(func: tp.Callable, incl_doc: bool = True, **kwargs) -> str:
    """Format a function."""
    if inspect.isclass(func):
        func_name = func.__name__ + ".__init__"
        func = func.__init__
    elif inspect.ismethod(func) and hasattr(func, "__self__"):
        if isinstance(func.__self__, type):
            func_name = func.__self__.__name__ + "." + func.__name__
        else:
            func_name = type(func.__self__).__name__ + "." + func.__name__
    else:
        func_name = func.__qualname__
    if incl_doc and func.__doc__ is not None:
        return "{}{}:\n{}".format(
            func_name,
            format_signature(inspect.signature(func), **kwargs),
            "    " + "\n    ".join(inspect.cleandoc(func.__doc__).splitlines()),
        )
    return "{}{}".format(
        func_name,
        format_signature(inspect.signature(func), **kwargs),
    )


def phelp(*args, **kwargs) -> None:
    """Print the output of `format_func`."""
    print(format_func(*args, **kwargs))


def pdir(*args, **kwargs) -> None:
    """Print the output of `vectorbtpro.utils.attr_.parse_attrs`."""
    from vectorbtpro.utils.attr_ import parse_attrs

    ptable(parse_attrs(*args, **kwargs))
