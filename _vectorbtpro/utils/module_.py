# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for modules."""

import importlib
import importlib.util
import inspect
import pkgutil
import sys
import urllib.request
import warnings
import webbrowser
from pathlib import Path
from types import ModuleType, FunctionType

from vectorbtpro import _typing as tp
from vectorbtpro._opt_deps import opt_dep_config
from vectorbtpro.utils.config import HybridConfig

__all__ = [
    "import_module_from_path",
    "get_refname",
    "imlucky",
    "get_api_ref",
    "open_api_ref",
]

__pdoc__ = {}

package_shortcut_config = HybridConfig(dict(vbt="vectorbtpro", pd="pandas", np="numpy", nb="numba"))
"""_"""

__pdoc__[
    "package_shortcut_config"
] = f"""Config for package shortcuts.

```python
{package_shortcut_config.prettify()}
```
"""


def get_module(obj: tp.Any) -> ModuleType:
    """Get module of an object."""
    return inspect.getmodule(inspect.unwrap(obj))


def is_from_module(obj: tp.Any, module: ModuleType) -> bool:
    """Return whether `obj` is from module `module`."""
    mod = get_module(obj)
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(
    module_or_name: tp.Union[str, ModuleType],
    whitelist: tp.Optional[tp.List[str]] = None,
    blacklist: tp.Optional[tp.List[str]] = None,
) -> tp.List[str]:
    """List the names of all public functions and classes defined in the module `module_name`.

    Includes the names listed in `whitelist` and excludes the names listed in `blacklist`."""
    if whitelist is None:
        whitelist = []
    if blacklist is None:
        blacklist = []
    if isinstance(module_or_name, str):
        module = sys.modules[module_or_name]
    else:
        module = module_or_name
    return [
        name
        for name, obj in inspect.getmembers(module)
        if (
            not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
            and name not in blacklist
        )
        or name in whitelist
    ]


def search_package_for_funcs(
    package: tp.Union[str, ModuleType],
    blacklist: tp.Optional[tp.Sequence[str]] = None,
) -> tp.Dict[str, FunctionType]:
    """Search a package for all functions."""
    if blacklist is None:
        blacklist = []
    if isinstance(package, str):
        package = importlib.import_module(package)
    results = {}
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if ".".join(name.split(".")[:-1]) != package.__name__:
            continue
        try:
            if name in blacklist:
                continue
            module = importlib.import_module(name)
            for attr in dir(module):
                if not attr.startswith("_") and isinstance(getattr(module, attr), FunctionType):
                    results[attr] = getattr(module, attr)
            if is_pkg:
                results.update(search_package_for_funcs(name, blacklist=blacklist))
        except ModuleNotFoundError as e:
            pass
    return results


def find_class(path: str) -> tp.Optional[tp.Type]:
    """Find the class by its path."""
    try:
        path_parts = path.split(".")
        module_path = ".".join(path_parts[:-1])
        class_name = path_parts[-1]
        if module_path.startswith("vectorbtpro.indicators.factory"):
            import vectorbtpro as vbt

            return getattr(vbt, path_parts[-2])(class_name)
        module = importlib.import_module(module_path)
        if hasattr(module, class_name):
            return getattr(module, class_name)
    except Exception as e:
        pass
    return None


def check_installed(pkg_name: str) -> bool:
    """Check if a package is installed."""
    return importlib.util.find_spec(pkg_name) is not None


def get_installed_overview() -> tp.Dict[str, bool]:
    """Get an overview of installed packages in `opt_dep_config`."""
    return {pkg_name: check_installed(pkg_name) for pkg_name in opt_dep_config.keys()}


def get_package_meta(pkg_name: str) -> dict:
    """Get metadata of a package."""
    if pkg_name not in opt_dep_config:
        raise KeyError(f"Package '{pkg_name}' not found in opt_dep_config")
    dist_name = opt_dep_config[pkg_name].get("dist_name", pkg_name)
    version = opt_dep_config[pkg_name].get("version", "")
    link = opt_dep_config[pkg_name]["link"]
    return dict(dist_name=dist_name, version=version, link=link)


def assert_can_import(pkg_name: str) -> None:
    """Assert that a package can be imported.

    Must be listed in `opt_dep_config`."""
    from importlib.metadata import version as get_version

    metadata = get_package_meta(pkg_name)
    dist_name = metadata["dist_name"]
    version = version_str = metadata["version"]
    link = metadata["link"]
    if not check_installed(pkg_name):
        raise ImportError(f"Please install {dist_name}{version_str} - {link}")
    if version != "":
        actual_version_parts = get_version(dist_name).split(".")
        actual_version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", actual_version_parts)
        actual_version = "(" + ",".join(actual_version_parts) + ")"
        if version[0].isdigit():
            operator = "=="
        else:
            operator = version[:2]
            version_parts = version[2:].split(".")
            version_parts = map(lambda x: x if x.isnumeric() else f"'{x}'", version_parts)
            version = "(" + ",".join(version_parts) + ")"
        if not eval(f"{actual_version} {operator} {version}"):
            raise ImportError(f"Please install {dist_name}{version_str} - {link}")


def assert_can_import_any(*pkg_names: str) -> None:
    """Assert that any from packages can be imported.

    Must be listed in `opt_dep_config`."""
    if len(pkg_names) == 1:
        return assert_can_import(pkg_names[0])
    for pkg_name in pkg_names:
        try:
            return assert_can_import(pkg_name)
        except ImportError:
            pass
    requirements = []
    for pkg_name in pkg_names:
        metadata = get_package_meta(pkg_name)
        dist_name = metadata["dist_name"]
        version_str = metadata["version"]
        link = metadata["link"]
        requirements.append(f"{dist_name}{version_str} - {link}")
    raise ImportError(f"Please install any of " + ", ".join(requirements))


def warn_cannot_import(pkg_name: str) -> bool:
    """Warn if a package is cannot be imported.

    Must be listed in `opt_dep_config`."""
    try:
        assert_can_import(pkg_name)
        return False
    except ImportError as e:
        warnings.warn(str(e), stacklevel=2)
        return True


def import_module_from_path(module_path: tp.PathLike, reload: bool = False) -> ModuleType:
    """Import the module from a path."""
    module_path = Path(module_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, str(module_path.resolve()))
    module = importlib.util.module_from_spec(spec)
    if module.__name__ in sys.modules and not reload:
        return sys.modules[module.__name__]
    spec.loader.exec_module(module)
    sys.modules[module.__name__] = module
    return module


def get_method_class(meth: tp.Callable) -> tp.Type:
    """Get the class of a method."""
    if inspect.ismethod(meth) or (
        inspect.isbuiltin(meth)
        and getattr(meth, "__self__", None) is not None
        and getattr(meth.__self__, "__class__", None)
    ):
        for cls in inspect.getmro(meth.__self__.__class__):
            if meth.__name__ in cls.__dict__:
                return cls
        meth = getattr(meth, "__func__", meth)
    if inspect.isfunction(meth):
        cls = getattr(get_module(meth), meth.__qualname__.split(".<locals>", 1)[0].rsplit(".", 1)[0], None)
        if isinstance(cls, type):
            return cls
    return getattr(meth, "__objclass__", None)


def parse_refname(obj: tp.Any) -> str:
    """Get the reference name of an object."""
    from vectorbtpro.utils.decorators import classproperty, class_or_instanceproperty, custom_property

    if inspect.ismodule(obj):
        return obj.__name__
    if inspect.isclass(obj):
        return obj.__module__ + "." + obj.__qualname__
    if inspect.ismethod(obj) or inspect.isfunction(obj):
        cls = get_method_class(obj)
        if cls is not None:
            return parse_refname(cls) + "." + obj.__name__
        if hasattr(obj, "func"):
            return parse_refname(obj.func)
    if isinstance(obj, (classproperty, class_or_instanceproperty, custom_property)):
        return parse_refname(obj.func)
    if isinstance(obj, property):
        return parse_refname(obj.fget)
    if hasattr(obj, "__name__"):
        module = get_module(obj)
        if module is not None:
            if obj.__name__ in module.__dict__:
                return parse_refname(module) + "." + obj.__name__
    module = get_module(obj)
    if module is not None:
        for k, v in module.__dict__.items():
            if obj is v:
                return parse_refname(module) + "." + k
    return parse_refname(type(obj))


def get_refname_module_and_qualname(
    refname: str,
    module: tp.Optional[ModuleType] = None,
) -> tp.Tuple[tp.Optional[ModuleType], tp.Optional[str]]:
    """Get the module and the qualified name from a reference name."""
    refname_parts = refname.split(".")
    if module is None:
        module = importlib.import_module(refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return get_refname_module_and_qualname(".".join(refname_parts), module=module)
    elif inspect.ismodule(getattr(module, refname_parts[0])):
        module = getattr(module, refname_parts[0])
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module, None
        return get_refname_module_and_qualname(".".join(refname_parts), module=module)
    else:
        return module, ".".join(refname_parts)


def resolve_refname(refname: str, module: tp.Union[None, str, ModuleType] = None) -> tp.Optional[tp.MaybeList[str]]:
    """Resolve a reference name."""
    if refname == "":
        if module is None:
            return None
        if isinstance(module, str):
            return module
        return module.__name__

    _module = module
    refname_parts = refname.split(".")
    if module is None:
        if refname_parts[0] in package_shortcut_config:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
            module = importlib.import_module(refname_parts[0])
            refname_parts = refname_parts[1:]
        else:
            try:
                module = importlib.import_module(refname_parts[0])
                refname_parts = refname_parts[1:]
            except ImportError:
                module = "vectorbtpro"
    if isinstance(module, str):
        module = importlib.import_module(module)
    if len(refname_parts) == 0:
        return module.__name__
    if refname_parts[0] in package_shortcut_config:
        if package_shortcut_config[refname_parts[0]] == module.__name__:
            refname_parts[0] = package_shortcut_config[refname_parts[0]]
    if refname_parts[0] == module.__name__ and refname_parts[0] not in module.__dict__:
        refname_parts = refname_parts[1:]
        if len(refname_parts) == 0:
            return module.__name__

    if refname_parts[0] in module.__dict__:
        obj = module.__dict__[refname_parts[0]]
        if inspect.ismodule(obj):
            parent_module = ".".join(obj.__name__.split(".")[:-1])
        else:
            parent_module = get_module(obj)
            if parent_module is not None:
                if refname_parts[0] in parent_module.__dict__:
                    parent_module = parent_module.__name__
                else:
                    parent_module = None
        if parent_module is None or parent_module == module.__name__:
            if inspect.ismodule(obj):
                module = getattr(module, refname_parts[0])
                refname_parts = refname_parts[1:]
                return resolve_refname(".".join(refname_parts), module=module)
            if hasattr(obj, "__name__") and obj.__name__ in module.__dict__:
                obj = module.__dict__[obj.__name__]
                refname_parts[0] = obj.__name__
            if len(refname_parts) == 1:
                return module.__name__ + "." + refname_parts[0]
            if not isinstance(obj, type):
                cls = type(obj)
            else:
                cls = obj
            k = refname_parts[1]
            v = inspect.getattr_static(cls, k, None)
            found_super_cls = None
            for i, super_cls in enumerate(inspect.getmro(cls)[1:]):
                if k in dir(super_cls):
                    v2 = inspect.getattr_static(super_cls, k, None)
                    if v2 is not None and v == v2:
                        found_super_cls = super_cls
            if found_super_cls is not None:
                cls_path = found_super_cls.__module__ + "." + found_super_cls.__name__
                return cls_path + "." + ".".join(refname_parts[1:])
            return module.__name__ + "." + ".".join(refname_parts)
        if inspect.ismodule(obj):
            parent_module = obj
            refname_parts = refname_parts[1:]
        return resolve_refname(".".join(refname_parts), module=parent_module)

    refnames = []
    visited_modules = set()
    for k, v in module.__dict__.items():
        if v is not module:
            if inspect.ismodule(v) and v.__name__.startswith(module.__name__) and v.__name__ not in visited_modules:
                visited_modules.add(v.__name__)
                new_refname = resolve_refname(".".join(refname_parts), module=v)
                if new_refname is not None:
                    if isinstance(new_refname, str):
                        new_refname = [new_refname]
                    for r in new_refname:
                        if r not in refnames:
                            refnames.append(r)
    if len(refnames) > 1:
        return refnames
    if len(refnames) == 1:
        return refnames[0]
    return None


def get_refname(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
) -> tp.Optional[str]:
    """Parse and (optionally) resolve the reference name of an object."""
    if isinstance(obj, tuple):
        if len(obj) == 1:
            obj = obj[0]
        else:
            first_refname = parse_refname(obj[0])
            obj = first_refname + "." + ".".join(obj[1:])
    if isinstance(obj, str):
        refname = obj
    else:
        refname = parse_refname(obj)
    if resolve:
        return resolve_refname(refname, module=module)
    return refname


def get_imlucky_url(query: str) -> str:
    """Get the "I'm lucky" URL on DuckDuckGo for a query."""
    return "https://duckduckgo.com/?q=!ducky+" + urllib.request.pathname2url(query)


def imlucky(query: str, **kwargs) -> None:
    """Open the "I'm lucky" URL on DuckDuckGo for a query."""
    webbrowser.open(get_imlucky_url(query), **kwargs)


def get_api_ref(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
) -> str:
    """Get the API reference to an object."""

    def _raise():
        raise ValueError(
            "Couldn't find the reference name, or the object is external. "
            "If the object is internal, please decompose the object or provide a string instead."
        )

    refname = get_refname(obj, module=module, resolve=resolve)
    if refname is None:
        _raise()
    if isinstance(refname, list):
        raise ValueError("Multiple reference names found: {}".format(refname))
    module, qualname = get_refname_module_and_qualname(refname)
    if module.__name__.split(".")[0] == "vectorbtpro":
        api_url = "https://github.com/polakowo/vectorbt.pro/blob/pvt-links/api/"
        md_url = api_url + module.__name__ + ".md/"
        if qualname is None:
            return md_url + "#" + module.__name__.replace(".", "")
        return md_url + "#" + module.__name__.replace(".", "") + qualname.replace(".", "")
    if resolve:
        if qualname is None:
            search_query = module.__name__
        else:
            search_query = module.__name__ + "." + qualname
    else:
        search_query = refname
    return get_imlucky_url(search_query)


def open_api_ref(
    obj: tp.Any,
    module: tp.Union[None, str, ModuleType] = None,
    resolve: bool = True,
    **kwargs,
) -> None:
    """Open the API reference to an object."""
    webbrowser.open(get_api_ref(obj, module=module, resolve=resolve), **kwargs)
