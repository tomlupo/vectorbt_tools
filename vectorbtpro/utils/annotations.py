# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for annotations."""

from collections import defaultdict

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "Annotatable",
    "VarArgs",
    "VarKwargs",
    "Union",
]

__pdoc__ = {}

try:
    from inspect import get_annotations as get_raw_annotations
except ImportError:
    import sys
    import types
    import functools

    def get_raw_annotations(obj, *, globals=None, locals=None, eval_str=False):
        """A backport of Python 3.10's inspect.get_annotations() function.

        See https://github.com/python/cpython/blob/main/Lib/inspect.py"""
        if isinstance(obj, type):
            # class
            obj_dict = getattr(obj, "__dict__", None)
            if obj_dict and hasattr(obj_dict, "get"):
                ann = obj_dict.get("__annotations__", None)
                if isinstance(ann, types.GetSetDescriptorType):
                    ann = None
            else:
                ann = None

            obj_globals = None
            module_name = getattr(obj, "__module__", None)
            if module_name:
                module = sys.modules.get(module_name, None)
                if module:
                    obj_globals = getattr(module, "__dict__", None)
            obj_locals = dict(vars(obj))
            unwrap = obj
        elif isinstance(obj, types.ModuleType):
            # module
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__dict__")
            obj_locals = None
            unwrap = None
        elif callable(obj):
            # this includes types.Function, types.BuiltinFunctionType,
            # types.BuiltinMethodType, functools.partial, functools.singledispatch,
            # "class funclike" from Lib/test/test_inspect... on and on it goes.
            ann = getattr(obj, "__annotations__", None)
            obj_globals = getattr(obj, "__globals__", None)
            obj_locals = None
            unwrap = obj
        else:
            raise TypeError(f"{obj!r} is not a module, class, or callable.")

        if ann is None:
            return {}

        if not isinstance(ann, dict):
            raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

        if not ann:
            return {}

        if not eval_str:
            return dict(ann)

        if unwrap is not None:
            while True:
                if hasattr(unwrap, "__wrapped__"):
                    unwrap = unwrap.__wrapped__
                    continue
                if isinstance(unwrap, functools.partial):
                    unwrap = unwrap.func
                    continue
                break
            if hasattr(unwrap, "__globals__"):
                obj_globals = unwrap.__globals__

        if globals is None:
            globals = obj_globals
        if locals is None:
            locals = obj_locals

        return_value = {
            key: value if not isinstance(value, str) else eval(value, globals, locals) for key, value in ann.items()
        }
        return return_value


def get_annotations(*args, **kwargs) -> tp.Annotations:
    """Get annotations."""
    annotations = get_raw_annotations(*args, **kwargs)
    new_annotations = {}
    for k, v in annotations.items():
        if isinstance(v, Union):
            v = v.resolve()
        new_annotations[k] = v
    return new_annotations


def flatten_annotations(
    annotations: tp.Annotations,
    only_var_args: bool = False,
    return_var_arg_maps: bool = False,
) -> tp.Union[tp.Annotations, tp.Tuple[tp.Annotations, tp.Dict[str, str], tp.Dict[str, str]]]:
    """Flatten annotations of variable arguments."""
    flat_annotations = {}
    var_args_map = {}
    var_kwargs_map = {}
    for k, v in annotations.items():
        if isinstance(v, VarArgs):
            for i, arg_v in enumerate(v.args):
                if isinstance(arg_v, Union):
                    arg_v = arg_v.resolve()
                new_k = f"{k}_{i}"
                if new_k in annotations:
                    raise ValueError(f"Unpacked key {new_k} already exists in annotations")
                flat_annotations[new_k] = arg_v
                var_args_map[new_k] = k
        elif isinstance(v, VarKwargs):
            for arg_k, arg_v in v.kwargs.items():
                if isinstance(arg_v, Union):
                    arg_v = arg_v.resolve()
                if arg_k in annotations:
                    raise ValueError(f"Unpacked key {arg_k} already exists in annotations")
                flat_annotations[arg_k] = arg_v
                var_kwargs_map[arg_k] = k
        elif not only_var_args:
            flat_annotations[k] = v
    if return_var_arg_maps:
        return flat_annotations, var_args_map, var_kwargs_map
    return flat_annotations


class MetaAnnotatable(type):
    """Metaclass that can be used in annotations."""

    def __or__(cls, other: tp.Annotation) -> tp.Annotation:
        return Union(cls, other).resolve()

    def __ror__(cls, other: tp.Annotation) -> tp.Annotation:
        return Union(cls, other).resolve()


class Annotatable(metaclass=MetaAnnotatable):
    """Class that can be used in annotations."""

    def __or__(self, other: tp.Annotation) -> tp.Annotation:
        return Union(self, other).resolve()

    def __ror__(self, other: tp.Annotation) -> tp.Annotation:
        return Union(self, other).resolve()


def has_annotatables(func: tp.Callable, target_cls: tp.Type[Annotatable] = Annotatable) -> bool:
    """Check if a function has subclasses or instances of `Annotatable` in its signature."""
    annotations = flatten_annotations(get_annotations(func))
    for k, v in annotations.items():
        if isinstance(v, type) and issubclass(v, target_cls):
            return True
        if not isinstance(v, type) and isinstance(v, target_cls):
            return True
    return False


@define
class VarArgs(Annotatable, DefineMixin):
    """Class representing annotations for variable positional arguments."""

    args: tp.Tuple[tp.Annotation, ...] = define.field()
    """Tuple with annotations."""

    def __init__(self, *args) -> None:
        DefineMixin.__init__(self, args=args)


@define
class VarKwargs(Annotatable, DefineMixin):
    """Class representing annotations for variable keyword arguments."""

    kwargs: tp.Dict[str, tp.Annotation] = define.field()
    """Dict with annotations."""

    def __init__(self, **kwargs) -> None:
        DefineMixin.__init__(self, kwargs=kwargs)


@define
class Union(Annotatable, DefineMixin):
    """Class representing a union of one to multiple annotations."""

    annotations: tp.Tuple[tp.Annotation, ...] = define.field()
    """Annotations."""

    resolved: bool = define.field(default=False)
    """Whether the instance is resolved."""

    def __init__(self, *annotations, resolved: bool = False) -> None:
        DefineMixin.__init__(self, annotations=annotations, resolved=resolved)

    def resolve(self) -> tp.Annotation:
        """Resolve the union."""
        if self.resolved:
            return self
        annotations = []
        for annotation in self.annotations:
            if isinstance(annotation, Union):
                annotation = annotation.resolve()
            if isinstance(annotation, Union):
                for annotation in annotation.annotations:
                    if annotation not in annotations:
                        annotations.append(annotation)
            else:
                if annotation not in annotations:
                    annotations.append(annotation)
        var_args_found = False
        var_kwargs_found = False
        for annotation in annotations:
            if isinstance(annotation, VarArgs):
                var_args_found = True
            if isinstance(annotation, VarKwargs):
                var_kwargs_found = True
        if var_args_found and var_kwargs_found:
            raise ValueError("Cannot make a union of VarArgs and VarKwargs")

        if var_args_found:
            if len(annotations) == 1:
                return annotations[0]
            max_n_args = 0
            for annotation in annotations:
                if isinstance(annotation, VarArgs):
                    if len(annotation.args) > max_n_args:
                        max_n_args = len(annotation.args)
            var_args_annotations = [[] for _ in range(max_n_args)]
            for annotation in annotations:
                if isinstance(annotation, VarArgs):
                    for i, v in enumerate(annotation.args):
                        var_args_annotations[i].append(v)
                else:
                    for i in range(len(var_args_annotations)):
                        var_args_annotations[i].append(annotation)
            var_args_unions = []
            for v in var_args_annotations:
                v_union = Union(*v).resolve()
                if isinstance(v_union, VarArgs):
                    raise ValueError("Found VarArgs inside VarArgs")
                if isinstance(v_union, VarKwargs):
                    raise ValueError("Found VarKwargs inside VarArgs")
                var_args_unions.append(v_union)
            return VarArgs(*var_args_unions)

        if var_kwargs_found:
            if len(annotations) == 1:
                return annotations[0]
            all_keys = set()
            for annotation in annotations:
                if isinstance(annotation, VarKwargs):
                    for k in annotation.kwargs.keys():
                        all_keys.add(k)
            var_kwargs_annotations = defaultdict(list)
            for annotation in annotations:
                if isinstance(annotation, VarKwargs):
                    for k, v in annotation.kwargs.items():
                        var_kwargs_annotations[k].append(v)
                else:
                    for k in all_keys:
                        var_kwargs_annotations[k].append(annotation)
            var_kwargs_unions = {}
            for k, v in var_kwargs_annotations.items():
                v_union = Union(*v).resolve()
                if isinstance(v_union, VarArgs):
                    raise ValueError("Found VarArgs inside VarKwargs")
                if isinstance(v_union, VarKwargs):
                    raise ValueError("Found VarKwargs inside VarKwargs")
                var_kwargs_unions[k] = v_union
            return VarKwargs(**var_kwargs_unions)

        return Union(*annotations, resolved=True)
