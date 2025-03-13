# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for merging."""

from functools import partial

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import get_annotations, Annotatable, Union
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.template import substitute_templates

__all__ = [
    "MergeFunc",
]

__pdoc__ = {}

MergeFuncT = tp.TypeVar("MergeFuncT", bound="MergeFunc")


@define
class MergeFunc(Evaluable, Annotatable, DefineMixin):
    """Class representing a merging function and its keyword arguments.

    Can be directly called to call the underlying (already resolved and with keyword
    arguments attached) merging function."""

    merge_func: tp.MergeFuncLike = define.field()
    """Merging function."""

    merge_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to the merging function."""

    context: tp.KwargsLike = define.field(default=None)
    """Context for substituting templates in `MergeFunc.merge_func` and `MergeFunc.merge_kwargs`."""

    sub_id_prefix: str = define.field(default="")
    """Prefix for the substitution id."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def __init__(self, *args, **kwargs) -> None:
        attr_names = [a.name for a in self.fields]
        if attr_names.index("merge_kwargs") < len(args):
            new_args = list(args)
            merge_kwargs = new_args[attr_names.index("merge_kwargs")]
            if merge_kwargs is None:
                merge_kwargs = {}
            else:
                merge_kwargs = dict(merge_kwargs)
            merge_kwargs.update({k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names})
            new_args[attr_names.index("merge_kwargs")] = merge_kwargs
            args = tuple(new_args)
        else:
            merge_kwargs = kwargs.pop("merge_kwargs", None)
            if merge_kwargs is None:
                merge_kwargs = {}
            else:
                merge_kwargs = dict(merge_kwargs)
            merge_kwargs.update({k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names})
            kwargs["merge_kwargs"] = merge_kwargs

        DefineMixin.__init__(self, *args, **kwargs)

    def resolve_merge_func(self) -> tp.Optional[tp.Callable]:
        """Get the merging function where keyword arguments are hard-coded."""
        from vectorbtpro.base.merging import resolve_merge_func

        merge_func = resolve_merge_func(self.merge_func)
        if merge_func is None:
            return None
        merge_kwargs = self.merge_kwargs
        if merge_kwargs is None:
            merge_kwargs = {}
        merge_func = substitute_templates(merge_func, self.context, eval_id=self.sub_id_prefix + "merge_func")
        merge_kwargs = substitute_templates(merge_kwargs, self.context, eval_id=self.sub_id_prefix + "merge_kwargs")
        return partial(merge_func, **merge_kwargs)

    def __call__(self, *objs, **kwargs) -> tp.Any:
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        merge_func = self.resolve_merge_func()
        if merge_func is None:
            return objs
        return merge_func(objs, **kwargs)


def parse_merge_func(func: tp.Callable, eval_id: tp.Optional[tp.Hashable] = None) -> tp.Optional[MergeFunc]:
    """Parser the merging function from the function's annotations."""
    annotations = get_annotations(func)
    merge_func = None
    for k, v in annotations.items():
        if k == "return":
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                if isinstance(annotation, str):
                    from vectorbtpro.base.merging import merge_func_config

                    if annotation in merge_func_config:
                        annotation = MergeFunc(annotation)
                if checks.is_complex_sequence(annotation):
                    for o in annotation:
                        if o is None or isinstance(o, str) or (isinstance(o, MergeFunc) and o.meets_eval_id(eval_id)):
                            if merge_func is None:
                                merge_func = []
                            elif not isinstance(merge_func, list):
                                raise ValueError(f"Two merging functions found in annotations: {merge_func} and {o}")
                            merge_func.append(o)
                elif isinstance(annotation, MergeFunc) and annotation.meets_eval_id(eval_id):
                    if merge_func is not None:
                        raise ValueError(f"Two merging functions found in annotations: {merge_func} and {annotation}")
                    merge_func = annotation
    return merge_func
