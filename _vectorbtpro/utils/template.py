# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with templates."""

import importlib
from string import Template

import vectorbtpro as vbt
from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.eval_ import multiline_eval, Evaluable
from vectorbtpro.utils.module_ import package_shortcut_config
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.search import any_in_obj, find_and_replace_in_obj

__all__ = [
    "CustomTemplate",
    "Sub",
    "Rep",
    "RepEval",
    "RepFunc",
    "substitute_templates",
]


@define
class CustomTemplate(Evaluable, DefineMixin):
    """Class for substituting templates."""

    template: tp.Any = define.field()
    """Template to be processed."""

    context: tp.KwargsLike = define.field(default=None)
    """Context mapping."""

    strict: tp.Optional[bool] = define.field(default=None)
    """Whether to raise an error if processing template fails.

    If not None, overrides `strict` passed by `substitute_templates`."""

    context_merge_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to `vectorbtpro.utils.config.merge_dicts`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def resolve_context(
        self,
        context: tp.KwargsLike = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Kwargs:
        """Resolve `CustomTemplate.context`.

        Merges `context` in `vectorbtpro._settings.template`, `CustomTemplate.context`, and `context`.
        Automatically appends `eval_id`, `np` (NumPy), `pd` (Pandas), and `vbt` (vectorbtpro)."""
        from vectorbtpro._settings import settings

        template_cfg = settings["template"]

        context_merge_kwargs = self.context_merge_kwargs
        if context_merge_kwargs is None:
            context_merge_kwargs = {}
        new_context = merge_dicts(
            template_cfg["context"],
            self.context,
            context,
            **context_merge_kwargs,
        )
        if "context" not in new_context:
            new_context["context"] = dict(new_context)
        if "eval_id" not in new_context:
            new_context["eval_id"] = eval_id
        for k, v in package_shortcut_config.items():
            if k not in new_context:
                try:
                    new_context[k] = importlib.import_module(v)
                except ImportError:
                    pass
        return new_context

    def resolve_strict(self, strict: tp.Optional[bool] = None) -> bool:
        """Resolve `CustomTemplate.strict`.

        If `strict` is None, uses `strict` in `vectorbtpro._settings.template`."""
        if strict is None:
            strict = self.strict
        if strict is None:
            from vectorbtpro._settings import settings

            template_cfg = settings["template"]

            strict = template_cfg["strict"]
        return strict

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Abstract method to substitute the template `CustomTemplate.template` using
        the context from merging `CustomTemplate.context` and `context`."""
        raise NotImplementedError


class Sub(CustomTemplate):
    """Template string to substitute parts with the respective values from `context`.

    Always returns a string."""

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Substitute parts of `Sub.template` as a regular template."""
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return Template(self.template).substitute(context)
        except KeyError as e:
            if strict:
                raise e
        return self


class Rep(CustomTemplate):
    """Template string to be replaced with the respective value from `context`."""

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Replace `Rep.template` as a key."""
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return context[self.template]
        except KeyError as e:
            if strict:
                raise e
        return self


class RepEval(CustomTemplate):
    """Template expression to be evaluated using `vectorbtpro.utils.eval_.multiline_eval`
    with `context` used as locals."""

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Any:
        """Evaluate `RepEval.template` as an expression."""
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        try:
            return multiline_eval(self.template, context)
        except NameError as e:
            if strict:
                raise e
        return self


class RepFunc(CustomTemplate):
    """Template function to be called with argument names from `context`."""

    def substitute(
        self,
        context: tp.KwargsLike = None,
        strict: tp.Optional[bool] = None,
        eval_id: int = 0,
    ) -> tp.Any:
        """Call `RepFunc.template` as a function."""
        if not self.meets_eval_id(eval_id):
            return self
        context = self.resolve_context(context=context, eval_id=eval_id)
        strict = self.resolve_strict(strict=strict)

        func_arg_names = get_func_arg_names(self.template)
        func_kwargs = dict()
        for k, v in context.items():
            if k in func_arg_names:
                func_kwargs[k] = v

        try:
            return self.template(**func_kwargs)
        except TypeError as e:
            if strict:
                raise e
        return self


def has_templates(obj: tp.Any, **kwargs) -> tp.Any:
    """Check if the object has any templates.

    Uses `vectorbtpro.utils.search.any_in_obj`.

    Default can be overridden with `search_kwargs` under `vectorbtpro._settings.template`."""
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    search_kwargs = merge_dicts(template_cfg["search_kwargs"], kwargs)

    def _match_func(k, v):
        return isinstance(v, (CustomTemplate, Template))

    return any_in_obj(obj, _match_func, **search_kwargs)


def substitute_templates(
    obj: tp.Any,
    context: tp.KwargsLike = None,
    strict: tp.Optional[bool] = None,
    eval_id: tp.Optional[tp.Hashable] = None,
    **kwargs,
) -> tp.Any:
    """Traverses the object recursively and, if any template found, substitutes it using a context.

    Uses `vectorbtpro.utils.search.find_and_replace_in_obj`.

    If `strict` is True, raises an error if processing template fails. Otherwise, returns the original template.

    Default can be overridden with `search_kwargs` under `vectorbtpro._settings.template`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}))
        100
        >>> vbt.substitute_templates(vbt.Sub('$key', {'key': 100}), {'key': 200})
        200
        >>> vbt.substitute_templates(vbt.Sub('$key$key'), {'key': 100})
        100100
        >>> vbt.substitute_templates(vbt.Rep('key'), {'key': 100})
        100
        >>> vbt.substitute_templates([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100}, incl_types=list)
        [100, '100100']
        >>> vbt.substitute_templates(vbt.RepFunc(lambda key: key == 100), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100'), {'key': 100})
        True
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=True))
        NameError: name 'key' is not defined
        >>> vbt.substitute_templates(vbt.RepEval('key == 100', strict=False))
        <vectorbtpro.utils.template.RepEval at 0x7fe3ad2ab668>
        ```
    """
    from vectorbtpro._settings import settings

    template_cfg = settings["template"]

    search_kwargs = merge_dicts(template_cfg["search_kwargs"], kwargs)

    def _match_func(k, v):
        return isinstance(v, (CustomTemplate, Template))

    def _replace_func(k, v):
        if isinstance(v, CustomTemplate):
            return v.substitute(context=context, strict=strict, eval_id=eval_id)
        if isinstance(v, Template):
            return v.substitute(context=context)

    return find_and_replace_in_obj(obj, _match_func, _replace_func, **search_kwargs)
