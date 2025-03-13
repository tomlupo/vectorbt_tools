# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for evaluation and compilation."""

import ast
import inspect

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks

__all__ = []


def multiline_eval(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate several lines of input, returning the result of the last line."""
    if context is None:
        context = {}
    tree = ast.parse(inspect.cleandoc(expr))
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
    exec(compile(exec_expr, "file", "exec"), context)
    return eval(compile(eval_expr, "file", "eval"), context)


class Evaluable:
    """Abstract class for instances that can be evaluated."""

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """Return whether the evaluation id of the instance meets the global evaluation id."""
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True
