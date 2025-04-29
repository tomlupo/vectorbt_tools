# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for selecting."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils.attr_ import DefineMixin, define

__all__ = [
    "PosSel",
    "LabelSel",
]


@define
class PosSel(DefineMixin):
    """Class that represents a selection by position."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more positions."""


@define
class LabelSel(DefineMixin):
    """Class that represents a selection by label."""

    value: tp.MaybeIterable[tp.Hashable] = define.field()
    """Selection of one or more labels."""
