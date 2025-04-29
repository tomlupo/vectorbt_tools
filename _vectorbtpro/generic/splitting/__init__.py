# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Modules for splitting."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vectorbtpro.generic.splitting.base import *
    from vectorbtpro.generic.splitting.decorators import *
    from vectorbtpro.generic.splitting.nb import *
    from vectorbtpro.generic.splitting.purged import *
    from vectorbtpro.generic.splitting.sklearn_ import *

__import_if_installed__ = dict()
__import_if_installed__["sklearn_"] = "sklearn"
