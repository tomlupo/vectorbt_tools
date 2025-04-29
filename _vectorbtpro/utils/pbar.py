# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for progress bars."""

import warnings
from functools import wraps
from numbers import Number
from time import time as utc_time

from tqdm.std import tqdm

from vectorbtpro import _typing as tp
from vectorbtpro.registries.pbar_registry import PBarRegistry, pbar_reg
from vectorbtpro.utils.attr_ import MISSING
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ProgressBar",
    "ProgressHidden",
    "with_progress_hidden",
    "ProgressShown",
    "with_progress_shown",
]

ProgressBarT = tp.TypeVar("ProgressBarT", bound="ProgressBar")


class ProgressBar:
    """Context manager to manage a progress bar.

    Supported types:

    * 'tqdm_auto'
    * 'tqdm_notebook'
    * 'tqdm_gui'
    * 'tqdm'

    For defaults, see `vectorbtpro._settings.pbar`."""

    def __init__(
        self,
        iterable: tp.Optional[tp.Iterable] = None,
        bar_id: tp.Optional[tp.Hashable] = None,
        bar_type: tp.Optional[str] = None,
        force_open_bar: tp.Optional[bool] = None,
        reuse: tp.Optional[bool] = None,
        disable: tp.Optional[bool] = None,
        show_progress: tp.Optional[bool] = None,
        show_progress_desc: tp.Optional[bool] = None,
        desc_kwargs: tp.KwargsLike = None,
        registry: tp.Optional[PBarRegistry] = pbar_reg,
        silence_warnings: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        if bar_type is None:
            bar_type = pbar_cfg["type"]
        if bar_type.lower() == "tqdm_auto":
            from tqdm.auto import tqdm as bar_type
        elif bar_type.lower() == "tqdm_notebook":
            from tqdm.notebook import tqdm as bar_type
        elif bar_type.lower() == "tqdm_gui":
            from tqdm.gui import tqdm as bar_type
        elif bar_type.lower() == "tqdm":
            from tqdm import tqdm as bar_type
        else:
            raise ValueError(f"Invalid option bar_type='{bar_type}'")
        if force_open_bar is None:
            force_open_bar = pbar_cfg["force_open_bar"]
        if reuse is None:
            reuse = pbar_cfg["reuse"]
        if disable is not None:
            if show_progress is not None:
                if disable or not show_progress:
                    show_progress = False
                else:
                    show_progress = True
            else:
                show_progress = not disable
        else:
            if pbar_cfg["disable"]:
                show_progress = False
            if show_progress is None:
                show_progress = True
        if pbar_cfg["disable_machinery"]:
            show_progress = False
        kwargs = merge_dicts(pbar_cfg["kwargs"], kwargs)
        if pbar_cfg["disable_desc"] or pbar_cfg["disable_machinery"]:
            show_progress_desc = False
        if show_progress_desc is None:
            show_progress_desc = True
        desc_kwargs = merge_dicts(pbar_cfg["desc_kwargs"], desc_kwargs)
        if pbar_cfg["disable_registry"] or pbar_cfg["disable_machinery"]:
            registry = None
        if silence_warnings is None:
            silence_warnings = pbar_cfg["silence_warnings"]

        self._iterable = iterable
        self._bar_id = bar_id
        self._bar_type = bar_type
        self._force_open_bar = force_open_bar
        self._reuse = reuse
        self._show_progress = show_progress
        self._kwargs = kwargs
        self._show_progress_desc = show_progress_desc
        self._desc_kwargs = desc_kwargs
        self._registry = registry
        self._silence_warnings = silence_warnings

        self._bar = None
        self._open_time = None
        self._update_time = None
        self._refresh_time = None
        self._close_time = None

    @property
    def bar_id(self) -> tp.Optional[tp.Hashable]:
        """Bar id."""
        return self._bar_id

    @property
    def bar_type(self) -> tp.Type[tqdm]:
        """Bar type."""
        return self._bar_type

    @property
    def force_open_bar(self) -> bool:
        """Whether to force-open a bar even if progress is not shown."""
        return self._force_open_bar

    @property
    def reuse(self) -> bool:
        """Whether the bar can be reused."""
        return self._reuse

    @property
    def show_progress(self) -> bool:
        """Whether to show the bar."""
        return self._show_progress

    @property
    def iterable(self) -> tp.Optional[tp.Iterable]:
        """Iterable."""
        return self._iterable

    @property
    def kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to initialize the bar."""
        return self._kwargs

    @property
    def show_progress_desc(self) -> bool:
        """Whether show the bar description."""
        return self._show_progress_desc

    @property
    def desc_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ProgressBar.set_description`."""
        return self._desc_kwargs

    @property
    def registry(self) -> tp.Optional[PBarRegistry]:
        """Registry of type `vectorbtpro.registries.pbar_registry.PBarRegistry`.

        If None, registry is disabled."""
        return self._registry

    @property
    def silence_warnings(self) -> bool:
        """Whether to silence warnings."""
        return self._silence_warnings

    @property
    def bar(self) -> tp.Optional[tqdm]:
        """Bar."""
        return self._bar

    @property
    def open_time(self) -> tp.Optional[int]:
        """Time the bar was opened."""
        return self._open_time

    @property
    def update_time(self) -> tp.Optional[int]:
        """Time the bar was updated."""
        return self._update_time

    @property
    def refresh_time(self) -> tp.Optional[int]:
        """Time the bar was refreshed."""
        return self._refresh_time

    @property
    def close_time(self) -> tp.Optional[int]:
        """Time the bar was closed."""
        return self._close_time

    def set_bar(self, bar: tp.Optional[tqdm] = None) -> None:
        """Set the bar."""
        if bar is None:
            bar = self.bar_type(self.iterable, disable=not self.show_progress, **self.kwargs)
        self._bar = bar

    def remove_bar(self) -> None:
        """Remove the bar."""
        self._bar = None

    def reset(self) -> None:
        """Reset the bar."""
        for pbar in self.registry.get_parent_instances(self):
            if not pbar.disabled and not pbar.displayed:
                pbar.refresh()  # refresh parents first
        self.bar.iterable = self.iterable
        self.bar.delay = 0
        self.bar.reset(total=self.kwargs.get("total", None))  # tqdm-forced refresh

    def open(self, reuse: tp.Optional[bool] = None) -> None:
        """Open the bar."""
        if self.bar is None:
            self._open_time = utc_time()
            if self.registry is None:
                self.set_bar()
            else:
                if reuse is None:
                    reuse = self.reuse
                if reuse and self.registry.has_conflict(self):
                    new_bar_id = self.registry.generate_bar_id()
                    if isinstance(self.bar_id, str):
                        bar_id = f"'{self.bar_id}'"
                    else:
                        bar_id = self.bar_id
                    if not self.silence_warnings:
                        warnings.warn(
                            f"Two active progress bars share the same bar id {bar_id}. "
                            f"Setting bar id of the new progress bar to '{new_bar_id}'."
                        )
                    self._bar_id = new_bar_id
                pending_bar = self.registry.get_pending_instance(self)
                if reuse and pending_bar is not None:
                    self.set_bar(bar=pending_bar.bar)
                    pending_bar.remove_bar()
                    self.registry.deregister_instance(pending_bar)
                    self.reset()
                else:
                    if self.bar_id is None:
                        self._bar_id = self.registry.generate_bar_id()
                    self.set_bar()
                self.registry.register_instance(self)

    def close(self, reuse: tp.Optional[bool] = None, close_children: bool = True) -> None:
        """Close the bar."""
        if self.bar is not None:
            self._close_time = utc_time()
            if self.registry is None:
                if not self.disabled:
                    self.bar.close()
                self.remove_bar()
            else:
                if reuse is None:
                    if not self.displayed:
                        reuse = False
                    else:
                        reuse = self.reuse
                parent_pbar = self.registry.get_parent_instance(self)
                if not reuse or parent_pbar is None:
                    if close_children:
                        for pbar in self.registry.get_child_instances(self):
                            pbar.close(reuse=False, close_children=False)
                    if not self.disabled:
                        self.bar.close()
                    self.remove_bar()
                    self.registry.deregister_instance(self)

    @property
    def active(self) -> bool:
        """Whether the bar is active."""
        return self.bar is not None and self.close_time is None

    @property
    def pending(self) -> bool:
        """Whether the bar is pending."""
        return self.bar is not None and self.close_time is not None

    @property
    def disabled(self) -> bool:
        """Whether the bar is disabled."""
        if not self.show_progress:
            return True
        if self.bar is None:
            if self.bar_id is not None:
                if isinstance(self.bar_id, str):
                    bar_id = f"'{self.bar_id}'"
                else:
                    bar_id = self.bar_id
                raise ValueError(f'Progress bar with bar id {bar_id} must be opened first. Use "with" statement.')
            else:
                raise ValueError('Progress bar must be opened first. Use "with" statement.')
        if self.bar.disable:
            return True
        return False

    @property
    def displayed(self) -> bool:
        """Whether the bar is displayed."""
        if self.disabled:
            return False
        if self.refresh_time is not None:
            return True
        if self.bar.delay == 0:
            return True
        delay_end_t = self.bar.start_t + self.bar.delay
        return self.bar.last_print_t >= delay_end_t

    @property
    def should_display(self) -> bool:
        """Whether the bar should be displayed."""
        if self.disabled:
            return False
        if self.bar.delay == 0:
            return True
        delay_end_t = self.bar.start_t + self.bar.delay
        if utc_time() >= delay_end_t:
            return True
        return False

    def refresh(self) -> None:
        """Refresh the bar."""
        self.bar.refresh()
        self._refresh_time = utc_time()

    def before_update(self) -> None:
        """Do something before an update."""
        if self.disabled:
            return
        if self.registry is not None:
            displayed = self.displayed
            should_display = self.should_display
            for pbar in self.registry.get_parent_instances(self):
                if not pbar.disabled and not pbar.displayed:
                    if (displayed or should_display) or pbar.should_display:
                        pbar.refresh()

    def update(self, n: int = 1) -> None:
        """Update with one or more iterations."""
        if self.disabled:
            return
        self.before_update()
        self.bar.update(n=n)
        self._update_time = utc_time()
        self.after_update()

    def update_to(self, n: int) -> None:
        """Update to a specific number."""
        if self.disabled:
            return
        self.update(n=n - self.bar.n)

    def after_update(self) -> None:
        """Do something after an update."""
        if self.disabled:
            return

    def prepare_desc(self, desc: tp.Union[None, str, dict]) -> str:
        """Prepare description."""
        if desc is None or desc is MISSING:
            return ""
        if isinstance(desc, dict):
            new_desc = []
            for k, v in desc.items():
                if v is MISSING:
                    if k not in (None, MISSING):
                        if not isinstance(k, str):
                            k = str(k)
                        k = k.strip()
                        new_desc.append(k)
                else:
                    if isinstance(v, Number):
                        v = self.bar.format_num(v)
                    if not isinstance(v, str):
                        v = str(v)
                    v = v.strip()
                    if k not in (None, MISSING):
                        if not isinstance(k, str):
                            k = str(k)
                        k = k.strip()
                        new_desc.append(k + "=" + v)
                    else:
                        new_desc.append(v)
            return ", ".join(new_desc)
        return str(desc)

    def set_prefix(self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None) -> None:
        """Set prefix.

        Prepares it with `ProgressBar.prepare_desc`."""
        if self.disabled or not self.show_progress_desc:
            return
        desc = self.prepare_desc(desc)
        self.bar.set_description_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_prefix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set prefix without preparation."""
        if self.disabled or not self.show_progress_desc:
            return
        self.bar.set_description_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_postfix(self, desc: tp.Union[None, str, dict], refresh: tp.Optional[bool] = None) -> None:
        """Set postfix.

        Prepares it with `ProgressBar.prepare_desc`."""
        if self.disabled or not self.show_progress_desc:
            return
        desc = self.prepare_desc(desc)
        self.bar.set_postfix_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_postfix_str(self, desc: str, refresh: tp.Optional[bool] = None) -> None:
        """Set postfix without preparation."""
        if self.disabled or not self.show_progress_desc:
            return
        self.bar.set_postfix_str(desc, refresh=False)
        if refresh is None:
            refresh = self.desc_kwargs.get("refresh", True)
        if refresh:
            self.refresh()

    def set_description(
        self,
        desc: tp.Union[None, str, dict],
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set description.

        Uses the method `ProgressBar.set_prefix` if `as_postfix=True` in `ProgressBar.desc_kwargs`.
        Otherwise, uses the method `ProgressBar.set_postfix`.

        Uses `ProgressBar.desc_kwargs` as keyword arguments."""
        if self.disabled or not self.show_progress_desc:
            return
        if as_postfix is None:
            as_postfix = self.desc_kwargs.get("as_postfix", True)
        if as_postfix:
            self.set_postfix(desc, refresh=refresh)
        else:
            self.set_prefix(desc, refresh=refresh)

    def set_description_str(
        self,
        desc: str,
        as_postfix: tp.Optional[bool] = None,
        refresh: tp.Optional[bool] = None,
    ) -> None:
        """Set description without preparation.

        Uses the method `ProgressBar.set_prefix_str` if `as_postfix=True` in `ProgressBar.desc_kwargs`.
        Otherwise, uses the method `ProgressBar.set_postfix_str`.

        Uses `ProgressBar.desc_kwargs` as keyword arguments."""
        if self.disabled or not self.show_progress_desc:
            return
        if as_postfix is None:
            as_postfix = self.desc_kwargs.get("as_postfix", True)
        if as_postfix:
            self.set_postfix_str(desc, refresh=refresh)
        else:
            self.set_prefix_str(desc, refresh=refresh)

    def enter(self: ProgressBarT, **kwargs) -> ProgressBarT:
        """Enter the bar."""
        if self.show_progress or self.force_open_bar:
            self.open(**kwargs)
        return self

    def __enter__(self: ProgressBarT) -> ProgressBarT:
        return self.enter()

    def exit(self, **kwargs) -> None:
        """Exit the bar."""
        self.close(**kwargs)

    def __exit__(self, *args) -> None:
        self.exit()

    def __len__(self) -> int:
        return self.bar.__len__()

    def __contains__(self, item: tp.Any) -> bool:
        return self.bar.__contains__(item)

    def iter(self) -> tp.Iterator:
        """Get iterator over `ProgressBar.iterable`."""
        for obj in self.iterable:
            yield obj

    def __iter__(self) -> tp.Iterator:
        if self.bar is None:
            if self.show_progress or self.force_open_bar:
                self.open()
        if self.disabled:
            return self.iter()
        for i, obj in enumerate(self.bar.__iter__()):
            if i > 0:
                self._update_time = utc_time()
                self.after_update()
            yield obj
            self.before_update()
        self._update_time = utc_time()
        self.after_update()
        self.close()

    def __del__(self) -> None:
        self.close()


ProgressHiddenT = tp.TypeVar("ProgressHiddenT", bound="ProgressHidden")


class ProgressHidden:
    """Context manager to hide progress."""

    def __init__(self, disable_registry: bool = True, disable_machinery: bool = True) -> None:
        self._disable_registry = disable_registry
        self._disable_machinery = disable_machinery
        self._init_settings = None

    @property
    def disable_registry(self) -> bool:
        """Whether to disable registry."""
        return self._disable_registry

    @property
    def disable_machinery(self) -> bool:
        """Whether to disable machinery."""
        return self._disable_machinery

    @property
    def init_settings(self) -> tp.Kwargs:
        """Initial settings."""
        return self._init_settings

    def __enter__(self: ProgressHiddenT) -> ProgressHiddenT:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_settings = dict(
            disable=pbar_cfg["disable"],
            disable_registry=pbar_cfg["disable_registry"],
            disable_machinery=pbar_cfg["disable_machinery"],
        )

        pbar_cfg["disable"] = True
        pbar_cfg["disable_registry"] = self.disable_registry
        pbar_cfg["disable_machinery"] = self.disable_machinery

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_settings["disable"]
        pbar_cfg["disable_registry"] = self.init_settings["disable_registry"]
        pbar_cfg["disable_machinery"] = self.init_settings["disable_machinery"]


def with_progress_hidden(*args) -> tp.Callable:
    """Decorator to run a function with `ProgressHidden`."""

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressHidden():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


ProgressShownT = tp.TypeVar("ProgressShownT", bound="ProgressShown")


class ProgressShown:
    """Context manager to show progress."""

    def __init__(self, enable_registry: bool = True, enable_machinery: bool = True) -> None:
        self._enable_registry = enable_registry
        self._enable_machinery = enable_machinery
        self._init_settings = None

    @property
    def enable_registry(self) -> bool:
        """Whether to enable registry."""
        return self._enable_registry

    @property
    def enable_machinery(self) -> bool:
        """Whether to enable machinery."""
        return self._enable_machinery

    @property
    def init_settings(self) -> tp.Kwargs:
        """Initial settings."""
        return self._init_settings

    def __enter__(self: ProgressShownT) -> ProgressShownT:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        self._init_settings = dict(
            disable=pbar_cfg["disable"],
            disable_registry=pbar_cfg["disable_registry"],
            disable_machinery=pbar_cfg["disable_machinery"],
        )

        pbar_cfg["disable"] = False
        pbar_cfg["disable_registry"] = not self.enable_registry
        pbar_cfg["disable_machinery"] = not self.enable_machinery

        return self

    def __exit__(self, *args) -> None:
        from vectorbtpro._settings import settings

        pbar_cfg = settings["pbar"]

        pbar_cfg["disable"] = self.init_settings["disable"]
        pbar_cfg["disable_registry"] = self.init_settings["disable_registry"]
        pbar_cfg["disable_machinery"] = self.init_settings["disable_machinery"]


def with_progress_shown(*args) -> tp.Callable:
    """Decorator to run a function with `ProgressShown`."""

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with ProgressShown():
                return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
