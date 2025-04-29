# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for profiling time and memory."""

import tracemalloc
from datetime import timedelta
from functools import wraps, partial
from timeit import default_timer, Timer as Timer_timeit

import humanize

from vectorbtpro import _typing as tp

__all__ = [
    "Timer",
    "with_timer",
    "timeit",
    "with_timeit",
    "MemTracer",
    "with_memtracer",
]

TimerT = tp.TypeVar("TimerT", bound="Timer")


class Timer:
    """Context manager to measure execution time using `timeit`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> with vbt.Timer() as timer:
        >>>     sleep(1)

        >>> print(timer.elapsed())
        1.01 seconds

        >>> timer.elapsed(readable=False)
        datetime.timedelta(seconds=1, microseconds=5110)
        ```
    """

    def __init__(self) -> None:
        self._start_time = default_timer()
        self._end_time = None

    @property
    def start_time(self) -> float:
        """Start time."""
        return self._start_time

    @property
    def end_time(self) -> float:
        """End time."""
        if self._end_time is None:
            return default_timer()
        return self._end_time

    def elapsed(self, readable: bool = True, **kwargs) -> tp.Union[str, timedelta]:
        """Get elapsed time.

        `**kwargs` are passed to `humanize.precisedelta`."""
        elapsed = self.end_time - self.start_time
        elapsed_delta = timedelta(seconds=elapsed)
        if readable:
            if "minimum_unit" not in kwargs:
                kwargs["minimum_unit"] = "seconds" if elapsed >= 1 else "milliseconds"
            return humanize.precisedelta(elapsed_delta, **kwargs)
        return elapsed_delta

    def __enter__(self: TimerT) -> TimerT:
        self._start_time = default_timer()
        return self

    def __exit__(self, *args) -> None:
        self._end_time = default_timer()


def with_timer(
    *args,
    timer_kwargs: tp.KwargsLike = None,
    elapsed_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Decorator to run a function with `Timer`."""

    if timer_kwargs is None:
        timer_kwargs = {}
    if elapsed_kwargs is None:
        elapsed_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = "{func_name} in {elapsed}"
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with Timer(**timer_kwargs) as timer:
                out = func(*args, **kwargs)
            elapsed = timer.elapsed(**elapsed_kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    elapsed=elapsed,
                ),
                **print_kwargs,
            )
            return out

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


def timeit(func: tp.Callable, readable: bool = True, **kwargs) -> tp.Union[str, timedelta]:
    """Run `timeit` on a function.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> def my_func():
        ...     sleep(1)

        >>> elapsed = vbt.timeit(my_func)
        >>> print(elapsed)
        1.01 seconds

        >>> vbt.timeit(my_func, readable=False)
        datetime.timedelta(seconds=1, microseconds=1870)
        ```
    """
    timer = Timer_timeit(stmt=func)
    number, time_taken = timer.autorange()
    elapsed = time_taken / number
    elapsed_delta = timedelta(seconds=elapsed)
    if readable:
        if "minimum_unit" not in kwargs:
            kwargs["minimum_unit"] = "seconds" if elapsed >= 1 else "milliseconds"
        return humanize.precisedelta(elapsed_delta, **kwargs)
    return elapsed_delta


def with_timeit(
    *args,
    timeit_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Decorator to run a function with `timeit`."""

    if timeit_kwargs is None:
        timeit_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = "{func_name} in {elapsed} on average"
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            elapsed = timeit(partial(func, *args, **kwargs), **timeit_kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    elapsed=elapsed,
                ),
                **print_kwargs,
            )
            return func(*args, **kwargs)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


MemTracerT = tp.TypeVar("MemTracerT", bound="MemTracer")


class MemTracer:
    """Context manager to trace peak and final memory usage using `tracemalloc`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> with vbt.MemTracer() as tracer:
        >>>     np.random.uniform(size=1000000)

        >>> print(tracer.peak_usage())
        8.0 MB

        >>> tracer.peak_usage(readable=False)
        8005360
        ```
    """

    def __init__(self) -> None:
        self._final_usage = None
        self._peak_usage = None

    def final_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get final memory usage.

        `**kwargs` are passed to `humanize.naturalsize`."""
        if self._final_usage is None:
            final_usage = tracemalloc.get_traced_memory()[0]
        else:
            final_usage = self._final_usage
        if readable:
            return humanize.naturalsize(final_usage, **kwargs)
        return final_usage

    def peak_usage(self, readable: bool = True, **kwargs) -> tp.Union[str, int]:
        """Get peak memory usage.

        `**kwargs` are passed to `humanize.naturalsize`."""
        if self._peak_usage is None:
            peak_usage = tracemalloc.get_traced_memory()[1]
        else:
            peak_usage = self._peak_usage
        if readable:
            return humanize.naturalsize(peak_usage, **kwargs)
        return peak_usage

    def __enter__(self: MemTracerT) -> MemTracerT:
        tracemalloc.start()
        tracemalloc.clear_traces()
        return self

    def __exit__(self, *args) -> None:
        self._final_usage, self._peak_usage = tracemalloc.get_traced_memory()
        tracemalloc.stop()


def with_memtracer(
    *args,
    memtracer_kwargs: tp.KwargsLike = None,
    usage_kwargs: tp.KwargsLike = None,
    print_func: tp.Optional[tp.Callable] = None,
    print_format: tp.Optional[str] = None,
    print_kwargs: tp.KwargsLike = None,
) -> tp.Callable:
    """Decorator to run a function with `MemTracer`."""

    if memtracer_kwargs is None:
        memtracer_kwargs = {}
    if usage_kwargs is None:
        usage_kwargs = {}
    if print_func is None:
        print_func = print
    if print_format is None:
        print_format = "{func_name} with peak usage of {peak_usage} and final usage of {final_usage}"
    if print_kwargs is None:
        print_kwargs = {}

    def decorator(func: tp.Callable) -> tp.Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            with MemTracer(**memtracer_kwargs) as memtracer:
                out = func(*args, **kwargs)
            print_func(
                print_format.format(
                    func_name=func.__qualname__,
                    peak_usage=memtracer.peak_usage(**usage_kwargs),
                    final_usage=memtracer.final_usage(**usage_kwargs),
                ),
                **print_kwargs,
            )
            return out

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
