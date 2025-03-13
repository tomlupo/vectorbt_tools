# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Engines for executing functions."""

import concurrent.futures
import enum
import time
import warnings
from functools import partial, wraps
from pathlib import Path
import inspect

import pandas as pd
from numba.core.registry import CPUDispatcher

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.parsing import (
    annotate_args,
    flat_ann_args_to_args,
    match_ann_arg,
    match_and_set_flat_ann_arg,
)
from vectorbtpro.utils.attr_ import define, DefineMixin, MISSING
from vectorbtpro.utils.config import merge_dicts, Configured, FrozenConfig
from vectorbtpro.utils.merging import MergeFunc
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.path_ import remove_dir, file_exists
from vectorbtpro.utils.pbar import ProgressBar, ProgressHidden
from vectorbtpro.utils.pickling import load, save
from vectorbtpro.utils.template import CustomTemplate, substitute_templates

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from ray.remote_function import RemoteFunction as RemoteFunctionT
    from ray import ObjectRef as ObjectRefT
except ImportError:
    RemoteFunctionT = tp.Any
    ObjectRefT = tp.Any

__all__ = [
    "Task",
    "NoResult",
    "NoResultsException",
    "ExecutionEngine",
    "SerialEngine",
    "ThreadPoolEngine",
    "ProcessPoolEngine",
    "PathosEngine",
    "DaskEngine",
    "RayEngine",
    "execute",
    "iterated",
]


@define
class Task(DefineMixin):
    """Class that represents an executable task."""

    func: tp.Callable = define.field()
    """Function."""

    args: tp.Args = define.field(factory=tuple)
    """Positional arguments."""

    kwargs: tp.Kwargs = define.field(factory=dict)
    """Keyword arguments."""

    def __init__(self, func: tp.Callable, *args, **kwargs) -> None:
        DefineMixin.__init__(self, func=func, args=args, kwargs=kwargs)

    def __iter__(self) -> tp.Iterator:
        return iter((self.func, self.args, self.kwargs))

    def __getitem__(self, item: int) -> tp.Any:
        return tuple(self)[item]

    def execute(self) -> tp.Any:
        """Execute the task."""
        return self.func(*self.args, **self.kwargs)


class _NoResult(enum.Enum):
    """Sentinel that represents no result."""

    NoResult = enum.auto()

    def __repr__(self):
        return "NoResult"

    def __bool__(self):
        return False


NoResult = _NoResult.NoResult
"""Sentinel that represents no result."""


class NoResultsException(Exception):
    """Gets raised when there are no results."""

    pass


def filter_out_no_results(
    objs: tp.Iterable[tp.Any],
    keys: tp.Optional[tp.Index] = MISSING,
    raise_error: bool = True,
) -> tp.Union[tp.List, tp.Tuple[tp.List, tp.Optional[tp.Index]]]:
    """Filter objects and keys by removing `NoResult` objects."""
    skip_indices = set()
    for i, obj in enumerate(objs):
        if obj is NoResult:
            skip_indices.add(i)
    if len(skip_indices) > 0:
        new_objs = []
        keep_indices = []
        for i, obj in enumerate(objs):
            if i not in skip_indices:
                new_objs.append(obj)
                keep_indices.append(i)
        objs = new_objs
        if keys is not None and keys is not MISSING:
            if isinstance(keys, pd.Index):
                keys = keys[keep_indices]
            else:
                keys = [keys[i] for i in keep_indices]
        if len(objs) == 0 and raise_error:
            raise NoResultsException
    if keys is not MISSING:
        return objs, keys
    return objs


class ExecutionEngine(Configured):
    """Abstract class for executing functions."""

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        """Run an iterable of tuples out of a function, arguments, and keyword arguments.

        Provide `size` in case `tasks` is a generator and the underlying engine needs it."""
        raise NotImplementedError


class SerialEngine(ExecutionEngine):
    """Class for executing functions sequentially.

    For defaults, see `engines.serial` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.serial"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "show_progress",
        "pbar_kwargs",
        "clear_cache",
        "collect_garbage",
        "delay",
    }

    def __init__(
        self,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        clear_cache: tp.Union[None, bool, int] = None,
        collect_garbage: tp.Union[None, bool, int] = None,
        delay: tp.Optional[float] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            clear_cache=clear_cache,
            collect_garbage=collect_garbage,
            delay=delay,
            **kwargs,
        )

        self._show_progress = self.resolve_setting(show_progress, "show_progress")
        self._pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        self._clear_cache = self.resolve_setting(clear_cache, "clear_cache")
        self._collect_garbage = self.resolve_setting(collect_garbage, "collect_garbage")
        self._delay = self.resolve_setting(delay, "delay")

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    @property
    def clear_cache(self) -> tp.Union[bool, int]:
        """Whether to clear vectorbt's cache after each iteration.

        If integer, do it once a number of tasks."""
        return self._clear_cache

    @property
    def collect_garbage(self) -> tp.Union[bool, int]:
        """Whether to clear garbage after each iteration.

        If integer, do it once a number of tasks."""
        return self._collect_garbage

    @property
    def delay(self) -> tp.Optional[float]:
        """Number of seconds to sleep after each call."""
        return self._delay

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.registries.ca_registry import clear_cache, collect_garbage
        from vectorbtpro.base.indexes import to_any_index

        results = []
        if size is None and hasattr(tasks, "__len__"):
            size = len(tasks)
        elif keys is not None and hasattr(keys, "__len__"):
            size = len(keys)
        if keys is not None:
            keys = to_any_index(keys)
        pbar_kwargs = dict(self.pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = tuple(keys.names)
                else:
                    pbar_kwargs["bar_id"] = keys.name
        pbar = ProgressBar(total=size, show_progress=self.show_progress, **pbar_kwargs)

        with pbar:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar.set_description(dict(zip(keys.names, keys[0])))
                else:
                    pbar.set_description(dict(zip(keys.names, [keys[0]])))

            for i, (func, args, kwargs) in enumerate(tasks):
                results.append(func(*args, **kwargs))
                if isinstance(self.clear_cache, bool):
                    if self.clear_cache:
                        clear_cache()
                elif i > 0 and (i + 1) % self.clear_cache == 0:
                    clear_cache()
                if isinstance(self.collect_garbage, bool):
                    if self.collect_garbage:
                        collect_garbage()
                elif i > 0 and (i + 1) % self.collect_garbage == 0:
                    collect_garbage()
                if self.delay is not None:
                    time.sleep(self.delay)

                if keys is not None and i + 1 < len(keys):
                    if isinstance(keys, pd.MultiIndex):
                        pbar.set_description(dict(zip(keys.names, keys[i + 1])))
                    else:
                        pbar.set_description(dict(zip(keys.names, [keys[i + 1]])))
                pbar.update()

        return results


class ThreadPoolEngine(ExecutionEngine):
    """Class for executing functions using `ThreadPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.threadpool` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.threadpool"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "timeout",
    }

    def __init__(self, init_kwargs: tp.KwargsLike = None, timeout: tp.Optional[int] = None, **kwargs) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ThreadPoolExecutor`."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        with ProgressHidden():
            with concurrent.futures.ThreadPoolExecutor(**self.init_kwargs) as executor:
                futures = {}
                for i, (func, args, kwargs) in enumerate(tasks):
                    future = executor.submit(func, *args, **kwargs)
                    futures[future] = i
                results = [None] * len(futures)
                for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    results[futures[fut]] = fut.result()
                return results


class ProcessPoolEngine(ExecutionEngine):
    """Class for executing functions using `ProcessPoolExecutor` from `concurrent.futures`.

    For defaults, see `engines.processpool` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.processpool"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "timeout",
    }

    def __init__(self, init_kwargs: tp.KwargsLike = None, timeout: tp.Optional[int] = None, **kwargs) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            timeout=timeout,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `ProcessPoolExecutor`."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        with ProgressHidden():
            with concurrent.futures.ProcessPoolExecutor(**self.init_kwargs) as executor:
                futures = {}
                for i, (func, args, kwargs) in enumerate(tasks):
                    future = executor.submit(func, *args, **kwargs)
                    futures[future] = i
                results = [None] * len(futures)
                for fut in concurrent.futures.as_completed(futures, timeout=self.timeout):
                    results[futures[fut]] = fut.result()
                return results


def pass_kwargs_as_args(func, args, kwargs):
    """Helper function for `pathos.pools.ParallelPool`."""
    return func(*args, **kwargs)


class PathosEngine(ExecutionEngine):
    """Class for executing functions using `pathos`.

    For defaults, see `engines.pathos` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.pathos"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "pool_type",
        "init_kwargs",
        "timeout",
        "check_delay",
        "show_progress",
        "pbar_kwargs",
        "join_pool",
    }

    def __init__(
        self,
        pool_type: tp.Optional[str] = None,
        init_kwargs: tp.KwargsLike = None,
        timeout: tp.Optional[int] = None,
        check_delay: tp.Optional[float] = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        join_pool: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            pool_type=pool_type,
            init_kwargs=init_kwargs,
            timeout=timeout,
            check_delay=check_delay,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            join_pool=join_pool,
            **kwargs,
        )

        self._pool_type = self.resolve_setting(pool_type, "pool_type")
        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._timeout = self.resolve_setting(timeout, "timeout")
        self._check_delay = self.resolve_setting(check_delay, "check_delay")
        self._show_progress = self.resolve_setting(show_progress, "show_progress")
        self._pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)
        self._join_pool = self.resolve_setting(join_pool, "join_pool")

    @property
    def pool_type(self) -> str:
        """Pool type."""
        return self._pool_type

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize the pool."""
        return self._init_kwargs

    @property
    def timeout(self) -> tp.Optional[int]:
        """Timeout."""
        return self._timeout

    @property
    def check_delay(self) -> tp.Optional[float]:
        """Number of seconds to sleep between checks."""
        return self._timeout

    @property
    def show_progress(self) -> bool:
        """Whether to show the progress bar."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    @property
    def join_pool(self) -> bool:
        """Whether to join the pool."""
        return self._join_pool

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("pathos")

        if self.pool_type.lower() in ("thread", "threadpool"):
            from pathos.pools import ThreadPool as Pool
        elif self.pool_type.lower() in ("process", "processpool"):
            from pathos.pools import ProcessPool as Pool
        elif self.pool_type.lower() in ("parallel", "parallelpool"):
            from pathos.pools import ParallelPool as Pool

            tasks = [(pass_kwargs_as_args, x, {}) for x in tasks]
        else:
            raise ValueError(f"Invalid option pool_type='{self.pool_type}'")
        if size is None and hasattr(tasks, "__len__"):
            size = len(tasks)
        elif keys is not None and hasattr(keys, "__len__"):
            size = len(keys)
        pbar_kwargs = dict(self.pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = tuple(keys.names)
                else:
                    pbar_kwargs["bar_id"] = keys.name
        pbar = ProgressBar(total=size, show_progress=self.show_progress, **pbar_kwargs)

        with ProgressHidden():
            with Pool(**self.init_kwargs) as pool:
                async_results = []
                for func, args, kwargs in tasks:
                    async_result = pool.apipe(func, *args, **kwargs)
                    async_results.append(async_result)
                if self.timeout is not None or self.show_progress:
                    pending = set(async_results)
                    total_futures = len(pending)
                    if self.timeout is not None:
                        end_time = self.timeout + time.monotonic()
                    with pbar:
                        while pending:
                            pending = {async_result for async_result in pending if not async_result.ready()}
                            pbar.update_to(total_futures - len(pending))
                            if len(pending) == 0:
                                break
                            if self.timeout is not None:
                                if time.monotonic() > end_time:
                                    raise TimeoutError("%d (of %d) futures unfinished" % (len(pending), total_futures))
                            if self.check_delay is not None:
                                time.sleep(self.check_delay)
                if self.join_pool:
                    pool.close()
                    pool.join()
                    pool.clear()
                return [async_result.get() for async_result in async_results]


class MpireEngine(ExecutionEngine):
    """Class for executing functions using `WorkerPool` from `mpire`.

    For defaults, see `engines.mpire` in `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution.engines.mpire"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "init_kwargs",
        "apply_kwargs",
        "timeout",
    }

    def __init__(
        self,
        init_kwargs: tp.KwargsLike = None,
        apply_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            init_kwargs=init_kwargs,
            apply_kwargs=apply_kwargs,
            **kwargs,
        )

        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._apply_kwargs = self.resolve_setting(apply_kwargs, "apply_kwargs", merge=True)

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments used to initialize `WorkerPool`."""
        return self._init_kwargs

    @property
    def apply_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `WorkerPool.async_apply`."""
        return self._apply_kwargs

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("mpire")
        from mpire import WorkerPool

        with ProgressHidden():
            with WorkerPool(**self.init_kwargs) as pool:
                async_results = []
                for i, (func, args, kwargs) in enumerate(tasks):
                    async_result = pool.apply_async(func, args=args, kwargs=kwargs, **self.apply_kwargs)
                    async_results.append(async_result)
                pool.stop_and_join()
                return [async_result.get() for async_result in async_results]


class DaskEngine(ExecutionEngine):
    """Class for executing functions in parallel using Dask.

    For defaults, see `engines.dask` in `vectorbtpro._settings.execution`.

    !!! note
        Use multi-threading mainly on numeric code that releases the GIL
        (like NumPy, Pandas, Scikit-Learn, Numba)."""

    _settings_path: tp.SettingsPath = "execution.engines.dask"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "compute_kwargs",
    }

    def __init__(self, compute_kwargs: tp.KwargsLike = None, **kwargs) -> None:
        ExecutionEngine.__init__(
            self,
            compute_kwargs=compute_kwargs,
            **kwargs,
        )

        self._compute_kwargs = self.resolve_setting(compute_kwargs, "compute_kwargs", merge=True)

    @property
    def compute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `dask.compute`."""
        return self._compute_kwargs

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("dask")
        import dask

        with ProgressHidden():
            results_delayed = []
            for func, args, kwargs in tasks:
                results_delayed.append(dask.delayed(func)(*args, **kwargs))
            return list(dask.compute(*results_delayed, **self.compute_kwargs))


class RayEngine(ExecutionEngine):
    """Class for executing functions in parallel using Ray.

    For defaults, see `engines.ray` in `vectorbtpro._settings.execution`.

    !!! note
        Ray spawns multiple processes as opposed to threads, so any argument and keyword argument must first
        be put into an object store to be shared. Make sure that the computation with `func` takes
        a considerable amount of time compared to this copying operation, otherwise there will be
        a little to no speedup."""

    _settings_path: tp.SettingsPath = "execution.engines.ray"

    _expected_keys: tp.ExpectedKeys = (ExecutionEngine._expected_keys or set()) | {
        "restart",
        "reuse_refs",
        "del_refs",
        "shutdown",
        "init_kwargs",
        "remote_kwargs",
    }

    def __init__(
        self,
        restart: tp.Optional[bool] = None,
        reuse_refs: tp.Optional[bool] = None,
        del_refs: tp.Optional[bool] = None,
        shutdown: tp.Optional[bool] = None,
        init_kwargs: tp.KwargsLike = None,
        remote_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        ExecutionEngine.__init__(
            self,
            restart=restart,
            reuse_refs=reuse_refs,
            del_refs=del_refs,
            shutdown=shutdown,
            init_kwargs=init_kwargs,
            remote_kwargs=remote_kwargs,
            **kwargs,
        )

        self._restart = self.resolve_setting(restart, "restart")
        self._reuse_refs = self.resolve_setting(reuse_refs, "reuse_refs")
        self._del_refs = self.resolve_setting(del_refs, "del_refs")
        self._shutdown = self.resolve_setting(shutdown, "shutdown")
        self._init_kwargs = self.resolve_setting(init_kwargs, "init_kwargs", merge=True)
        self._remote_kwargs = self.resolve_setting(remote_kwargs, "remote_kwargs", merge=True)

    @property
    def restart(self) -> bool:
        """Whether to terminate the Ray runtime and initialize a new one."""
        return self._restart

    @property
    def reuse_refs(self) -> bool:
        """Whether to re-use function and object references, such that each unique object
        will be copied only once."""
        return self._reuse_refs

    @property
    def del_refs(self) -> bool:
        """Whether to explicitly delete the result object references."""
        return self._del_refs

    @property
    def shutdown(self) -> bool:
        """Whether to True to terminate the Ray runtime upon the job end."""
        return self._shutdown

    @property
    def init_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.init`."""
        return self._init_kwargs

    @property
    def remote_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `ray.remote`."""
        return self._remote_kwargs

    @classmethod
    def get_ray_refs(
        cls,
        tasks: tp.TasksLike,
        reuse_refs: bool = True,
        remote_kwargs: tp.KwargsLike = None,
    ) -> tp.List[tp.Tuple[RemoteFunctionT, tp.Tuple[ObjectRefT, ...], tp.Dict[str, ObjectRefT]]]:
        """Get result references by putting each argument and keyword argument into the object store
        and invoking the remote decorator on each function using Ray.

        If `reuse_refs` is True, will generate one reference per unique object id."""
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray
        from ray.remote_function import RemoteFunction
        from ray import ObjectRef

        if remote_kwargs is None:
            remote_kwargs = {}

        func_id_remotes = {}
        obj_id_refs = {}
        task_refs = []
        for func, args, kwargs in tasks:
            # Get remote function
            if isinstance(func, RemoteFunction):
                func_remote = func
            else:
                if not reuse_refs or id(func) not in func_id_remotes:
                    if isinstance(func, CPUDispatcher):
                        # Numba-wrapped function is not recognized by ray as a function
                        _func = lambda *_args, **_kwargs: func(*_args, **_kwargs)
                    else:
                        _func = func
                    if len(remote_kwargs) > 0:
                        func_remote = ray.remote(**remote_kwargs)(_func)
                    else:
                        func_remote = ray.remote(_func)
                    if reuse_refs:
                        func_id_remotes[id(func)] = func_remote
                else:
                    func_remote = func_id_remotes[id(func)]

            # Get id of each (unique) arg
            arg_refs = ()
            for arg in args:
                if isinstance(arg, ObjectRef):
                    arg_ref = arg
                else:
                    if not reuse_refs or id(arg) not in obj_id_refs:
                        arg_ref = ray.put(arg)
                        obj_id_refs[id(arg)] = arg_ref
                    else:
                        arg_ref = obj_id_refs[id(arg)]
                arg_refs += (arg_ref,)

            # Get id of each (unique) kwarg
            kwarg_refs = {}
            for kwarg_name, kwarg in kwargs.items():
                if isinstance(kwarg, ObjectRef):
                    kwarg_ref = kwarg
                else:
                    if not reuse_refs or id(kwarg) not in obj_id_refs:
                        kwarg_ref = ray.put(kwarg)
                        obj_id_refs[id(kwarg)] = kwarg_ref
                    else:
                        kwarg_ref = obj_id_refs[id(kwarg)]
                kwarg_refs[kwarg_name] = kwarg_ref

            task_refs.append((func_remote, arg_refs, kwarg_refs))
        return task_refs

    def execute(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("ray")
        import ray

        with ProgressHidden():
            if self.restart:
                if ray.is_initialized():
                    ray.shutdown()
            if not ray.is_initialized():
                ray.init(**self.init_kwargs)
            task_refs = self.get_ray_refs(tasks, reuse_refs=self.reuse_refs, remote_kwargs=self.remote_kwargs)
            result_refs = []
            for func_remote, arg_refs, kwarg_refs in task_refs:
                result_refs.append(func_remote.remote(*arg_refs, **kwarg_refs))
            try:
                results = ray.get(result_refs)
            finally:
                if self.del_refs:
                    # clear object store
                    del result_refs
                if self.shutdown:
                    ray.shutdown()
            return results


class _Dummy(enum.Enum):
    """Sentinel that represents a dummy value."""

    DUMMY = enum.auto()

    def __repr__(self):
        return "DUMMY"

    def __bool__(self):
        return False


DUMMY = _Dummy.DUMMY
"""Sentinel that represents a missing value."""


class Executor(Configured):
    """Class responsible executing functions.

    Supported values for `engine`:

    * Name of the engine (see supported engines)
    * Subclass of `ExecutionEngine` - initializes with `engine_config`
    * Instance of `ExecutionEngine` - calls `ExecutionEngine.execute` with `size`
    * Callable - passes `tasks`, `size` (if not None), and `engine_config`

    Can execute per chunk if `chunk_meta` is provided. Otherwise, if any of `n_chunks` and `chunk_len`
    are set, passes them to `vectorbtpro.utils.chunking.yield_chunk_meta` to generate `chunk_meta`.
    Arguments `n_chunks` and `chunk_len` can be set globally in the engine-specific settings.
    Set `n_chunks` and `chunk_len` to 'auto' to set them to the number of cores.

    If `distribute` is "tasks", distributes tasks within each chunk.
    If indices in `chunk_meta` are perfectly sorted and `tasks` is an iterable, iterates
    over `tasks` to avoid converting it into a list. Otherwise, iterates over `chunk_meta`.
    If `in_chunk_order` is True, returns the results in the order they appear in `chunk_meta`.
    Otherwise, always returns them in the same order as in `tasks`.

    If `distribute` is "chunks", distributes chunks. For this, executes tasks within each chunk serially
    using `Executor.execute_serially`. Also, compresses each chunk such that each unique function,
    positional argument, and keyword argument is serialized only once.

    If `tasks` is a custom template, substitutes it once `chunk_meta` is established.
    Use `template_context` as an additional context. All the resolved functions and arguments
    will be immediately passed to the executor.

    If `pre_chunk_func` is not None, calls the function before processing a chunk. If it returns anything
    other than None, the returned object will be appended to the results and the chunk won't be executed.
    This enables use cases such as caching. If `post_chunk_func` is not None, calls the function after
    processing the chunk. It should return either None to keep the old call results, or return new ones.
    Will also substitute any templates in `pre_chunk_kwargs` and `post_chunk_kwargs` and pass them as
    keyword arguments. The following additional arguments are available in the contexts: the index of
    the current chunk `chunk_idx`, the list of call indices `call_indices` in the chunk, the list of call
    results `chunk_cache` returned from caching (only for `pre_chunk_func`), the list of call results
    `call_results` returned by executing the chunk (only for `post_chunk_func`), and whether the chunk
    was executed `chunk_executed` or otherwise returned by `pre_chunk_func` (only for `post_chunk_func`).

    !!! note
        The both callbacks above are effective only when `distribute` is "tasks" and chunking is enabled.

    If `pre_execute_func` is not None, calls the function before processing all tasks. Should return
    nothing (None). Will also substitute any templates in `post_execute_kwargs` and pass them as keyword
    arguments. The following additional arguments are available in the context: the number of chunks `n_chunks`.

    If `post_execute_func` is not None, calls the function after processing all tasks. Will also substitute
    any templates in `post_execute_kwargs` and pass them as keyword arguments. Should return either None
    to keep the default results or return the new ones. The following additional arguments are available
    in the context: the number of chunks `n_chunks` and the generated flattened list of results `results`.
    If `post_execute_on_sorted` is True, will run the callback after sorting the call indices.

    !!! info
        Chunks are processed sequentially, while functions within each chunk can be processed distributively.

    For defaults, see `vectorbtpro._settings.execution`."""

    _settings_path: tp.SettingsPath = "execution"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "engine",
        "engine_config",
        "min_size",
        "n_chunks",
        "chunk_len",
        "chunk_meta",
        "distribute",
        "warmup",
        "in_chunk_order",
        "cache_chunks",
        "chunk_cache_dir",
        "chunk_cache_save_kwargs",
        "chunk_cache_load_kwargs",
        "pre_clear_chunk_cache",
        "post_clear_chunk_cache",
        "release_chunk_cache",
        "chunk_clear_cache",
        "chunk_collect_garbage",
        "chunk_delay",
        "pre_execute_func",
        "pre_execute_kwargs",
        "pre_chunk_func",
        "pre_chunk_kwargs",
        "post_chunk_func",
        "post_chunk_kwargs",
        "post_execute_func",
        "post_execute_kwargs",
        "post_execute_on_sorted",
        "filter_results",
        "raise_no_results",
        "merge_func",
        "merge_kwargs",
        "template_context",
        "show_progress",
        "pbar_kwargs",
    }

    @classmethod
    def get_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> dict:
        """`Executor.get_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """`Executor.has_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def get_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """`Executor.get_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.get_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def has_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> bool:
        """`Executor.has_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.has_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine_setting(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> tp.Any:
        """`Executor.resolve_setting` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        return cls.resolve_setting(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def set_engine_settings(cls, *args, engine_name: tp.Optional[str] = None, **kwargs) -> None:
        """`Executor.set_settings` with `sub_path=engine_name`."""
        if engine_name is not None:
            sub_path = "engines." + engine_name
        else:
            sub_path = None
        cls.set_settings(*args, sub_path=sub_path, **kwargs)

    @classmethod
    def resolve_engine(
        cls,
        engine: tp.ExecutionEngineLike,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **engine_config,
    ) -> tp.Tuple[tp.Union[ExecutionEngine, tp.Callable], tp.Optional[str]]:
        """Resolve engine and its name in settings."""
        from vectorbtpro._settings import settings

        execution_cfg = settings["execution"]
        engines_cfg = execution_cfg["engines"]

        engine_name = None
        if engine is None:
            engine = execution_cfg["engine"]
        if isinstance(engine, str):
            if engine in engines_cfg:
                engine_name = engine
                engine = engines_cfg[engine_name]["cls"]
            elif engine.lower() in engines_cfg:
                engine_name = engine.lower()
                engine = engines_cfg[engine_name]["cls"]
        if isinstance(engine, str):
            globals_dict = globals()
            if engine in globals_dict:
                engine = globals_dict[engine]
            else:
                raise ValueError(f"Invalid engine name '{engine}'")
        if isinstance(engine, type) and issubclass(engine, ExecutionEngine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is engine:
                        engine_name = k
            func_arg_names = get_func_arg_names(engine.__init__)
            if show_progress is not None:
                if (
                    "show_progress" in func_arg_names
                    or (engine_name is not None and "show_progress" in engines_cfg[engine_name])
                ) and "show_progress" not in engine_config:
                    engine_config["show_progress"] = show_progress
            if pbar_kwargs is not None:
                if (
                    "pbar_kwargs" in func_arg_names
                    or (engine_name is not None and "pbar_kwargs" in engines_cfg[engine_name])
                ) and "pbar_kwargs" not in engine_config:
                    engine_config["pbar_kwargs"] = pbar_kwargs
            engine = engine(**engine_config)
        if not isinstance(engine, type) and isinstance(engine, ExecutionEngine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is type(engine):
                        engine_name = k
            if len(engine_config) > 0:
                engine = engine.replace(**engine_config)
        if callable(engine):
            if engine_name is None:
                for k, v in engines_cfg.items():
                    if v["cls"] is engine:
                        engine_name = k
            if engine_name is None:
                if engine.__name__ in engines_cfg:
                    engine_name = engine.__name__
            func_arg_names = get_func_arg_names(engine)
            if show_progress is not None:
                if (
                    "show_progress" in func_arg_names
                    or (engine_name is not None and "show_progress" in engines_cfg[engine_name])
                ) and "show_progress" not in engine_config:
                    engine_config["show_progress"] = show_progress
            if pbar_kwargs is not None:
                if (
                    "pbar_kwargs" in func_arg_names
                    or (engine_name is not None and "pbar_kwargs" in engines_cfg[engine_name])
                ) and "pbar_kwargs" not in engine_config:
                    engine_config["pbar_kwargs"] = pbar_kwargs
            engine = partial(engine, **engine_config)
        if not isinstance(engine, ExecutionEngine) and not callable(engine):
            raise TypeError(f"Invalid engine {engine}")
        return engine, engine_name

    def __init__(
        self,
        engine: tp.Optional[tp.ExecutionEngineLike] = None,
        engine_config: tp.KwargsLike = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Union[None, int, str] = None,
        chunk_len: tp.Union[None, int, str] = None,
        chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
        distribute: tp.Optional[str] = None,
        warmup: tp.Optional[bool] = None,
        in_chunk_order: tp.Optional[bool] = None,
        cache_chunks: tp.Optional[bool] = None,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_save_kwargs: tp.KwargsLike = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        pre_clear_chunk_cache: tp.Optional[bool] = None,
        post_clear_chunk_cache: tp.Optional[bool] = None,
        release_chunk_cache: tp.Optional[bool] = None,
        chunk_clear_cache: tp.Union[None, bool, int] = None,
        chunk_collect_garbage: tp.Union[None, bool, int] = None,
        chunk_delay: tp.Optional[float] = None,
        pre_execute_func: tp.Optional[tp.Callable] = None,
        pre_execute_kwargs: tp.KwargsLike = None,
        pre_chunk_func: tp.Optional[tp.Callable] = None,
        pre_chunk_kwargs: tp.KwargsLike = None,
        post_chunk_func: tp.Optional[tp.Callable] = None,
        post_chunk_kwargs: tp.KwargsLike = None,
        post_execute_func: tp.Optional[tp.Callable] = None,
        post_execute_kwargs: tp.KwargsLike = None,
        post_execute_on_sorted: tp.Optional[bool] = None,
        filter_results: tp.Optional[bool] = None,
        raise_no_results: tp.Optional[bool] = None,
        merge_func: tp.Optional[tp.MergeFuncLike] = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            engine=engine,
            engine_config=engine_config,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            distribute=distribute,
            warmup=warmup,
            in_chunk_order=in_chunk_order,
            cache_chunks=cache_chunks,
            chunk_cache_dir=chunk_cache_dir,
            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
            pre_clear_chunk_cache=pre_clear_chunk_cache,
            post_clear_chunk_cache=post_clear_chunk_cache,
            release_chunk_cache=release_chunk_cache,
            chunk_clear_cache=chunk_clear_cache,
            chunk_collect_garbage=chunk_collect_garbage,
            chunk_delay=chunk_delay,
            pre_execute_func=pre_execute_func,
            pre_execute_kwargs=pre_execute_kwargs,
            pre_chunk_func=pre_chunk_func,
            pre_chunk_kwargs=pre_chunk_kwargs,
            post_chunk_func=post_chunk_func,
            post_chunk_kwargs=post_chunk_kwargs,
            post_execute_func=post_execute_func,
            post_execute_kwargs=post_execute_kwargs,
            post_execute_on_sorted=post_execute_on_sorted,
            filter_results=filter_results,
            raise_no_results=raise_no_results,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            template_context=template_context,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            **kwargs,
        )

        if engine_config is None:
            engine_config = {}
        engine, engine_name = self.resolve_engine(
            engine,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
            **engine_config,
        )
        min_size = self.resolve_engine_setting(
            min_size,
            "min_size",
            engine_name=engine_name,
        )
        n_chunks = self.resolve_engine_setting(
            n_chunks,
            "n_chunks",
            engine_name=engine_name,
        )
        chunk_len = self.resolve_engine_setting(
            chunk_len,
            "chunk_len",
            engine_name=engine_name,
        )
        chunk_meta = self.resolve_engine_setting(
            chunk_meta,
            "chunk_meta",
            engine_name=engine_name,
        )
        distribute = self.resolve_engine_setting(
            distribute,
            "distribute",
            engine_name=engine_name,
        )
        warmup = self.resolve_engine_setting(
            warmup,
            "warmup",
            engine_name=engine_name,
        )
        in_chunk_order = self.resolve_engine_setting(
            in_chunk_order,
            "in_chunk_order",
            engine_name=engine_name,
        )
        cache_chunks = self.resolve_engine_setting(
            cache_chunks,
            "cache_chunks",
            engine_name=engine_name,
        )
        chunk_cache_dir = self.resolve_engine_setting(
            chunk_cache_dir,
            "chunk_cache_dir",
            engine_name=engine_name,
        )
        chunk_cache_save_kwargs = self.resolve_engine_setting(
            chunk_cache_save_kwargs,
            "chunk_cache_save_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        chunk_cache_load_kwargs = self.resolve_engine_setting(
            chunk_cache_load_kwargs,
            "chunk_cache_load_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        pre_clear_chunk_cache = self.resolve_engine_setting(
            pre_clear_chunk_cache,
            "pre_clear_chunk_cache",
            engine_name=engine_name,
        )
        post_clear_chunk_cache = self.resolve_engine_setting(
            post_clear_chunk_cache,
            "post_clear_chunk_cache",
            engine_name=engine_name,
        )
        release_chunk_cache = self.resolve_engine_setting(
            release_chunk_cache,
            "release_chunk_cache",
            engine_name=engine_name,
        )
        chunk_clear_cache = self.resolve_engine_setting(
            chunk_clear_cache,
            "chunk_clear_cache",
            engine_name=engine_name,
        )
        chunk_collect_garbage = self.resolve_engine_setting(
            chunk_collect_garbage,
            "chunk_collect_garbage",
            engine_name=engine_name,
        )
        chunk_delay = self.resolve_engine_setting(
            chunk_delay,
            "chunk_delay",
            engine_name=engine_name,
        )
        pre_execute_func = self.resolve_engine_setting(
            pre_execute_func,
            "pre_execute_func",
            engine_name=engine_name,
        )
        pre_execute_kwargs = self.resolve_engine_setting(
            pre_execute_kwargs,
            "pre_execute_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        pre_chunk_func = self.resolve_engine_setting(
            pre_chunk_func,
            "pre_chunk_func",
            engine_name=engine_name,
        )
        pre_chunk_kwargs = self.resolve_engine_setting(
            pre_chunk_kwargs,
            "pre_chunk_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_chunk_func = self.resolve_engine_setting(
            post_chunk_func,
            "post_chunk_func",
            engine_name=engine_name,
        )
        post_chunk_kwargs = self.resolve_engine_setting(
            post_chunk_kwargs,
            "post_chunk_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_execute_func = self.resolve_engine_setting(
            post_execute_func,
            "post_execute_func",
            engine_name=engine_name,
        )
        post_execute_kwargs = self.resolve_engine_setting(
            post_execute_kwargs,
            "post_execute_kwargs",
            merge=True,
            engine_name=engine_name,
        )
        post_execute_on_sorted = self.resolve_engine_setting(
            post_execute_on_sorted,
            "post_execute_on_sorted",
            engine_name=engine_name,
        )
        if release_chunk_cache and post_execute_on_sorted:
            raise ValueError("Cannot use release_chunk_cache and post_execute_on_sorted together")
        filter_results = self.resolve_engine_setting(filter_results, "filter_results")
        raise_no_results = self.resolve_engine_setting(raise_no_results, "raise_no_results")
        merge_func = self.resolve_engine_setting(merge_func, "merge_func")
        merge_kwargs = self.resolve_engine_setting(merge_kwargs, "merge_kwargs", merge=True)
        template_context = self.resolve_engine_setting(
            template_context,
            "template_context",
            merge=True,
            engine_name=engine_name,
        )
        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        self._engine = engine
        self._min_size = min_size
        self._n_chunks = n_chunks
        self._chunk_len = chunk_len
        self._chunk_meta = chunk_meta
        self._distribute = distribute
        self._warmup = warmup
        self._in_chunk_order = in_chunk_order
        self._cache_chunks = cache_chunks
        self._chunk_cache_dir = chunk_cache_dir
        self._chunk_cache_save_kwargs = chunk_cache_save_kwargs
        self._chunk_cache_load_kwargs = chunk_cache_load_kwargs
        self._pre_clear_chunk_cache = pre_clear_chunk_cache
        self._post_clear_chunk_cache = post_clear_chunk_cache
        self._release_chunk_cache = release_chunk_cache
        self._chunk_clear_cache = chunk_clear_cache
        self._chunk_collect_garbage = chunk_collect_garbage
        self._chunk_delay = chunk_delay
        self._pre_execute_func = pre_execute_func
        self._pre_execute_kwargs = pre_execute_kwargs
        self._pre_chunk_func = pre_chunk_func
        self._pre_chunk_kwargs = pre_chunk_kwargs
        self._post_chunk_func = post_chunk_func
        self._post_chunk_kwargs = post_chunk_kwargs
        self._post_execute_func = post_execute_func
        self._post_execute_kwargs = post_execute_kwargs
        self._post_execute_on_sorted = post_execute_on_sorted
        self._filter_results = filter_results
        self._raise_no_results = raise_no_results
        self._merge_func = merge_func
        self._merge_kwargs = merge_kwargs
        self._template_context = template_context
        self._show_progress = show_progress
        self._pbar_kwargs = pbar_kwargs

    @property
    def engine(self) -> tp.Union[ExecutionEngine, tp.Callable]:
        """Engine resolved with `Executor.resolve_engine`."""
        return self._engine

    @property
    def min_size(self) -> tp.Optional[int]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`."""
        return self._min_size

    @property
    def n_chunks(self) -> tp.Union[None, int, str]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`."""
        return self._n_chunks

    @property
    def chunk_len(self) -> tp.Union[None, int, str]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`."""
        return self._chunk_len

    @property
    def chunk_meta(self) -> tp.Optional[tp.ChunkMetaLike]:
        """See `vectorbtpro.utils.chunking.yield_chunk_meta`."""
        return self._chunk_meta

    @property
    def distribute(self) -> str:
        """Distribution mode."""
        return self._distribute

    @property
    def warmup(self) -> bool:
        """Whether to call the first item of `tasks` once before distribution."""
        return self._warmup

    @property
    def in_chunk_order(self) -> bool:
        """Whether to return the results in the order they appear in `chunk_meta`.

        Otherwise, always returns them in the same order as in `tasks`."""
        return self._in_chunk_order

    @property
    def cache_chunks(self) -> bool:
        """Whether to cache chunks."""
        return self._cache_chunks

    @property
    def chunk_cache_dir(self) -> tp.PathLike:
        """Directory where to put chunk cache files."""
        return self._chunk_cache_dir

    @property
    def chunk_cache_save_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pickling.save` for chunk caching."""
        return self._chunk_cache_save_kwargs

    @property
    def chunk_cache_load_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pickling.load` for chunk caching."""
        return self._chunk_cache_load_kwargs

    @property
    def pre_clear_chunk_cache(self) -> bool:
        """Whether to remove the chunk cache directory before execution."""
        return self._pre_clear_chunk_cache

    @property
    def post_clear_chunk_cache(self) -> bool:
        """Whether to remove the chunk cache directory after execution."""
        return self._post_clear_chunk_cache

    @property
    def release_chunk_cache(self) -> bool:
        """Whether to replace chunk cache with dummy objects once the chunk has been executed
        and then load all cache at once after all chunks have been executed."""
        return self._release_chunk_cache

    @property
    def chunk_clear_cache(self) -> tp.Union[bool, int]:
        """Whether to clear global cache after each chunk or every n chunks."""
        return self._chunk_clear_cache

    @property
    def chunk_collect_garbage(self) -> tp.Union[bool, int]:
        """Whether to collect garbage after each chunk or every n chunks."""
        return self._chunk_collect_garbage

    @property
    def chunk_delay(self) -> tp.Optional[float]:
        """Number of seconds to sleep after each chunk."""
        return self._chunk_delay

    @property
    def pre_execute_func(self) -> tp.Optional[tp.Callable]:
        """Function to call before processing all tasks."""
        return self._pre_execute_func

    @property
    def pre_execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Executor.pre_execute_func`."""
        return self._pre_execute_kwargs

    @property
    def pre_chunk_func(self) -> tp.Optional[tp.Callable]:
        """Function to call before processing a chunk.

        If it returns anything other than None, the returned object will be appended to the
        results and the chunk won't be executed. This enables use cases such as caching."""
        return self._pre_chunk_func

    @property
    def pre_chunk_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Executor.pre_chunk_func`."""
        return self._pre_chunk_kwargs

    @property
    def post_chunk_func(self) -> tp.Optional[tp.Callable]:
        """Function to call after processing the chunk.

        It should return either None to keep the old call results, or return new ones."""
        return self._post_chunk_func

    @property
    def post_chunk_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Executor.post_chunk_func`."""
        return self._post_chunk_kwargs

    @property
    def post_execute_func(self) -> tp.Optional[tp.Callable]:
        """Function to call after processing all tasks.

        Should return either None to keep the default results, or return the new ones."""
        return self._post_execute_func

    @property
    def post_execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `Executor.post_execute_func`."""
        return self._post_execute_kwargs

    @property
    def post_execute_on_sorted(self) -> bool:
        """Whether to run `Executor.post_execute_func` after sorting the call indices."""
        return self._post_execute_func

    @property
    def filter_results(self) -> bool:
        """Whether to filter `NoResult` results."""
        return self._filter_results

    @property
    def raise_no_results(self) -> bool:
        """Whether to raise `NoResultsException` if there are no results. Otherwise, returns `NoResult`.

        Has effect only if `Executor.filter_results` is True. But regardless of this setting,
        gets passed to the merging function if the merging function is pre-configured."""
        return self._raise_no_results

    @property
    def merge_func(self) -> tp.Optional[tp.MergeFuncLike]:
        """Merging function.

        Resolved using `vectorbtpro.base.merging.resolve_merge_func`."""
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the merging function."""
        return self._merge_kwargs

    @property
    def template_context(self) -> tp.Kwargs:
        """Context used to substitute templates."""
        return self._template_context

    @property
    def show_progress(self) -> bool:
        """Whether to show progress bar when iterating over chunks.

        If `Executor.engine` accepts `show_progress` and there's no key `show_progress`
        in `Executor.engine_config`, then passes it to the engine as well."""
        return self._show_progress

    @property
    def pbar_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`."""
        return self._pbar_kwargs

    @staticmethod
    def execute_serially(tasks: tp.TasksLike, id_objs: tp.Dict[int, tp.Any]) -> tp.ExecResults:
        """Execute serially."""
        results = []
        for func, args, kwargs in tasks:
            new_func = id_objs[func]
            new_args = tuple(id_objs[arg] for arg in args)
            new_kwargs = {k: id_objs[v] for k, v in kwargs.items()}
            results.append(new_func(*new_args, **new_kwargs))
        return results

    @classmethod
    def build_serial_chunk(cls, tasks: tp.TasksLike) -> Task:
        """Build a serial chunk."""
        ref_ids = dict()
        id_objs = dict()

        def _prepare(x):
            if id(x) in ref_ids:
                return ref_ids[id(x)]
            new_id = len(id_objs)
            ref_ids[id(x)] = new_id
            id_objs[new_id] = x
            return new_id

        new_tasks = []
        for func, args, kwargs in tasks:
            new_func = _prepare(func)
            new_args = tuple(_prepare(arg) for arg in args)
            new_kwargs = {k: _prepare(v) for k, v in kwargs.items()}
            new_tasks.append(Task(new_func, *new_args, **new_kwargs))
        return Task(cls.execute_serially, new_tasks, id_objs)

    @classmethod
    def call_pre_execute_func(
        cls,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        pre_clear_chunk_cache: bool = False,
        pre_execute_func: tp.Optional[tp.Callable] = None,
        pre_execute_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> None:
        """Call `Executor.pre_execute_func`."""
        if cache_chunks and pre_clear_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            remove_dir(chunk_cache_dir, missing_ok=True, with_contents=True)

        if pre_execute_kwargs is None:
            pre_execute_kwargs = {}
        if pre_execute_func is not None:
            pre_execute_func = substitute_templates(
                pre_execute_func,
                template_context,
                eval_id="pre_execute_func",
            )
            pre_execute_kwargs = substitute_templates(
                pre_execute_kwargs,
                template_context,
                eval_id="pre_execute_kwargs",
            )
            pre_execute_func(**pre_execute_kwargs)

    @classmethod
    def call_pre_chunk_func(
        cls,
        chunk_idx: int,
        call_indices: tp.List[int],
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        release_chunk_cache: bool = False,
        pre_chunk_func: tp.Optional[tp.Callable] = None,
        pre_chunk_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Optional[tp.ExecResults]:
        """Call `Executor.pre_chunk_func`."""
        chunk_cache = None
        if cache_chunks:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            if not isinstance(chunk_cache_dir, Path):
                chunk_cache_dir = Path(chunk_cache_dir)
            chunk_path = chunk_cache_dir / ("chunk_%d.pickle" % chunk_idx)
            if file_exists(chunk_path):
                if release_chunk_cache:
                    chunk_cache = [DUMMY] * len(call_indices)
                else:
                    if chunk_cache_load_kwargs is None:
                        chunk_cache_load_kwargs = {}
                    chunk_cache = load(chunk_path, **chunk_cache_load_kwargs)

        if pre_chunk_func is not None:
            template_context = merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    call_indices=call_indices,
                    chunk_cache=chunk_cache,
                ),
                template_context,
            )
            pre_chunk_func = substitute_templates(
                pre_chunk_func,
                template_context,
                eval_id="pre_chunk_func",
            )
            pre_chunk_kwargs = substitute_templates(
                pre_chunk_kwargs,
                template_context,
                eval_id="pre_chunk_kwargs",
            )
            call_results = pre_chunk_func(**pre_chunk_kwargs)
            if call_results is not None:
                return call_results
        return chunk_cache

    @classmethod
    def call_execute(
        cls,
        engine: tp.Union[ExecutionEngine, tp.Callable],
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.ExecResults:
        """Call `ExecutionEngine.execute`."""
        if isinstance(engine, ExecutionEngine):
            return engine.execute(tasks, size=size, keys=keys)
        func_arg_names = get_func_arg_names(engine)
        execute_kwargs = {}
        if "size" in func_arg_names:
            execute_kwargs["size"] = size
        if "keys" in func_arg_names:
            execute_kwargs["keys"] = keys
        return engine(tasks, **execute_kwargs)

    @classmethod
    def call_post_chunk_func(
        cls,
        chunk_idx: int,
        call_indices: tp.List[int],
        call_results: tp.ExecResults,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_save_kwargs: tp.KwargsLike = None,
        release_chunk_cache: bool = False,
        chunk_clear_cache: tp.Union[bool, int] = False,
        chunk_collect_garbage: tp.Union[bool, int] = False,
        chunk_delay: tp.Optional[float] = None,
        post_chunk_func: tp.Optional[tp.Callable] = None,
        post_chunk_kwargs: tp.KwargsLike = None,
        chunk_executed: bool = True,
        template_context: tp.KwargsLike = None,
    ) -> tp.ExecResults:
        """Call `Executor.post_chunk_func`."""
        from vectorbtpro.registries.ca_registry import clear_cache, collect_garbage

        if chunk_executed:
            if cache_chunks:
                if chunk_cache_dir is None:
                    raise ValueError("Must provide chunk_cache_dir")
                if not isinstance(chunk_cache_dir, Path):
                    chunk_cache_dir = Path(chunk_cache_dir)
                chunk_path = chunk_cache_dir / ("chunk_%d.pickle" % chunk_idx)
                if chunk_cache_save_kwargs is None:
                    chunk_cache_save_kwargs = {}
                save(call_results, chunk_path, **chunk_cache_save_kwargs)
                if release_chunk_cache:
                    call_results = [DUMMY] * len(call_indices)

        if post_chunk_func is not None:
            template_context = merge_dicts(
                dict(
                    chunk_idx=chunk_idx,
                    call_indices=call_indices,
                    call_results=call_results,
                    chunk_executed=chunk_executed,
                ),
                template_context,
            )
            post_chunk_func = substitute_templates(
                post_chunk_func,
                template_context,
                eval_id="post_chunk_func",
            )
            post_chunk_kwargs = substitute_templates(
                post_chunk_kwargs,
                template_context,
                eval_id="post_chunk_kwargs",
            )
            new_call_results = post_chunk_func(**post_chunk_kwargs)
            if new_call_results is not None:
                call_results = new_call_results

        if isinstance(chunk_clear_cache, bool):
            if chunk_clear_cache:
                clear_cache()
        elif chunk_idx > 0 and (chunk_idx + 1) % chunk_clear_cache == 0:
            clear_cache()
        if isinstance(chunk_collect_garbage, bool):
            if chunk_collect_garbage:
                collect_garbage()
        elif chunk_idx > 0 and (chunk_idx + 1) % chunk_collect_garbage == 0:
            collect_garbage()
        if chunk_delay is not None:
            time.sleep(chunk_delay)

        return call_results

    @classmethod
    def call_post_execute_func(
        cls,
        results: tp.ExecResults,
        cache_chunks: bool = False,
        chunk_cache_dir: tp.Optional[tp.PathLike] = None,
        chunk_cache_load_kwargs: tp.KwargsLike = None,
        post_clear_chunk_cache: bool = True,
        release_chunk_cache: bool = False,
        post_execute_func: tp.Optional[tp.Callable] = None,
        post_execute_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Optional[tp.ExecResults]:
        """Call `Executor.post_execute_func`."""
        if cache_chunks and release_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            if not isinstance(chunk_cache_dir, Path):
                chunk_cache_dir = Path(chunk_cache_dir)
            if chunk_cache_load_kwargs is None:
                chunk_cache_load_kwargs = {}
            chunk_paths = Path(chunk_cache_dir).rglob("chunk_*.pickle")
            chunk_paths = sorted(chunk_paths, key=lambda x: int(x.name.split("_")[1].split(".")[0]))
            new_results = []
            for chunk_path in chunk_paths:
                chunk_cache = load(chunk_path, **chunk_cache_load_kwargs)
                new_results.extend(chunk_cache)
            results = new_results
        if cache_chunks and post_clear_chunk_cache:
            if chunk_cache_dir is None:
                raise ValueError("Must provide chunk_cache_dir")
            remove_dir(chunk_cache_dir, missing_ok=True, with_contents=True)

        if post_execute_func is not None:
            template_context = merge_dicts(
                dict(results=results),
                template_context,
            )
            post_execute_func = substitute_templates(
                post_execute_func,
                template_context,
                eval_id="post_execute_func",
            )
            post_execute_kwargs = substitute_templates(
                post_execute_kwargs,
                template_context,
                eval_id="post_execute_kwargs",
            )
            new_results = post_execute_func(**post_execute_kwargs)
            if new_results is not None:
                return new_results
        return results

    @classmethod
    def merge_results(
        cls,
        results: tp.ExecResults,
        keys: tp.Optional[tp.IndexLike] = None,
        filter_results: bool = False,
        raise_no_results: bool = True,
        merge_func: tp.Optional[tp.MergeFuncLike] = None,
        merge_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Union[tp.ExecResults, tp.MergeResult]:
        """Merge results using `Executor.merge_func` and `Executor.merge_kwargs`."""
        if filter_results:
            try:
                results, keys = filter_out_no_results(results, keys=keys)
            except NoResultsException as e:
                if raise_no_results:
                    raise e
                return NoResult
            no_results_filtered = True
        else:
            no_results_filtered = False
        if merge_func is not None:
            template_context = merge_dicts(
                dict(results=results),
                template_context,
            )
            from vectorbtpro.base.merging import is_merge_func_from_config

            if is_merge_func_from_config(merge_func):
                merge_kwargs = merge_dicts(
                    dict(
                        keys=keys,
                        filter_results=not no_results_filtered,
                        raise_no_results=raise_no_results,
                    ),
                    merge_kwargs,
                )
            if isinstance(merge_func, MergeFunc):
                merge_func = merge_func.replace(merge_kwargs=merge_kwargs, context=template_context)
            else:
                merge_func = MergeFunc(merge_func, merge_kwargs=merge_kwargs, context=template_context)
            return merge_func(results)
        return results

    def run(
        self,
        tasks: tp.TasksLike,
        size: tp.Optional[int] = None,
        keys: tp.Optional[tp.IndexLike] = None,
    ) -> tp.Union[tp.ExecResults, tp.MergeResult]:
        """Execute functions and their arguments."""
        from vectorbtpro.base.indexes import to_any_index

        engine = self.engine
        min_size = self.min_size
        n_chunks = self.n_chunks
        chunk_len = self.chunk_len
        chunk_meta = self.chunk_meta
        distribute = self.distribute
        warmup = self.warmup
        in_chunk_order = self.in_chunk_order
        cache_chunks = self.cache_chunks
        chunk_cache_dir = self.chunk_cache_dir
        chunk_cache_load_kwargs = self.chunk_cache_load_kwargs
        chunk_cache_save_kwargs = self.chunk_cache_save_kwargs
        pre_clear_chunk_cache = self.pre_clear_chunk_cache
        post_clear_chunk_cache = self.post_clear_chunk_cache
        release_chunk_cache = self.release_chunk_cache
        chunk_clear_cache = self.chunk_clear_cache
        chunk_collect_garbage = self.chunk_collect_garbage
        chunk_delay = self.chunk_delay
        pre_execute_func = self.pre_execute_func
        pre_execute_kwargs = self.pre_execute_kwargs
        pre_chunk_func = self.pre_chunk_func
        pre_chunk_kwargs = self.pre_chunk_kwargs
        post_chunk_func = self.post_chunk_func
        post_chunk_kwargs = self.post_chunk_kwargs
        post_execute_func = self.post_execute_func
        post_execute_kwargs = self.post_execute_kwargs
        post_execute_on_sorted = self.post_execute_on_sorted
        filter_results = self.filter_results
        raise_no_results = self.raise_no_results
        merge_func = self.merge_func
        merge_kwargs = self.merge_kwargs
        template_context = self.template_context
        show_progress = self.show_progress
        pbar_kwargs = self.pbar_kwargs

        if keys is not None:
            keys = to_any_index(keys)
        pbar_kwargs = dict(pbar_kwargs)
        if "bar_id" not in pbar_kwargs:
            if keys is not None:
                if isinstance(keys, pd.MultiIndex):
                    pbar_kwargs["bar_id"] = ("chunk_tasks", tuple(keys.names))
                else:
                    pbar_kwargs["bar_id"] = ("chunk_tasks", keys.name)
        if cache_chunks and chunk_cache_dir is None:
            if "bar_id" in pbar_kwargs:
                import hashlib

                chunk_cache_dir_hash = hashlib.md5(str(pbar_kwargs["bar_id"]).encode("utf-8")).hexdigest()
                chunk_cache_dir = "chunk_cache_%s" % chunk_cache_dir_hash

        if warmup:
            if not hasattr(tasks, "__getitem__"):
                tasks = list(tasks)
            tasks[0][0](*tasks[0][1], **tasks[0][2])

        if n_chunks is None and chunk_len is None and chunk_meta is None:
            if isinstance(tasks, CustomTemplate):
                n_chunks = 1
            else:
                if cache_chunks:
                    warnings.warn("Cannot cache chunks without chunking", stacklevel=2)
                    cache_chunks = False
                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                if "n_chunks" not in template_context:
                    template_context["n_chunks"] = 1
                results = self.call_execute(engine, tasks, size=size, keys=keys)
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )

        if chunk_meta is None:
            from vectorbtpro.utils.chunking import yield_chunk_meta

            if not isinstance(tasks, CustomTemplate) and hasattr(tasks, "__len__"):
                _size = len(tasks)
            elif size is not None:
                _size = size
            elif keys is not None and hasattr(keys, "__len__"):
                _size = len(keys)
            else:
                if isinstance(tasks, CustomTemplate):
                    raise ValueError("When tasks is a template, must provide size")
                tasks = list(tasks)
                _size = len(tasks)
            chunk_meta = yield_chunk_meta(
                size=_size,
                min_size=min_size,
                n_chunks=n_chunks,
                chunk_len=chunk_len,
            )
            if "chunk_meta" not in template_context:
                template_context["chunk_meta"] = chunk_meta

        if isinstance(tasks, CustomTemplate):
            if cache_chunks:
                warnings.warn("Cannot cache chunks with custom chunking", stacklevel=2)
                cache_chunks = False
            tasks = substitute_templates(tasks, template_context, eval_id="tasks")
            if hasattr(tasks, "__len__"):
                size = len(tasks)
            elif keys is not None and hasattr(keys, "__len__"):
                size = len(keys)
            else:
                size = None
            self.call_pre_execute_func(
                cache_chunks=cache_chunks,
                chunk_cache_dir=chunk_cache_dir,
                pre_clear_chunk_cache=pre_clear_chunk_cache,
                pre_execute_func=pre_execute_func,
                pre_execute_kwargs=pre_execute_kwargs,
                template_context=template_context,
            )
            if "n_chunks" not in template_context:
                template_context["n_chunks"] = 1
            results = self.call_execute(
                engine,
                tasks,
                size=size,
                keys=keys,
            )
            results = self.call_post_execute_func(
                results,
                cache_chunks=cache_chunks,
                chunk_cache_dir=chunk_cache_dir,
                chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                post_clear_chunk_cache=post_clear_chunk_cache,
                release_chunk_cache=release_chunk_cache,
                post_execute_func=post_execute_func,
                post_execute_kwargs=post_execute_kwargs,
                template_context=template_context,
            )
            return self.merge_results(
                results,
                keys=keys,
                filter_results=filter_results,
                raise_no_results=raise_no_results,
                merge_func=merge_func,
                merge_kwargs=merge_kwargs,
                template_context=template_context,
            )

        last_idx = -1
        indices_sorted = True
        all_call_indices = []
        for _chunk_meta in chunk_meta:
            if _chunk_meta.indices is not None:
                call_indices = list(_chunk_meta.indices)
            else:
                if _chunk_meta.start is None or _chunk_meta.end is None:
                    raise ValueError("Each chunk must have a start and an end index")
                call_indices = list(range(_chunk_meta.start, _chunk_meta.end))
            if indices_sorted:
                for idx in call_indices:
                    if idx != last_idx + 1:
                        indices_sorted = False
                        break
                    last_idx = idx
            all_call_indices.append(call_indices)
        if "n_chunks" not in template_context:
            template_context["n_chunks"] = len(all_call_indices)

        if distribute.lower() == "tasks":
            if indices_sorted and not hasattr(tasks, "__len__"):
                results = []
                chunk_idx = 0
                _tasks = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                with ProgressBar(total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                    pbar.set_description(
                        dict(
                            chunk_tasks="{}..{}".format(
                                all_call_indices[chunk_idx][0],
                                all_call_indices[chunk_idx][-1],
                            )
                        )
                    )
                    for i, func_args in enumerate(tasks):
                        if i > all_call_indices[chunk_idx][-1]:
                            call_indices = all_call_indices[chunk_idx]
                            call_results = self.call_pre_chunk_func(
                                chunk_idx,
                                call_indices,
                                cache_chunks=cache_chunks,
                                chunk_cache_dir=chunk_cache_dir,
                                chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                                release_chunk_cache=release_chunk_cache,
                                pre_chunk_func=pre_chunk_func,
                                pre_chunk_kwargs=pre_chunk_kwargs,
                                template_context=template_context,
                            )
                            if call_results is None:
                                call_results = self.call_execute(
                                    engine,
                                    _tasks,
                                    size=len(call_indices),
                                    keys=keys[call_indices] if keys is not None else None,
                                )
                                chunk_executed = True
                            else:
                                chunk_executed = False
                            call_results = self.call_post_chunk_func(
                                chunk_idx,
                                call_indices,
                                call_results,
                                cache_chunks=cache_chunks,
                                chunk_cache_dir=chunk_cache_dir,
                                chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                                release_chunk_cache=release_chunk_cache,
                                chunk_clear_cache=chunk_clear_cache,
                                chunk_collect_garbage=chunk_collect_garbage,
                                chunk_delay=chunk_delay,
                                post_chunk_func=post_chunk_func,
                                post_chunk_kwargs=post_chunk_kwargs,
                                chunk_executed=chunk_executed,
                                template_context=template_context,
                            )
                            results.extend(call_results)
                            chunk_idx += 1
                            _tasks = []
                            if chunk_idx < len(all_call_indices):
                                pbar.set_description(
                                    dict(
                                        chunk_tasks="{}..{}".format(
                                            all_call_indices[chunk_idx][0],
                                            all_call_indices[chunk_idx][-1],
                                        )
                                    )
                                )
                            pbar.update()
                        _tasks.append(func_args)
                    if len(_tasks) > 0:
                        call_indices = all_call_indices[chunk_idx]
                        call_results = self.call_pre_chunk_func(
                            chunk_idx,
                            call_indices,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            pre_chunk_func=pre_chunk_func,
                            pre_chunk_kwargs=pre_chunk_kwargs,
                            template_context=template_context,
                        )
                        if call_results is None:
                            call_results = self.call_execute(
                                engine,
                                _tasks,
                                size=len(call_indices),
                                keys=keys[call_indices] if keys is not None else None,
                            )
                            chunk_executed = True
                        else:
                            chunk_executed = False
                        call_results = self.call_post_chunk_func(
                            chunk_idx,
                            call_indices,
                            call_results,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            chunk_clear_cache=chunk_clear_cache,
                            chunk_collect_garbage=chunk_collect_garbage,
                            chunk_delay=chunk_delay,
                            post_chunk_func=post_chunk_func,
                            post_chunk_kwargs=post_chunk_kwargs,
                            chunk_executed=chunk_executed,
                            template_context=template_context,
                        )
                        results.extend(call_results)
                        pbar.update()
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
            else:
                tasks = list(tasks)
                results = []
                output_indices = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                with ProgressBar(total=len(all_call_indices), show_progress=show_progress, **pbar_kwargs) as pbar:
                    pbar.set_description(
                        dict(
                            chunk_tasks="{}..{}".format(
                                all_call_indices[0][0],
                                all_call_indices[0][-1],
                            )
                        )
                    )
                    for chunk_idx, call_indices in enumerate(all_call_indices):
                        call_results = self.call_pre_chunk_func(
                            chunk_idx,
                            call_indices,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            pre_chunk_func=pre_chunk_func,
                            pre_chunk_kwargs=pre_chunk_kwargs,
                            template_context=template_context,
                        )
                        if call_results is None:
                            _tasks = []
                            for idx in call_indices:
                                _tasks.append(tasks[idx])
                            call_results = self.call_execute(
                                engine,
                                _tasks,
                                size=len(call_indices),
                                keys=keys[call_indices] if keys is not None else None,
                            )
                            chunk_executed = True
                        else:
                            chunk_executed = False
                        call_results = self.call_post_chunk_func(
                            chunk_idx,
                            call_indices,
                            call_results,
                            cache_chunks=cache_chunks,
                            chunk_cache_dir=chunk_cache_dir,
                            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
                            release_chunk_cache=release_chunk_cache,
                            chunk_clear_cache=chunk_clear_cache,
                            chunk_collect_garbage=chunk_collect_garbage,
                            chunk_delay=chunk_delay,
                            post_chunk_func=post_chunk_func,
                            post_chunk_kwargs=post_chunk_kwargs,
                            chunk_executed=chunk_executed,
                            template_context=template_context,
                        )
                        results.extend(call_results)
                        output_indices.extend(call_indices)
                        if chunk_idx + 1 < len(all_call_indices):
                            pbar.set_description(
                                dict(
                                    chunk_tasks="{}..{}".format(
                                        all_call_indices[chunk_idx + 1][0],
                                        all_call_indices[chunk_idx + 1][-1],
                                    )
                                )
                            )
                        pbar.update()
                if not post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                if not in_chunk_order and not indices_sorted:
                    results = [x for _, x in sorted(zip(output_indices, results))]
                if post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )

        elif distribute.lower() == "chunks":
            if cache_chunks:
                warnings.warn("Cannot cache chunks with chunk distribution", stacklevel=2)
                cache_chunks = False
            if indices_sorted and not hasattr(tasks, "__len__"):
                chunk_idx = 0
                _tasks = []
                task_chunks = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                for i, func_args in enumerate(tasks):
                    if i > all_call_indices[chunk_idx][-1]:
                        task_chunks.append(self.build_serial_chunk(_tasks))
                        chunk_idx += 1
                        _tasks = []
                    _tasks.append(func_args)
                if len(_tasks) > 0:
                    task_chunks.append(self.build_serial_chunk(_tasks))
                results = self.call_execute(
                    engine,
                    task_chunks,
                    size=len(task_chunks),
                )
                results = [x for o in results for x in o]
                results = self.call_post_execute_func(
                    results,
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                    post_clear_chunk_cache=post_clear_chunk_cache,
                    release_chunk_cache=release_chunk_cache,
                    post_execute_func=post_execute_func,
                    post_execute_kwargs=post_execute_kwargs,
                    template_context=template_context,
                )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
            else:
                tasks = list(tasks)
                task_chunks = []
                output_indices = []

                self.call_pre_execute_func(
                    cache_chunks=cache_chunks,
                    chunk_cache_dir=chunk_cache_dir,
                    pre_clear_chunk_cache=pre_clear_chunk_cache,
                    pre_execute_func=pre_execute_func,
                    pre_execute_kwargs=pre_execute_kwargs,
                    template_context=template_context,
                )
                for call_indices in all_call_indices:
                    _tasks = []
                    for idx in call_indices:
                        _tasks.append(tasks[idx])
                    task_chunks.append(self.build_serial_chunk(_tasks))
                    output_indices.extend(call_indices)
                results = self.call_execute(
                    engine,
                    task_chunks,
                    size=len(task_chunks),
                )
                results = [x for o in results for x in o]
                if not post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                if not in_chunk_order and not indices_sorted:
                    results = [x for _, x in sorted(zip(output_indices, results))]
                if post_execute_on_sorted:
                    results = self.call_post_execute_func(
                        results,
                        cache_chunks=cache_chunks,
                        chunk_cache_dir=chunk_cache_dir,
                        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
                        post_clear_chunk_cache=post_clear_chunk_cache,
                        release_chunk_cache=release_chunk_cache,
                        post_execute_func=post_execute_func,
                        post_execute_kwargs=post_execute_kwargs,
                        template_context=template_context,
                    )
                return self.merge_results(
                    results,
                    keys=keys,
                    filter_results=filter_results,
                    raise_no_results=raise_no_results,
                    merge_func=merge_func,
                    merge_kwargs=merge_kwargs,
                    template_context=template_context,
                )
        else:
            raise ValueError(f"Invalid option distribute='{self.distribute}'")


def execute(
    tasks: tp.TasksLike,
    size: tp.Optional[int] = None,
    keys: tp.Optional[tp.IndexLike] = None,
    executor_cls: tp.Optional[tp.Type[Executor]] = None,
    engine: tp.Optional[tp.ExecutionEngineLike] = None,
    engine_config: tp.KwargsLike = None,
    min_size: tp.Optional[int] = None,
    n_chunks: tp.Union[None, int, str] = None,
    chunk_len: tp.Union[None, int, str] = None,
    chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    distribute: tp.Optional[str] = None,
    warmup: tp.Optional[bool] = None,
    in_chunk_order: tp.Optional[bool] = None,
    cache_chunks: tp.Optional[bool] = None,
    chunk_cache_dir: tp.Optional[tp.PathLike] = None,
    chunk_cache_save_kwargs: tp.KwargsLike = None,
    chunk_cache_load_kwargs: tp.KwargsLike = None,
    pre_clear_chunk_cache: tp.Optional[bool] = None,
    post_clear_chunk_cache: tp.Optional[bool] = None,
    release_chunk_cache: tp.Optional[bool] = None,
    chunk_clear_cache: tp.Union[None, bool, int] = None,
    chunk_collect_garbage: tp.Union[None, bool, int] = None,
    chunk_delay: tp.Optional[float] = None,
    pre_execute_func: tp.Optional[tp.Callable] = None,
    pre_execute_kwargs: tp.KwargsLike = None,
    pre_chunk_func: tp.Optional[tp.Callable] = None,
    pre_chunk_kwargs: tp.KwargsLike = None,
    post_chunk_func: tp.Optional[tp.Callable] = None,
    post_chunk_kwargs: tp.KwargsLike = None,
    post_execute_func: tp.Optional[tp.Callable] = None,
    post_execute_kwargs: tp.KwargsLike = None,
    post_execute_on_sorted: tp.Optional[bool] = None,
    filter_results: tp.Optional[bool] = None,
    raise_no_results: tp.Optional[bool] = None,
    merge_func: tp.Optional[tp.MergeFuncLike] = None,
    merge_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
    show_progress: tp.Optional[bool] = None,
    pbar_kwargs: tp.KwargsLike = None,
    merge_to_engine_config: tp.Optional[bool] = None,
    **kwargs,
) -> tp.MergeableResults:
    """Execute functions and their arguments using `Executor`.

    Keyword arguments `**kwargs` and `engine_config` are merged into `engine_config`
    if `merge_to_engine_config` is True, otherwise, `**kwargs` are passed directly to `Executor`."""
    from vectorbtpro._settings import settings

    execution_cfg = settings["execution"]

    if executor_cls is None:
        executor_cls = execution_cfg["executor_cls"]
    if executor_cls is None:
        executor_cls = Executor
    if merge_to_engine_config is None:
        merge_to_engine_config = execution_cfg["merge_to_engine_config"]
    if merge_to_engine_config:
        engine_config = merge_dicts(kwargs, engine_config)
        kwargs = {}

    return executor_cls(
        engine=engine,
        engine_config=engine_config,
        min_size=min_size,
        n_chunks=n_chunks,
        chunk_len=chunk_len,
        chunk_meta=chunk_meta,
        distribute=distribute,
        warmup=warmup,
        in_chunk_order=in_chunk_order,
        cache_chunks=cache_chunks,
        chunk_cache_dir=chunk_cache_dir,
        chunk_cache_save_kwargs=chunk_cache_save_kwargs,
        chunk_cache_load_kwargs=chunk_cache_load_kwargs,
        pre_clear_chunk_cache=pre_clear_chunk_cache,
        post_clear_chunk_cache=post_clear_chunk_cache,
        release_chunk_cache=release_chunk_cache,
        chunk_clear_cache=chunk_clear_cache,
        chunk_collect_garbage=chunk_collect_garbage,
        chunk_delay=chunk_delay,
        pre_execute_func=pre_execute_func,
        pre_execute_kwargs=pre_execute_kwargs,
        pre_chunk_func=pre_chunk_func,
        pre_chunk_kwargs=pre_chunk_kwargs,
        post_chunk_func=post_chunk_func,
        post_chunk_kwargs=post_chunk_kwargs,
        post_execute_func=post_execute_func,
        post_execute_kwargs=post_execute_kwargs,
        post_execute_on_sorted=post_execute_on_sorted,
        filter_results=filter_results,
        raise_no_results=raise_no_results,
        merge_func=merge_func,
        merge_kwargs=merge_kwargs,
        template_context=template_context,
        show_progress=show_progress,
        pbar_kwargs=pbar_kwargs,
        **kwargs,
    ).run(tasks, size=size, keys=keys)


def parse_iterable_and_keys(
    iterable_like: tp.Union[int, tp.Iterable],
    keys: tp.Optional[tp.IndexLike] = None,
) -> tp.Tuple[tp.Iterable, tp.Optional[tp.Index]]:
    """Parse the iterable and the keys from an iterable-like and a keys-like object respectively.

    Object can be an integer that will be interpreted as a total or any iterable.

    If object is a dictionary, a Pandas Index, or a Pandas Series, keys will be set to the index.
    Otherwise, keys will be extracted using `vectorbtpro.base.indexes.index_from_values`.
    Keys won't be extracted if the object is not a sequence to avoid materializing it."""
    if keys is not None:
        from vectorbtpro.base.indexes import to_any_index

        keys = to_any_index(keys)
    if checks.is_int(iterable_like):
        iterable = range(iterable_like)
        if keys is None:
            keys = pd.Index(iterable)
    elif isinstance(iterable_like, dict):
        iterable = iterable_like.values()
        if keys is None:
            keys = pd.Index(list(iterable_like.keys()))
    elif isinstance(iterable_like, pd.Index):
        iterable = iterable_like
        if keys is None:
            keys = iterable_like
    elif isinstance(iterable_like, pd.Series):
        iterable = iterable_like.values
        if keys is None:
            keys = iterable_like.index
    elif checks.is_sequence(iterable_like):
        from vectorbtpro.base.indexes import index_from_values

        iterable = list(iterable_like)
        if keys is None:
            keys = index_from_values(iterable_like)
    else:
        iterable = iterable_like
    return iterable, keys


def iterated(
    *args,
    over_arg: tp.Optional[tp.AnnArgQuery] = None,
    executor_cls: tp.Optional[tp.Type[Executor]] = None,
    engine: tp.Optional[tp.ExecutionEngineLike] = None,
    engine_config: tp.KwargsLike = None,
    min_size: tp.Optional[int] = None,
    n_chunks: tp.Union[None, int, str] = None,
    chunk_len: tp.Union[None, int, str] = None,
    chunk_meta: tp.Optional[tp.Iterable[tp.ChunkMeta]] = None,
    distribute: tp.Optional[str] = None,
    warmup: tp.Optional[bool] = None,
    in_chunk_order: tp.Optional[bool] = None,
    cache_chunks: tp.Optional[bool] = None,
    chunk_cache_dir: tp.Optional[tp.PathLike] = None,
    chunk_cache_save_kwargs: tp.KwargsLike = None,
    chunk_cache_load_kwargs: tp.KwargsLike = None,
    pre_clear_chunk_cache: tp.Optional[bool] = None,
    post_clear_chunk_cache: tp.Optional[bool] = None,
    release_chunk_cache: tp.Optional[bool] = None,
    chunk_clear_cache: tp.Union[None, bool, int] = None,
    chunk_collect_garbage: tp.Union[None, bool, int] = None,
    chunk_delay: tp.Optional[float] = None,
    pre_execute_func: tp.Optional[tp.Callable] = None,
    pre_execute_kwargs: tp.KwargsLike = None,
    pre_chunk_func: tp.Optional[tp.Callable] = None,
    pre_chunk_kwargs: tp.KwargsLike = None,
    post_chunk_func: tp.Optional[tp.Callable] = None,
    post_chunk_kwargs: tp.KwargsLike = None,
    post_execute_func: tp.Optional[tp.Callable] = None,
    post_execute_kwargs: tp.KwargsLike = None,
    post_execute_on_sorted: tp.Optional[bool] = None,
    filter_results: tp.Optional[bool] = None,
    raise_no_results: tp.Optional[bool] = None,
    merge_func: tp.Optional[tp.MergeFuncLike] = None,
    merge_kwargs: tp.KwargsLike = None,
    template_context: tp.KwargsLike = None,
    show_progress: tp.Optional[bool] = None,
    pbar_kwargs: tp.KwargsLike = None,
    merge_to_engine_config: tp.Optional[bool] = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that executes a function in iteration using `Executor`.

    Returns a new function with the same signature as the passed one.

    Use `over_arg` to specify which argument (position or name) should be iterated over.
    If it's None (default), uses the first argument.

    Each option can be modified in the `options` attribute of the wrapper function or
    directly passed as a keyword argument with a leading underscore. You can also explicitly specify
    keys and size by passing them as `_keys` and `_size` respectively if the range-like object is an iterator.

    Keyword arguments `**kwargs` and `engine_config` are merged into `engine_config`
    if `merge_to_engine_config` is True, otherwise, `**kwargs` are passed directly to `Executor`.

    If `NoResult` is returned, will skip the current iteration and remove it from the final index."""

    def decorator(func: tp.Callable) -> tp.Callable:
        from vectorbtpro._settings import settings

        execution_cfg = settings["execution"]

        if merge_to_engine_config is None:
            _merge_to_engine_config = execution_cfg["merge_to_engine_config"]
        else:
            _merge_to_engine_config = merge_to_engine_config
        if _merge_to_engine_config:
            _engine_config = merge_dicts(kwargs, engine_config)
            _executor_kwargs = {}
        else:
            _engine_config = engine_config
            _executor_kwargs = kwargs

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            def _resolve_key(key, merge=False):
                if "_" + key in kwargs:
                    if merge:
                        return merge_dicts(wrapper.options[key], kwargs.pop("_" + key))
                    return kwargs.pop("_" + key)
                return wrapper.options.get(key)

            executor_cls = wrapper.options["executor_cls"]
            if executor_cls is None:
                executor_cls = execution_cfg["executor_cls"]
            if executor_cls is None:
                executor_cls = Executor

            over_arg = _resolve_key("over_arg")
            if over_arg is None:
                iterable_like = args[0]
            else:
                ann_args = annotate_args(func, args, kwargs)
                iterable_like = match_ann_arg(ann_args, over_arg)
            size = kwargs.pop("_size", None)
            keys = kwargs.pop("_keys", None)
            iterable, keys = parse_iterable_and_keys(iterable_like, keys=keys)
            if keys is not None and size is None:
                size = len(keys)
            if keys is not None and keys.name is None:
                if over_arg is None:
                    keys = keys.rename(get_func_arg_names(func)[0])
                else:
                    ann_args = annotate_args(func, args, kwargs)
                    over_arg_name = match_ann_arg(ann_args, over_arg, return_name=True)
                    keys = keys.rename(over_arg_name)

            if over_arg is None:

                def _get_task_generator():
                    for x in iterable:
                        yield Task(func, x, *args[1:], **kwargs)

            else:

                def _get_task_generator():
                    for x in iterable:
                        flat_ann_args = annotate_args(func, args, kwargs, flatten=True)
                        match_and_set_flat_ann_arg(flat_ann_args, over_arg, x)
                        new_args, new_kwargs = flat_ann_args_to_args(flat_ann_args)
                        yield Task(func, *new_args, **new_kwargs)

            tasks = _get_task_generator()

            return executor_cls(
                engine=_resolve_key("engine"),
                engine_config=_resolve_key("engine_config", merge=True),
                min_size=_resolve_key("min_size"),
                n_chunks=_resolve_key("n_chunks"),
                chunk_len=_resolve_key("chunk_len"),
                chunk_meta=_resolve_key("chunk_meta"),
                distribute=_resolve_key("distribute"),
                warmup=_resolve_key("warmup"),
                in_chunk_order=_resolve_key("in_chunk_order"),
                cache_chunks=_resolve_key("cache_chunks"),
                chunk_cache_dir=_resolve_key("chunk_cache_dir"),
                chunk_cache_save_kwargs=_resolve_key("chunk_cache_save_kwargs", merge=True),
                chunk_cache_load_kwargs=_resolve_key("chunk_cache_load_kwargs", merge=True),
                pre_clear_chunk_cache=_resolve_key("pre_clear_chunk_cache"),
                post_clear_chunk_cache=_resolve_key("post_clear_chunk_cache"),
                release_chunk_cache=_resolve_key("release_chunk_cache"),
                chunk_clear_cache=_resolve_key("chunk_clear_cache"),
                chunk_collect_garbage=_resolve_key("chunk_collect_garbage"),
                chunk_delay=_resolve_key("chunk_delay"),
                pre_execute_func=_resolve_key("pre_execute_func"),
                pre_execute_kwargs=_resolve_key("pre_execute_kwargs", merge=True),
                pre_chunk_func=_resolve_key("pre_chunk_func"),
                pre_chunk_kwargs=_resolve_key("pre_chunk_kwargs", merge=True),
                post_chunk_func=_resolve_key("post_chunk_func"),
                post_chunk_kwargs=_resolve_key("post_chunk_kwargs", merge=True),
                post_execute_func=_resolve_key("post_execute_func"),
                post_execute_kwargs=_resolve_key("post_execute_kwargs", merge=True),
                post_execute_on_sorted=_resolve_key("post_execute_on_sorted"),
                filter_results=_resolve_key("filter_results"),
                raise_no_results=_resolve_key("raise_no_results"),
                merge_func=_resolve_key("merge_func"),
                merge_kwargs=_resolve_key("merge_kwargs", merge=True),
                template_context=_resolve_key("template_context", merge=True),
                show_progress=_resolve_key("show_progress"),
                pbar_kwargs=_resolve_key("pbar_kwargs", merge=True),
                **_resolve_key("executor_kwargs", merge=True),
            ).run(tasks, size=size, keys=keys)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_executor = True
        wrapper.options = FrozenConfig(
            over_arg=over_arg,
            executor_cls=executor_cls,
            executor_kwargs=_executor_kwargs,
            engine=engine,
            engine_config=_engine_config,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            distribute=distribute,
            warmup=warmup,
            in_chunk_order=in_chunk_order,
            cache_chunks=cache_chunks,
            chunk_cache_dir=chunk_cache_dir,
            chunk_cache_save_kwargs=chunk_cache_save_kwargs,
            chunk_cache_load_kwargs=chunk_cache_load_kwargs,
            pre_clear_chunk_cache=pre_clear_chunk_cache,
            post_clear_chunk_cache=post_clear_chunk_cache,
            release_chunk_cache=release_chunk_cache,
            chunk_clear_cache=chunk_clear_cache,
            chunk_collect_garbage=chunk_collect_garbage,
            chunk_delay=chunk_delay,
            pre_execute_func=pre_execute_func,
            pre_execute_kwargs=pre_execute_kwargs,
            pre_chunk_func=pre_chunk_func,
            pre_chunk_kwargs=pre_chunk_kwargs,
            post_chunk_func=post_chunk_func,
            post_chunk_kwargs=post_chunk_kwargs,
            post_execute_func=post_execute_func,
            post_execute_kwargs=post_execute_kwargs,
            post_execute_on_sorted=post_execute_on_sorted,
            filter_results=filter_results,
            raise_no_results=raise_no_results,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            template_context=template_context,
            show_progress=show_progress,
            pbar_kwargs=pbar_kwargs,
        )
        signature = inspect.signature(wrapper)
        lists_var_kwargs = False
        for k, v in signature.parameters.items():
            if v.kind == v.VAR_KEYWORD:
                lists_var_kwargs = True
                break
        if not lists_var_kwargs:
            var_kwargs_param = inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD)
            new_parameters = tuple(signature.parameters.values()) + (var_kwargs_param,)
            wrapper.__signature__ = signature.replace(parameters=new_parameters)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")
