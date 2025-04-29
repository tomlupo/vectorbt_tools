# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Mixin class for working with simulation ranges."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.indexing import AutoIdxr
from vectorbtpro.base.reshaping import broadcast_array_to
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.generic import nb
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.decorators import class_or_instancemethod

SimRangeMixinT = tp.TypeVar("SimRangeMixinT", bound="SimRangeMixin")


class SimRangeMixin:
    """Mixin class for working with simulation ranges.

    Should be subclassed by a subclass of `vectorbtpro.base.wrapping.Wrapping`."""

    @classmethod
    def row_stack_sim_start(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeTuple[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Row-stack simulation start."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        if objs[0]._sim_start is not None:
            new_sim_start = broadcast_array_to(objs[0]._sim_start, len(new_wrapper.columns))
        else:
            new_sim_start = None
        for obj in objs[1:]:
            if obj._sim_start is not None:
                raise ValueError("Objects to be merged (except the first one) must have 'sim_start=None'")
        return new_sim_start

    @classmethod
    def row_stack_sim_end(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeTuple[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Row-stack simulation end."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        if objs[-1]._sim_end is not None:
            new_sim_end = len(new_wrapper.index) - len(objs[-1].wrapper.index) + objs[-1]._sim_end
            new_sim_end = broadcast_array_to(new_sim_end, len(new_wrapper.columns))
        else:
            new_sim_end = None
        for obj in objs[:-1]:
            if obj._sim_end is not None:
                raise ValueError("Objects to be merged (except the last one) must have 'sim_end=None'")
        return new_sim_end

    @classmethod
    def column_stack_sim_start(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeTuple[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Column-stack simulation start."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        stack_sim_start_objs = False
        for obj in objs:
            if obj._sim_start is not None:
                stack_sim_start_objs = True
                break
        if stack_sim_start_objs:
            obj_sim_starts = []
            for obj in objs:
                obj_sim_start = np.empty(len(obj._sim_start), dtype=np.int_)
                for i in range(len(obj._sim_start)):
                    if obj._sim_start[i] == 0:
                        obj_sim_start[i] = 0
                    elif obj._sim_start[i] == len(obj.wrapper.index):
                        obj_sim_start[i] = len(new_wrapper.index)
                    else:
                        _obj_sim_start = new_wrapper.index.get_indexer([obj.wrapper.index[obj._sim_start[i]]])[0]
                        if _obj_sim_start == -1:
                            _obj_sim_start = 0
                        obj_sim_start[i] = _obj_sim_start
                obj_sim_starts.append(obj_sim_start)
            new_sim_start = new_wrapper.concat_arrs(*obj_sim_starts, wrap=False)
        else:
            new_sim_start = None
        return new_sim_start

    @classmethod
    def column_stack_sim_end(
        cls,
        new_wrapper: ArrayWrapper,
        *objs: tp.MaybeTuple[SimRangeMixinT],
    ) -> tp.Optional[tp.ArrayLike]:
        """Column-stack simulation end."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)

        stack_sim_end_objs = False
        for obj in objs:
            if obj._sim_end is not None:
                stack_sim_end_objs = True
                break
        if stack_sim_end_objs:
            obj_sim_ends = []
            for obj in objs:
                obj_sim_end = np.empty(len(obj._sim_end), dtype=np.int_)
                for i in range(len(obj._sim_end)):
                    if obj._sim_end[i] == 0:
                        obj_sim_end[i] = 0
                    elif obj._sim_end[i] == len(obj.wrapper.index):
                        obj_sim_end[i] = len(new_wrapper.index)
                    else:
                        _obj_sim_end = new_wrapper.index.get_indexer([obj.wrapper.index[obj._sim_end[i]]])[0]
                        if _obj_sim_end == -1:
                            _obj_sim_end = 0
                        obj_sim_end[i] = _obj_sim_end
                obj_sim_ends.append(obj_sim_end)
            new_sim_end = new_wrapper.concat_arrs(*obj_sim_ends, wrap=False)
        else:
            new_sim_end = None
        return new_sim_end

    def __init__(
        self,
        sim_start: tp.Optional[tp.Array1d] = None,
        sim_end: tp.Optional[tp.Array1d] = None,
    ) -> None:
        sim_start = type(self).resolve_sim_start(sim_start=sim_start, wrapper=self.wrapper, group_by=False)
        sim_end = type(self).resolve_sim_end(sim_end=sim_end, wrapper=self.wrapper, group_by=False)

        self._sim_start = sim_start
        self._sim_end = sim_end

    def sim_start_indexing_func(self, wrapper_meta: dict) -> tp.Optional[tp.ArrayLike]:
        """Indexing function for simulation start."""
        if self._sim_start is None:
            new_sim_start = None
        elif not wrapper_meta["rows_changed"]:
            new_sim_start = self._sim_start
        else:
            if checks.is_int(wrapper_meta["row_idxs"]):
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"]
            elif isinstance(wrapper_meta["row_idxs"], slice):
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"].start
            else:
                new_sim_start = self._sim_start - wrapper_meta["row_idxs"][0]
            new_sim_start = np.clip(new_sim_start, 0, len(wrapper_meta["new_wrapper"].index))
        return new_sim_start

    def sim_end_indexing_func(self, wrapper_meta: dict) -> tp.Optional[tp.ArrayLike]:
        """Indexing function for simulation end."""
        if self._sim_end is None:
            new_sim_end = None
        elif not wrapper_meta["rows_changed"]:
            new_sim_end = self._sim_end
        else:
            if checks.is_int(wrapper_meta["row_idxs"]):
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"]
            elif isinstance(wrapper_meta["row_idxs"], slice):
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"].start
            else:
                new_sim_end = self._sim_end - wrapper_meta["row_idxs"][0]
            new_sim_end = np.clip(new_sim_end, 0, len(wrapper_meta["new_wrapper"].index))
        return new_sim_end

    def resample_sim_start(self, new_wrapper: ArrayWrapper) -> tp.Optional[tp.ArrayLike]:
        """Resample simulation start."""
        if self._sim_start is not None:
            new_sim_start = np.empty(len(self._sim_start), dtype=np.int_)
            for i in range(len(self._sim_start)):
                if self._sim_start[i] == 0:
                    new_sim_start[i] = 0
                elif self._sim_start[i] == len(self.wrapper.index):
                    new_sim_start[i] = len(new_wrapper.index)
                else:
                    _new_sim_start = new_wrapper.index.get_indexer(
                        [self.wrapper.index[self._sim_start[i]]],
                        method="ffill",
                    )[0]
                    if _new_sim_start == -1:
                        _new_sim_start = 0
                    new_sim_start[i] = _new_sim_start
        else:
            new_sim_start = None
        return new_sim_start

    def resample_sim_end(self, new_wrapper: ArrayWrapper) -> tp.Optional[tp.ArrayLike]:
        """Resample simulation end."""
        if self._sim_end is not None:
            new_sim_end = np.empty(len(self._sim_end), dtype=np.int_)
            for i in range(len(self._sim_end)):
                if self._sim_end[i] == 0:
                    new_sim_end[i] = 0
                elif self._sim_end[i] == len(self.wrapper.index):
                    new_sim_end[i] = len(new_wrapper.index)
                else:
                    _new_sim_end = new_wrapper.index.get_indexer(
                        [self.wrapper.index[self._sim_end[i]]],
                        method="bfill",
                    )[0]
                    if _new_sim_end == -1:
                        _new_sim_end = len(new_wrapper.index)
                    new_sim_end[i] = _new_sim_end
        else:
            new_sim_end = None
        return new_sim_end

    @class_or_instancemethod
    def resolve_sim_start_value(
        cls_or_self,
        value: tp.Scalar,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> int:
        """Resolve a single value of simulation start."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        auto_idxr = AutoIdxr(value, indexer_method="bfill", below_to_zero=True)
        return auto_idxr.get(wrapper.index, freq=wrapper.freq)

    @class_or_instancemethod
    def resolve_sim_end_value(
        cls_or_self,
        value: tp.Scalar,
        wrapper: tp.Optional[ArrayWrapper] = None,
    ) -> int:
        """Resolve a single value of simulation end."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        auto_idxr = AutoIdxr(value, indexer_method="bfill", above_to_len=True)
        return auto_idxr.get(wrapper.index, freq=wrapper.freq)

    @class_or_instancemethod
    def resolve_sim_start(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Optional[tp.ArrayLike]:
        """Resolve simulation start."""
        already_resolved = False
        if not isinstance(cls_or_self, type):
            if sim_start is None:
                sim_start = cls_or_self._sim_start
                already_resolved = True
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if sim_start is False:
            sim_start = None
        if allow_none and sim_start is None:
            return None
        if not already_resolved and sim_start is not None:
            sim_start_arr = np.asarray(sim_start)
            if not np.issubdtype(sim_start_arr.dtype, np.integer):
                if sim_start_arr.ndim == 0:
                    sim_start = cls_or_self.resolve_sim_start_value(sim_start, wrapper=wrapper)
                else:
                    new_sim_start = np.empty(len(sim_start), dtype=np.int_)
                    for i in range(len(sim_start)):
                        new_sim_start[i] = cls_or_self.resolve_sim_start_value(sim_start[i], wrapper=wrapper)
                    sim_start = new_sim_start
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            sim_start = nb.resolve_grouped_sim_start_nb(
                wrapper.shape_2d,
                group_lens,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        elif not already_resolved and wrapper.grouper.is_grouped():
            group_lens = wrapper.grouper.get_group_lens()
            sim_start = nb.resolve_ungrouped_sim_start_nb(
                wrapper.shape_2d,
                group_lens,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        else:
            sim_start = nb.resolve_sim_start_nb(
                wrapper.shape_2d,
                sim_start=sim_start,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        return sim_start

    @class_or_instancemethod
    def resolve_sim_end(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
    ) -> tp.Optional[tp.ArrayLike]:
        """Resolve simulation end."""
        already_resolved = False
        if not isinstance(cls_or_self, type):
            if sim_end is None:
                sim_end = cls_or_self._sim_end
                already_resolved = True
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        if sim_end is False:
            sim_end = None
        if allow_none and sim_end is None:
            return None
        if not already_resolved and sim_end is not None:
            sim_end_arr = np.asarray(sim_end)
            if not np.issubdtype(sim_end_arr.dtype, np.integer):
                if sim_end_arr.ndim == 0:
                    sim_end = cls_or_self.resolve_sim_end_value(sim_end, wrapper=wrapper)
                else:
                    new_sim_end = np.empty(len(sim_end), dtype=np.int_)
                    for i in range(len(sim_end)):
                        new_sim_end[i] = cls_or_self.resolve_sim_end_value(sim_end[i], wrapper=wrapper)
                    sim_end = new_sim_end
        if wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
            sim_end = nb.resolve_grouped_sim_end_nb(
                wrapper.shape_2d,
                group_lens,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        elif not already_resolved and wrapper.grouper.is_grouped():
            group_lens = wrapper.grouper.get_group_lens()
            sim_end = nb.resolve_ungrouped_sim_end_nb(
                wrapper.shape_2d,
                group_lens,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        else:
            sim_end = nb.resolve_sim_end_nb(
                wrapper.shape_2d,
                sim_end=sim_end,
                allow_none=allow_none,
                check_bounds=not already_resolved,
            )
        return sim_end

    @class_or_instancemethod
    def get_sim_start(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, tp.Array1d, tp.Series]:
        """Get simulation start."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is None:
            return None
        if keep_flex:
            return sim_start
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_end"), wrap_kwargs)
        return wrapper.wrap_reduced(sim_start, group_by=group_by, **wrap_kwargs)

    @property
    def sim_start(self) -> tp.Series:
        """`SimRangeMixin.get_sim_start` with default arguments."""
        return self.get_sim_start()

    @class_or_instancemethod
    def get_sim_end(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        keep_flex: bool = False,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Union[None, tp.Array1d, tp.Series]:
        """Get simulation end."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_end is None:
            return None
        if keep_flex:
            return sim_end
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_start"), wrap_kwargs)
        return wrapper.wrap_reduced(sim_end, group_by=group_by, **wrap_kwargs)

    @property
    def sim_end(self) -> tp.Series:
        """`SimRangeMixin.get_sim_end` with default arguments."""
        return self.get_sim_end()

    @class_or_instancemethod
    def get_sim_start_index(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = False,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Get index of simulation start."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is None:
            return None
        start_index = []
        for i in range(len(sim_start)):
            _sim_start = sim_start[i]
            if _sim_start == 0:
                start_index.append(wrapper.index[0])
            elif _sim_start == len(wrapper.index):
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    start_index.append(wrapper.index[-1] + wrapper.freq)
                elif isinstance(wrapper.index, pd.RangeIndex):
                    start_index.append(wrapper.index[-1] + 1)
                else:
                    start_index.append(None)
            else:
                start_index.append(wrapper.index[_sim_start])
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_start_index"), wrap_kwargs)
        return wrapper.wrap_reduced(pd.Index(start_index), group_by=group_by, **wrap_kwargs)

    @property
    def sim_start_index(self) -> tp.Series:
        """`SimRangeMixin.get_sim_start_index` with default arguments."""
        return self.get_sim_start_index()

    @class_or_instancemethod
    def get_sim_end_index(
        cls_or_self,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        allow_none: bool = False,
        inclusive: bool = True,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Get index of simulation end."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=allow_none,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_end is None:
            return None
        end_index = []
        for i in range(len(sim_end)):
            _sim_end = sim_end[i]
            if _sim_end == 0:
                if inclusive:
                    end_index.append(None)
                else:
                    end_index.append(wrapper.index[0])
            elif _sim_end == len(wrapper.index):
                if inclusive:
                    end_index.append(wrapper.index[-1])
                else:
                    if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                        end_index.append(wrapper.index[-1] + wrapper.freq)
                    elif isinstance(wrapper.index, pd.RangeIndex):
                        end_index.append(wrapper.index[-1] + 1)
                    else:
                        end_index.append(None)
            else:
                if inclusive:
                    end_index.append(wrapper.index[_sim_end - 1])
                else:
                    end_index.append(wrapper.index[_sim_end])
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_end_index"), wrap_kwargs)
        return wrapper.wrap_reduced(pd.Index(end_index), group_by=group_by, **wrap_kwargs)

    @property
    def sim_end_index(self) -> tp.Series:
        """`SimRangeMixin.get_sim_end_index` with default arguments."""
        return self.get_sim_end_index()

    @class_or_instancemethod
    def get_sim_duration(
        cls_or_self,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        wrap_kwargs: tp.KwargsLike = None,
    ) -> tp.Optional[tp.Series]:
        """Get duration of simulation range."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.resolve_sim_start(
            sim_start=sim_start,
            allow_none=False,
            wrapper=wrapper,
            group_by=group_by,
        )
        sim_end = cls_or_self.resolve_sim_end(
            sim_end=sim_end,
            allow_none=False,
            wrapper=wrapper,
            group_by=group_by,
        )
        total_duration = sim_end - sim_start
        wrap_kwargs = merge_dicts(dict(name_or_index="sim_duration"), wrap_kwargs)
        return wrapper.wrap_reduced(total_duration, group_by=group_by, **wrap_kwargs)

    @property
    def sim_duration(self) -> tp.Series:
        """`SimRangeMixin.get_sim_duration` with default arguments."""
        return self.get_sim_duration()

    @class_or_instancemethod
    def fit_fig_to_sim_range(
        cls_or_self,
        fig: tp.BaseFigure,
        column: tp.Optional[tp.Label] = None,
        sim_start: tp.Optional[tp.ArrayLike] = None,
        sim_end: tp.Optional[tp.ArrayLike] = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        group_by: tp.GroupByLike = None,
        xref: tp.Optional[str] = None,
    ) -> tp.BaseFigure:
        """Fit figure to simulation range."""
        if not isinstance(cls_or_self, type):
            if wrapper is None:
                wrapper = cls_or_self.wrapper
        else:
            checks.assert_not_none(wrapper, arg_name="wrapper")

        sim_start = cls_or_self.get_sim_start(
            sim_start=sim_start,
            allow_none=True,
            wrapper=wrapper,
            group_by=group_by,
        )
        sim_end = cls_or_self.get_sim_end(
            sim_end=sim_end,
            allow_none=True,
            wrapper=wrapper,
            group_by=group_by,
        )
        if sim_start is not None:
            sim_start = wrapper.select_col_from_obj(sim_start, column=column, group_by=group_by)
        if sim_end is not None:
            sim_end = wrapper.select_col_from_obj(sim_end, column=column, group_by=group_by)
        if sim_start is not None or sim_end is not None:
            if sim_start == len(wrapper.index) or sim_end == 0 or sim_start == sim_end:
                return fig
            if sim_start is None:
                sim_start = 0
            if sim_start > 0:
                sim_start_index = wrapper.index[sim_start - 1]
            else:
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    sim_start_index = wrapper.index[0] - wrapper.freq
                elif isinstance(wrapper.index, pd.RangeIndex):
                    sim_start_index = wrapper.index[0] - 1
                else:
                    sim_start_index = wrapper.index[0]
            if sim_end is None:
                sim_end = len(wrapper.index)
            if sim_end < len(wrapper.index):
                sim_end_index = wrapper.index[sim_end]
            else:
                if isinstance(wrapper.index, pd.DatetimeIndex) and wrapper.freq is not None:
                    sim_end_index = wrapper.index[-1] + wrapper.freq
                elif isinstance(wrapper.index, pd.RangeIndex):
                    sim_end_index = wrapper.index[-1] + 1
                else:
                    sim_end_index = wrapper.index[-1]
            if xref is not None:
                xaxis = "xaxis" + xref[1:]
                fig.update_layout(**{xaxis: dict(range=[sim_start_index, sim_end_index])})
            else:
                fig.update_xaxes(range=[sim_start_index, sim_end_index])
        return fig
