# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for label generation.

!!! note
    Set `wait` to 1 to exclude the current value from calculation of future values.

!!! warning
    Do not attempt to use these functions for building predictor variables as they may introduce
    the look-ahead bias to your model - only use for building target variables."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_nb, flex_select_col_nb
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.indicators.enums import Pivot
from vectorbtpro.labels.enums import TrendLabelMode
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils import chunking as ch

__all__ = []


# ############# FMEAN ############# #


@register_jitted(cache=True)
def future_mean_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    wait: int = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Rolling average over future values."""
    future_mean = generic_nb.ma_1d_nb(close[::-1], window, wtype=wtype, minp=minp, adjust=adjust)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_mean, wait)
    return future_mean


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_mean_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """2-dim version of `future_mean_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_mean = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        future_mean[:, col] = future_mean_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
        )
    return future_mean


# ############# FSTD ############# #


@register_jitted(cache=True)
def future_std_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wtype: int = generic_enums.WType.Simple,
    wait: int = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array1d:
    """Rolling standard deviation over future values."""
    future_std = generic_nb.msd_1d_nb(close[::-1], window, wtype=wtype, minp=minp, adjust=adjust, ddof=ddof)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_std, wait)
    return future_std


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
        ddof=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_std_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
    ddof: int = 0,
) -> tp.Array2d:
    """2-dim version of `future_std_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_std = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        future_std[:, col] = future_std_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
            ddof=ddof,
        )
    return future_std


# ############# FMIN ############# #


@register_jitted(cache=True)
def future_min_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wait: int = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Rolling minimum over future values."""
    future_min = generic_nb.rolling_min_1d_nb(close[::-1], window, minp=minp)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_min, wait)
    return future_min


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_min_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """2-dim version of `future_min_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_min = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        future_min[:, col] = future_min_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
        )
    return future_min


# ############# FMAX ############# #


@register_jitted(cache=True)
def future_max_1d_nb(
    close: tp.Array1d,
    window: int = 14,
    wait: int = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array1d:
    """Rolling maximum over future values."""
    future_max = generic_nb.rolling_max_1d_nb(close[::-1], window, minp=minp)[::-1]
    if wait > 0:
        return generic_nb.bshift_1d_nb(future_max, wait)
    return future_max


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def future_max_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
) -> tp.Array2d:
    """2-dim version of `future_max_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    wait_ = to_1d_array_nb(np.asarray(wait))

    future_max = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        future_max[:, col] = future_max_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
        )
    return future_max


# ############# FIXLB ############# #


@register_jitted(cache=True)
def fixed_labels_1d_nb(
    close: tp.Array1d,
    n: int = 1,
) -> tp.Array1d:
    """Percentage change of the current value relative to a future value."""
    return (generic_nb.bshift_1d_nb(close, n) - close) / close


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        n=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def fixed_labels_nb(
    close: tp.Array2d,
    n: tp.FlexArray1dLike = 1,
) -> tp.Array2d:
    """2-dim version of `fixed_labels_1d_nb`."""
    n_ = to_1d_array_nb(np.asarray(n))

    fixed_labels = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        fixed_labels[:, col] = fixed_labels_1d_nb(
            close=close[:, col],
            n=flex_select_1d_nb(n_, col),
        )
    return fixed_labels


# ############# MEANLB ############# #


@register_jitted(cache=True)
def mean_labels_1d_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array1d:
    """Percentage change of the current value relative to the average of a future period."""
    future_mean = future_mean_1d_nb(close, window=window, wtype=wtype, wait=wait, minp=minp, adjust=adjust)
    return (future_mean - close) / close


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        wtype=base_ch.FlexArraySlicer(),
        wait=base_ch.FlexArraySlicer(),
        minp=None,
        adjust=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def mean_labels_nb(
    close: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    wtype: tp.FlexArray1dLike = generic_enums.WType.Simple,
    wait: tp.FlexArray1dLike = 1,
    minp: tp.Optional[int] = None,
    adjust: bool = False,
) -> tp.Array2d:
    """2-dim version of `mean_labels_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    wtype_ = to_1d_array_nb(np.asarray(wtype))
    wait_ = to_1d_array_nb(np.asarray(wait))

    mean_labels = np.empty(close.shape, dtype=np.float_)
    for col in prange(close.shape[1]):
        mean_labels[:, col] = mean_labels_1d_nb(
            close=close[:, col],
            window=flex_select_1d_nb(window_, col),
            wtype=flex_select_1d_nb(wtype_, col),
            wait=flex_select_1d_nb(wait_, col),
            minp=minp,
            adjust=adjust,
        )
    return mean_labels


# ############# PIVOTLB ############# #


@register_jitted(cache=True)
def iter_symmetric_up_th_nb(down_th: float) -> float:
    """Positive upper threshold that is symmetric to a negative one at one iteration.

    For example, 50% down requires 100% to go up to the initial level."""
    return down_th / (1 - down_th)


@register_jitted(cache=True)
def iter_symmetric_down_th_nb(up_th: float) -> float:
    """Negative upper threshold that is symmetric to a positive one at one iteration."""
    return up_th / (1 + up_th)


@register_jitted(cache=True)
def pivots_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
) -> tp.Array1d:
    """Pivots denoted by 1 (peak), 0 (no pivot) or -1 (valley).

    Two adjacent peak and valley points should exceed the given threshold parameters.

    If any threshold is given element-wise, it will be applied per new/updated pivot."""
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    pivots = np.full(high.shape, 0, dtype=np.int_)

    last_pivot = 0
    last_i = -1
    last_value = np.nan
    first_valid_i = -1
    for i in range(high.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            if first_valid_i == -1:
                first_valid_i = 0
            if last_i == -1:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, first_valid_i))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, first_valid_i))
                if not np.isnan(_up_th) and high[i] >= low[first_valid_i] * _up_th:
                    if not np.isnan(_down_th) and low[i] <= high[first_valid_i] * _down_th:
                        pass  # wait
                    else:
                        pivots[first_valid_i] = Pivot.Valley
                        last_i = i
                        last_value = high[i]
                        last_pivot = Pivot.Peak
                if not np.isnan(_down_th) and low[i] <= high[first_valid_i] * _down_th:
                    if not np.isnan(_up_th) and high[i] >= low[first_valid_i] * _up_th:
                        pass  # wait
                    else:
                        pivots[first_valid_i] = Pivot.Peak
                        last_i = i
                        last_value = low[i]
                        last_pivot = Pivot.Valley
            else:
                _up_th = 1 + abs(flex_select_1d_nb(up_th_, last_i))
                _down_th = 1 - abs(flex_select_1d_nb(down_th_, last_i))
                if last_pivot == Pivot.Valley:
                    if not np.isnan(last_value) and not np.isnan(_up_th) and high[i] >= last_value * _up_th:
                        pivots[last_i] = last_pivot
                        last_i = i
                        last_value = high[i]
                        last_pivot = Pivot.Peak
                    elif np.isnan(last_value) or low[i] < last_value:
                        last_i = i
                        last_value = low[i]
                elif last_pivot == Pivot.Peak:
                    if not np.isnan(last_value) and not np.isnan(_down_th) and low[i] <= last_value * _down_th:
                        pivots[last_i] = last_pivot
                        last_i = i
                        last_value = low[i]
                        last_pivot = Pivot.Valley
                    elif np.isnan(last_value) or high[i] > last_value:
                        last_i = i
                        last_value = high[i]

        if last_i != -1 and i == high.shape[0] - 1:
            pivots[last_i] = last_pivot

    return pivots


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pivots_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
) -> tp.Array2d:
    """2-dim version of `pivots_1d_nb`."""
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))

    pivots = np.empty(high.shape, dtype=np.int_)
    for col in prange(high.shape[1]):
        pivots[:, col] = pivots_1d_nb(
            high[:, col],
            low[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
        )
    return pivots


# ############# TRENDLB ############# #


@register_jitted(cache=True)
def bin_trend_labels_1d_nb(pivots: tp.Array1d) -> tp.Array1d:
    """Values classified into 0 (downtrend) and 1 (uptrend)."""
    bin_trend_labels = np.full(pivots.shape, np.nan, dtype=np.float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return bin_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]

        for i in range(prev_i, next_i):
            if pivots[next_i] == Pivot.Peak:
                bin_trend_labels[i] = 1
            else:
                bin_trend_labels[i] = 0

    return bin_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        pivots=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bin_trend_labels_nb(pivots: tp.Array2d) -> tp.Array2d:
    """2-dim version of `bin_trend_labels_1d_nb`."""
    bin_trend_labels = np.empty(pivots.shape, dtype=np.float_)
    for col in prange(pivots.shape[1]):
        bin_trend_labels[:, col] = bin_trend_labels_1d_nb(pivots[:, col])
    return bin_trend_labels


@register_jitted(cache=True)
def binc_trend_labels_1d_nb(high: tp.Array1d, low: tp.Array1d, pivots: tp.Array1d) -> tp.Array1d:
    """Median values normalized between 0 (downtrend) and 1 (uptrend)."""
    binc_trend_labels = np.full(pivots.shape, np.nan, dtype=np.float_)
    idxs = np.flatnonzero(pivots[:])
    if idxs.shape[0] == 0:
        return binc_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]
        _min = np.nanmin(low[prev_i : next_i + 1])
        _max = np.nanmax(high[prev_i : next_i + 1])

        for i in range(prev_i, next_i):
            _med = (high[i] + low[i]) / 2
            binc_trend_labels[i] = 1 - (_med - _min) / (_max - _min)

    return binc_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def binc_trend_labels_nb(high: tp.Array2d, low: tp.Array2d, pivots: tp.Array2d) -> tp.Array2d:
    """2-dim version of `binc_trend_labels_1d_nb`."""
    binc_trend_labels = np.empty(pivots.shape, dtype=np.float_)
    for col in prange(pivots.shape[1]):
        binc_trend_labels[:, col] = binc_trend_labels_1d_nb(high[:, col], low[:, col], pivots[:, col])
    return binc_trend_labels


@register_jitted(cache=True)
def bincs_trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    pivots: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
) -> tp.Array1d:
    """Median values normalized between 0 (downtrend) and 1 (uptrend) but capped once
    the threshold defined at the beginning of the trend is exceeded."""
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    bincs_trend_labels = np.full(pivots.shape, np.nan, dtype=np.float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return bincs_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]
        _up_th = 1 + abs(flex_select_1d_nb(up_th_, prev_i))
        _down_th = 1 - abs(flex_select_1d_nb(down_th_, prev_i))
        _min = np.min(low[prev_i : next_i + 1])
        _max = np.max(high[prev_i : next_i + 1])

        for i in range(prev_i, next_i):
            if not np.isnan(high[i]) and not np.isnan(low[i]):
                _med = (high[i] + low[i]) / 2
                if pivots[next_i] == Pivot.Peak:
                    if not np.isnan(_up_th):
                        _start = _max / _up_th
                        _end = _min * _up_th
                        if _max >= _end and _med <= _start:
                            bincs_trend_labels[i] = 1
                        else:
                            bincs_trend_labels[i] = 1 - (_med - _start) / (_max - _start)
                else:
                    if not np.isnan(_down_th):
                        _start = _min / _down_th
                        _end = _max * _down_th
                        if _min <= _end and _med >= _start:
                            bincs_trend_labels[i] = 0
                        else:
                            bincs_trend_labels[i] = 1 - (_med - _min) / (_start - _min)

    return bincs_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def bincs_trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    pivots: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
) -> tp.Array2d:
    """2-dim version of `bincs_trend_labels_1d_nb`."""
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))

    bincs_trend_labels = np.empty(pivots.shape, dtype=np.float_)
    for col in prange(pivots.shape[1]):
        bincs_trend_labels[:, col] = bincs_trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            pivots[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
        )
    return bincs_trend_labels


@register_jitted(cache=True)
def pct_trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    pivots: tp.Array1d,
    normalize: bool = False,
) -> tp.Array1d:
    """Percentage change of median values relative to the next pivot."""
    pct_trend_labels = np.full(pivots.shape, np.nan, dtype=np.float_)
    idxs = np.flatnonzero(pivots)
    if idxs.shape[0] == 0:
        return pct_trend_labels

    for k in range(1, idxs.shape[0]):
        prev_i = idxs[k - 1]
        next_i = idxs[k]

        for i in range(prev_i, next_i):
            _med = (high[i] + low[i]) / 2
            if pivots[next_i] == Pivot.Peak:
                if normalize:
                    pct_trend_labels[i] = (high[next_i] - _med) / high[next_i]
                else:
                    pct_trend_labels[i] = (high[next_i] - _med) / _med
            else:
                if normalize:
                    pct_trend_labels[i] = (low[next_i] - _med) / _med
                else:
                    pct_trend_labels[i] = (low[next_i] - _med) / low[next_i]

    return pct_trend_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="pivots", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        pivots=ch.ArraySlicer(axis=1),
        normalize=None,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def pct_trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    pivots: tp.Array2d,
    normalize: bool = False,
) -> tp.Array2d:
    """2-dim version of `pct_trend_labels_1d_nb`."""
    pct_trend_labels = np.empty(pivots.shape, dtype=np.float_)
    for col in prange(pivots.shape[1]):
        pct_trend_labels[:, col] = pct_trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            pivots[:, col],
            normalize=normalize,
        )
    return pct_trend_labels


@register_jitted(cache=True)
def trend_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    up_th: tp.FlexArray1dLike,
    down_th: tp.FlexArray1dLike,
    mode: int = TrendLabelMode.Binary,
) -> tp.Array2d:
    """Trend labels based on `TrendLabelMode`."""
    pivots = pivots_1d_nb(high, low, up_th, down_th)
    if mode == TrendLabelMode.Binary:
        return bin_trend_labels_1d_nb(pivots)
    if mode == TrendLabelMode.BinaryCont:
        return binc_trend_labels_1d_nb(high, low, pivots)
    if mode == TrendLabelMode.BinaryContSat:
        return bincs_trend_labels_1d_nb(high, low, pivots, up_th, down_th)
    if mode == TrendLabelMode.PctChange:
        return pct_trend_labels_1d_nb(high, low, pivots, normalize=False)
    if mode == TrendLabelMode.PctChangeNorm:
        return pct_trend_labels_1d_nb(high, low, pivots, normalize=True)
    raise ValueError("Invalid trend mode")


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        mode=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def trend_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    up_th: tp.FlexArray2dLike,
    down_th: tp.FlexArray2dLike,
    mode: tp.FlexArray1dLike = TrendLabelMode.Binary,
) -> tp.Array2d:
    """2-dim version of `trend_labels_1d_nb`."""
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))
    mode_ = to_1d_array_nb(np.asarray(mode))

    trend_labels = np.empty(high.shape, dtype=np.float_)
    for col in prange(high.shape[1]):
        trend_labels[:, col] = trend_labels_1d_nb(
            high[:, col],
            low[:, col],
            flex_select_col_nb(up_th_, col),
            flex_select_col_nb(down_th_, col),
            mode=flex_select_1d_nb(mode_, col),
        )
    return trend_labels


# ############# BOLB ############# #


@register_jitted(cache=True)
def breakout_labels_1d_nb(
    high: tp.Array1d,
    low: tp.Array1d,
    window: int = 14,
    up_th: tp.FlexArray1dLike = np.inf,
    down_th: tp.FlexArray1dLike = np.inf,
    wait: int = 1,
) -> tp.Array1d:
    """For each value, return 1 if any value in the next period is greater than the
    positive threshold (in %), -1 if less than the negative threshold, and 0 otherwise.

    First hit wins. Continue search if both thresholds were hit at the same time."""
    up_th_ = to_1d_array_nb(np.asarray(up_th))
    down_th_ = to_1d_array_nb(np.asarray(down_th))

    breakout_labels = np.full(high.shape, 0, dtype=np.float_)
    for i in range(high.shape[0]):
        if not np.isnan(high[i]) and not np.isnan(low[i]):
            _up_th = 1 + abs(flex_select_1d_nb(up_th_, i))
            _down_th = 1 - abs(flex_select_1d_nb(down_th_, i))

            for j in range(i + wait, min(i + window + wait, high.shape[0])):
                if not np.isnan(high[j]) and not np.isnan(low[j]):
                    if not np.isnan(_up_th) and high[j] >= low[i] * _up_th:
                        breakout_labels[i] = 1
                        break
                    if not np.isnan(_down_th) and low[j] <= high[i] * _down_th:
                        if breakout_labels[i] == 1:
                            breakout_labels[i] = 0
                            continue
                        breakout_labels[i] = -1
                        break

    return breakout_labels


@register_chunkable(
    size=ch.ArraySizer(arg_query="high", axis=1),
    arg_take_spec=dict(
        high=ch.ArraySlicer(axis=1),
        low=ch.ArraySlicer(axis=1),
        window=base_ch.FlexArraySlicer(),
        up_th=base_ch.FlexArraySlicer(axis=1),
        down_th=base_ch.FlexArraySlicer(axis=1),
        wait=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def breakout_labels_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    window: tp.FlexArray1dLike = 14,
    up_th: tp.FlexArray2dLike = np.inf,
    down_th: tp.FlexArray2dLike = np.inf,
    wait: tp.FlexArray1dLike = 1,
) -> tp.Array2d:
    """2-dim version of `breakout_labels_1d_nb`."""
    window_ = to_1d_array_nb(np.asarray(window))
    up_th_ = to_2d_array_nb(np.asarray(up_th))
    down_th_ = to_2d_array_nb(np.asarray(down_th))
    wait_ = to_1d_array_nb(np.asarray(wait))

    breakout_labels = np.empty(high.shape, dtype=np.float_)
    for col in prange(high.shape[1]):
        breakout_labels[:, col] = breakout_labels_1d_nb(
            high[:, col],
            low[:, col],
            window=flex_select_1d_nb(window_, col),
            up_th=flex_select_col_nb(up_th_, col),
            down_th=flex_select_col_nb(down_th_, col),
            wait=flex_select_1d_nb(wait_, col),
        )
    return breakout_labels
