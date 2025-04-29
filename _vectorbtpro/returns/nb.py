# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for returns.

Provides an arsenal of Numba-compiled functions that are used by accessors and for measuring
portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import prange

from vectorbtpro import _typing as tp
from vectorbtpro._settings import settings
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb
from vectorbtpro.base.reshaping import to_1d_array_nb
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.returns.enums import RollSharpeAIS, RollSharpeAOS
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import add_nb

__all__ = []

_inf_to_nan = settings["returns"]["inf_to_nan"]
_nan_to_zero = settings["returns"]["nan_to_zero"]


# ############# Metrics ############# #


@register_jitted(cache=True)
def get_return_nb(
    input_value: float,
    output_value: float,
    log_returns: bool = False,
    inf_to_nan: bool = _inf_to_nan,
    nan_to_zero: bool = _nan_to_zero,
) -> float:
    """Calculate return from input and output value."""
    if input_value == 0:
        if output_value == 0:
            r = 0.0
        else:
            r = np.inf * np.sign(output_value)
    else:
        return_value = add_nb(output_value, -input_value) / input_value
        if log_returns:
            r = np.log1p(return_value)
        else:
            r = return_value
    if inf_to_nan and np.isinf(r):
        r = np.nan
    if nan_to_zero and np.isnan(r):
        r = 0.0
    return r


@register_jitted(cache=True)
def returns_1d_nb(
    arr: tp.Array1d,
    init_value: float = np.nan,
    log_returns: bool = False,
) -> tp.Array1d:
    """Calculate returns."""
    out = np.empty(arr.shape, dtype=np.float_)
    if np.isnan(init_value) and arr.shape[0] > 0:
        input_value = arr[0]
    else:
        input_value = init_value
    for i in range(arr.shape[0]):
        output_value = arr[i]
        out[i] = get_return_nb(input_value, output_value, log_returns=log_returns)
        input_value = output_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="arr", axis=1),
    arg_take_spec=dict(
        arr=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    arr: tp.Array2d,
    init_value: tp.FlexArray1dLike = np.nan,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """2-dim version of `returns_1d_nb`."""
    init_value_ = to_1d_array_nb(np.asarray(init_value))

    out = np.full(arr.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=arr.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(arr.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_value = flex_select_1d_pc_nb(init_value_, col)

        out[_sim_start:_sim_end, col] = returns_1d_nb(
            arr[_sim_start:_sim_end, col],
            init_value=_init_value,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def cumulative_returns_1d_nb(
    returns: tp.Array1d,
    start_value: float = 1.0,
    log_returns: bool = False,
) -> tp.Array1d:
    """Cumulative returns."""
    out = np.empty_like(returns, dtype=np.float_)
    if log_returns:
        cumsum = 0
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumsum += returns[i]
            if start_value == 0:
                out[i] = cumsum
            else:
                out[i] = np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumprod *= 1 + returns[i]
            if start_value == 0:
                out[i] = cumprod - 1
            else:
                out[i] = cumprod * start_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        start_value=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cumulative_returns_nb(
    returns: tp.Array2d,
    start_value: float = 1.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """2-dim version of `cumulative_returns_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = cumulative_returns_1d_nb(
            returns[_sim_start:_sim_end, col],
            start_value=start_value,
            log_returns=log_returns,
        )
    return out


@register_jitted(cache=True)
def final_value_1d_nb(
    returns: tp.Array1d,
    start_value: float = 1.0,
    log_returns: bool = False,
) -> float:
    """Final value."""
    if log_returns:
        cumsum = 0
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumsum += returns[i]
        if start_value == 0:
            return cumsum
        return np.exp(cumsum) * start_value
    else:
        cumprod = 1
        for i in range(returns.shape[0]):
            if not np.isnan(returns[i]):
                cumprod *= 1 + returns[i]
        if start_value == 0:
            return cumprod - 1
        return cumprod * start_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        start_value=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def final_value_nb(
    returns: tp.Array2d,
    start_value: float = 1.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `final_value_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = final_value_1d_nb(
            returns[_sim_start:_sim_end, col],
            start_value=start_value,
            log_returns=log_returns,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        start_value=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_final_value_nb(
    returns: tp.Array2d,
    window: int,
    start_value: float = 1.0,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `final_value_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            final_value_1d_nb,
            start_value,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def total_return_1d_nb(returns: tp.Array1d, log_returns: bool = False) -> float:
    """Total return."""
    return final_value_1d_nb(returns, start_value=0.0, log_returns=log_returns)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def total_return_nb(
    returns: tp.Array2d,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `total_return_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = total_return_1d_nb(returns[_sim_start:_sim_end, col], log_returns=log_returns)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_total_return_nb(
    returns: tp.Array2d,
    window: int,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `total_return_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            total_return_1d_nb,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def annualized_return_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[float] = None,
) -> float:
    """Annualized total return.

    This is equivalent to the compound annual growth rate (CAGR)."""
    if period is None:
        period = returns.shape[0]
    final_value = final_value_1d_nb(returns, log_returns=log_returns)
    if period == 0:
        return np.nan
    return final_value ** (ann_factor / period) - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        period=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_return_nb(
    returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `annualized_return_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if period is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(period).astype(np.int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = annualized_return_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            period=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_annualized_return_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `annualized_return_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            annualized_return_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def annualized_volatility_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
) -> float:
    """Annualized volatility of a strategy."""
    return generic_nb.nanstd_1d_nb(returns, ddof=ddof) * ann_factor ** (1.0 / levy_alpha)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        levy_alpha=None,
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def annualized_volatility_nb(
    returns: tp.Array2d,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `annualized_volatility_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = annualized_volatility_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            levy_alpha=levy_alpha,
            ddof=ddof,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        levy_alpha=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_annualized_volatility_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `annualized_volatility_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            annualized_volatility_1d_nb,
            ann_factor,
            levy_alpha,
            ddof,
        )
    return out


@register_jitted(cache=True)
def max_drawdown_1d_nb(returns: tp.Array1d, log_returns: bool = False) -> float:
    """Total maximum drawdown (MDD)."""
    cum_ret = np.nan
    value_max = 1.0
    out = 0.0
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            if np.isnan(cum_ret):
                cum_ret = 1.0
            if log_returns:
                ret = np.exp(returns[i]) - 1
            else:
                ret = returns[i]
            cum_ret *= ret + 1.0
        if cum_ret > value_max:
            value_max = cum_ret
        elif cum_ret < value_max:
            dd = cum_ret / value_max - 1
            if dd < out:
                out = dd
    if np.isnan(cum_ret):
        return np.nan
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def max_drawdown_nb(
    returns: tp.Array2d,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `max_drawdown_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = max_drawdown_1d_nb(returns[_sim_start:_sim_end, col], log_returns=log_returns)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_max_drawdown_nb(
    returns: tp.Array2d,
    window: int,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `max_drawdown_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            max_drawdown_1d_nb,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def calmar_ratio_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[float] = None,
) -> float:
    """Calmar ratio, or drawdown ratio, of a strategy."""
    max_drawdown = max_drawdown_1d_nb(returns, log_returns=log_returns)
    if max_drawdown == 0:
        return np.nan
    annualized_return = annualized_return_1d_nb(
        returns,
        ann_factor,
        log_returns=log_returns,
        period=period,
    )
    if max_drawdown == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / np.abs(max_drawdown)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        period=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def calmar_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `calmar_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if period is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(period).astype(np.int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = calmar_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            period=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_calmar_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `calmar_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            calmar_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def deannualized_return_nb(ret: float, ann_factor: float) -> float:
    """Deannualized return."""
    if ann_factor == 1:
        return ret
    if ann_factor <= -1:
        return np.nan
    return (1 + ret) ** (1.0 / ann_factor) - 1


@register_jitted(cache=True)
def omega_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Omega ratio of a strategy."""
    numer = 0.0
    denom = 0.0
    for i in range(returns.shape[0]):
        ret = returns[i]
        if ret > 0:
            numer += ret
        elif ret < 0:
            denom -= ret
    if denom == 0:
        if numer == 0:
            return np.nan
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def omega_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `omega_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = omega_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_omega_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `omega_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            omega_ratio_1d_nb,
        )
    return out


@register_jitted(cache=True)
def sharpe_ratio_1d_nb(
    returns: tp.Array1d,
    ann_factor: float,
    ddof: int = 0,
) -> float:
    """Sharpe ratio of a strategy."""
    mean = np.nanmean(returns)
    std = generic_nb.nanstd_1d_nb(returns, ddof=ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sharpe_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `sharpe_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = sharpe_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            ann_factor,
            ddof=ddof,
        )
    return out


@register_jitted(cache=True)
def rolling_sharpe_ratio_acc_nb(in_state: RollSharpeAIS) -> RollSharpeAOS:
    """Accumulator of `rolling_sharpe_ratio_stream_nb`.

    Takes a state of type `vectorbtpro.returns.enums.RollSharpeAIS` and returns
    a state of type `vectorbtpro.returns.enums.RollSharpeAOS`."""
    mean_in_state = generic_enums.RollMeanAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
    )
    mean_out_state = generic_nb.rolling_mean_acc_nb(mean_in_state)
    std_in_state = generic_enums.RollStdAIS(
        i=in_state.i,
        value=in_state.ret,
        pre_window_value=in_state.pre_window_ret,
        cumsum=in_state.cumsum,
        cumsum_sq=in_state.cumsum_sq,
        nancnt=in_state.nancnt,
        window=in_state.window,
        minp=in_state.minp,
        ddof=in_state.ddof,
    )
    std_out_state = generic_nb.rolling_std_acc_nb(std_in_state)
    mean = mean_out_state.value
    std = std_out_state.value
    if std == 0:
        sharpe = np.nan
    else:
        sharpe = mean / std * np.sqrt(in_state.ann_factor)

    return RollSharpeAOS(
        cumsum=std_out_state.cumsum,
        cumsum_sq=std_out_state.cumsum_sq,
        nancnt=std_out_state.nancnt,
        value=sharpe,
    )


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def rolling_sharpe_ratio_stream_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling Sharpe ratio in a streaming fashion.

    Uses `rolling_sharpe_ratio_acc_nb` at each iteration."""
    if window is None:
        window = returns.shape[0]
    if minp is None:
        minp = window

    out = np.full(returns.shape, np.nan, dtype=np.float_)
    if returns.shape[0] == 0:
        return out

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        cumsum = 0.0
        cumsum_sq = 0.0
        nancnt = 0

        for i in range(_sim_start, _sim_end):
            in_state = RollSharpeAIS(
                i=i - _sim_start,
                ret=returns[i, col],
                pre_window_ret=returns[i - window, col] if i - window >= 0 else np.nan,
                cumsum=cumsum,
                cumsum_sq=cumsum_sq,
                nancnt=nancnt,
                window=window,
                minp=minp,
                ddof=ddof,
                ann_factor=ann_factor,
            )
            out_state = rolling_sharpe_ratio_acc_nb(in_state)
            cumsum = out_state.cumsum
            cumsum_sq = out_state.cumsum_sq
            nancnt = out_state.nancnt
            out[i, col] = out_state.value

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        stream_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_sharpe_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    stream_mode: bool = True,
) -> tp.Array2d:
    """Rolling version of `sharpe_ratio_1d_nb`."""
    if stream_mode:
        return rolling_sharpe_ratio_stream_nb(
            returns,
            window,
            ann_factor,
            minp=minp,
            ddof=ddof,
            sim_start=sim_start,
            sim_end=sim_end,
        )

    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            sharpe_ratio_1d_nb,
            ann_factor,
            ddof,
        )
    return out


@register_jitted(cache=True)
def downside_risk_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """Downside deviation below a threshold."""
    cnt = 0
    adj_ret_sqrd_sum = np.nan
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            cnt += 1
            if np.isnan(adj_ret_sqrd_sum):
                adj_ret_sqrd_sum = 0.0
            if returns[i] <= 0:
                adj_ret_sqrd_sum += returns[i] ** 2
    if cnt == 0:
        return np.nan
    adj_ret_sqrd_mean = adj_ret_sqrd_sum / cnt
    return np.sqrt(adj_ret_sqrd_mean) * np.sqrt(ann_factor)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def downside_risk_nb(
    returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `downside_risk_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = downside_risk_1d_nb(returns[_sim_start:_sim_end, col], ann_factor)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_downside_risk_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `downside_risk_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            downside_risk_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def sortino_ratio_1d_nb(returns: tp.Array1d, ann_factor: float) -> float:
    """Sortino ratio of a strategy."""
    avg_annualized_return = np.nanmean(returns) * ann_factor
    downside_risk = downside_risk_1d_nb(returns, ann_factor)
    if downside_risk == 0:
        if avg_annualized_return == 0:
            return np.nan
        return np.inf
    return avg_annualized_return / downside_risk


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def sortino_ratio_nb(
    returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `sortino_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = sortino_ratio_1d_nb(returns[_sim_start:_sim_end, col], ann_factor)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_sortino_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `sortino_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            sortino_ratio_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def information_ratio_1d_nb(returns: tp.Array1d, ddof: int = 0) -> float:
    """Information ratio of a strategy."""
    mean = np.nanmean(returns)
    std = generic_nb.nanstd_1d_nb(returns, ddof=ddof)
    if std == 0:
        if mean == 0:
            return np.nan
        return np.inf
    return mean / std


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def information_ratio_nb(
    returns: tp.Array2d,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `information_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = information_ratio_1d_nb(returns[_sim_start:_sim_end, col], ddof=ddof)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_information_ratio_nb(
    returns: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `information_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            information_ratio_1d_nb,
            ddof,
        )
    return out


@register_jitted(cache=True)
def beta_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ddof: int = 0,
) -> float:
    """Beta."""
    cov = generic_nb.nancov_1d_nb(returns, bm_returns, ddof=ddof)
    var = generic_nb.nanvar_1d_nb(bm_returns, ddof=ddof)
    if var == 0:
        if cov == 0:
            return np.nan
        return np.inf
    return cov / var


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ddof=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def beta_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ddof: int = 0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `beta_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = beta_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ddof=ddof,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ddof=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_beta_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ddof: int = 0,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `beta_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            beta_1d_nb,
            ddof,
        )
    return out


@register_jitted(cache=True)
def alpha_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
) -> float:
    """Annualized alpha."""
    beta = beta_1d_nb(returns, bm_returns)
    return (np.nanmean(returns) - beta * np.nanmean(bm_returns) + 1) ** ann_factor - 1


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def alpha_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `alpha_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = alpha_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_alpha_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `alpha_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            alpha_1d_nb,
            ann_factor,
        )
    return out


@register_jitted(cache=True)
def tail_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Ratio between the right (95%) and left tail (5%)."""
    perc_95 = np.abs(np.nanpercentile(returns, 95))
    perc_5 = np.abs(np.nanpercentile(returns, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_jitted(cache=True)
def tail_ratio_noarr_1d_nb(returns: tp.Array1d) -> float:
    """`tail_ratio_1d_nb` that does not allocate any arrays."""
    perc_95 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(returns, 95))
    perc_5 = np.abs(generic_nb.nanpercentile_noarr_1d_nb(returns, 5))
    if perc_5 == 0:
        if perc_95 == 0:
            return np.nan
        return np.inf
    return perc_95 / perc_5


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def tail_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """2-dim version of `tail_ratio_1d_nb` and `tail_ratio_noarr_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = tail_ratio_noarr_1d_nb(returns[_sim_start:_sim_end, col])
        else:
            out[col] = tail_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_tail_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Rolling version of `tail_ratio_1d_nb` and `tail_ratio_noarr_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                tail_ratio_noarr_1d_nb,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                tail_ratio_1d_nb,
            )
    return out


@register_jitted(cache=True)
def profit_factor_1d_nb(returns: tp.Array1d) -> float:
    """Profit factor."""
    numer = 0
    denom = 0
    for i in range(returns.shape[0]):
        if not np.isnan(returns[i]):
            if returns[i] > 0:
                numer += returns[i]
            elif returns[i] < 0:
                denom += abs(returns[i])
    if denom == 0:
        if numer == 0:
            return np.nan
        return np.inf
    return numer / denom


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def profit_factor_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `profit_factor_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = profit_factor_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_profit_factor_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `profit_factor_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            profit_factor_1d_nb,
        )
    return out


@register_jitted(cache=True)
def common_sense_ratio_1d_nb(returns: tp.Array1d) -> float:
    """Common Sense Ratio."""
    tail_ratio = tail_ratio_1d_nb(returns)
    profit_factor = profit_factor_1d_nb(returns)
    return tail_ratio * profit_factor


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def common_sense_ratio_nb(
    returns: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `common_sense_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[col] = common_sense_ratio_1d_nb(returns[_sim_start:_sim_end, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_common_sense_ratio_nb(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `common_sense_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
            returns[_sim_start:_sim_end, col],
            window,
            minp,
            common_sense_ratio_1d_nb,
        )
    return out


@register_jitted(cache=True)
def value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Value at risk (VaR) of a returns stream."""
    return np.nanpercentile(returns, 100 * cutoff)


@register_jitted(cache=True)
def value_at_risk_noarr_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """`value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpercentile_noarr_1d_nb(returns, 100 * cutoff)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        cutoff=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_at_risk_nb(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """2-dim version of `value_at_risk_1d_nb` and `value_at_risk_noarr_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = value_at_risk_noarr_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
        else:
            out[col] = value_at_risk_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        cutoff=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_value_at_risk_nb(
    returns: tp.Array2d,
    window: int,
    cutoff: float = 0.05,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Rolling version of `value_at_risk_1d_nb` and `value_at_risk_noarr_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                value_at_risk_noarr_1d_nb,
                cutoff,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                value_at_risk_1d_nb,
                cutoff,
            )
    return out


@register_jitted(cache=True)
def cond_value_at_risk_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """Conditional value at risk (CVaR) of a returns stream."""
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[: cutoff_index + 1])


@register_jitted(cache=True)
def cond_value_at_risk_noarr_1d_nb(returns: tp.Array1d, cutoff: float = 0.05) -> float:
    """`cond_value_at_risk_1d_nb` that does not allocate any arrays."""
    return generic_nb.nanpartition_mean_noarr_1d_nb(returns, cutoff * 100)


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        cutoff=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cond_value_at_risk_nb(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array1d:
    """2-dim version of `cond_value_at_risk_1d_nb` and `cond_value_at_risk_noarr_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[col] = cond_value_at_risk_noarr_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
        else:
            out[col] = cond_value_at_risk_1d_nb(returns[_sim_start:_sim_end, col], cutoff=cutoff)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        window=None,
        cutoff=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
        noarr_mode=None,
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_cond_value_at_risk_nb(
    returns: tp.Array2d,
    window: int,
    cutoff: float = 0.05,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
    noarr_mode: bool = True,
) -> tp.Array2d:
    """Rolling version of `cond_value_at_risk_1d_nb` and `cond_value_at_risk_noarr_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        if noarr_mode:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                cond_value_at_risk_noarr_1d_nb,
                cutoff,
            )
        else:
            out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_1d_nb(
                returns[_sim_start:_sim_end, col],
                window,
                minp,
                cond_value_at_risk_1d_nb,
                cutoff,
            )
    return out


@register_jitted(cache=True)
def capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[float] = None,
) -> float:
    """Capture ratio."""
    annualized_return1 = annualized_return_1d_nb(
        returns,
        ann_factor,
        log_returns=log_returns,
        period=period,
    )
    annualized_return2 = annualized_return_1d_nb(
        bm_returns,
        ann_factor,
        log_returns=log_returns,
        period=period,
    )
    if annualized_return2 == 0:
        if annualized_return1 == 0:
            return np.nan
        return np.inf
    return annualized_return1 / annualized_return2


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        period=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `capture_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if period is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(period).astype(np.int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            period=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `capture_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def up_capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[float] = None,
) -> float:
    """Capture ratio for periods when the benchmark return is positive."""
    if period is None:
        period = returns.shape[0]

    def _annualized_pos_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a > 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        if period == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_pos_return(returns)
    annualized_bm_return = _annualized_pos_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        period=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def up_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `up_capture_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if period is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(period).astype(np.int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = up_capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            period=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_up_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `up_capture_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            up_capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out


@register_jitted(cache=True)
def down_capture_ratio_1d_nb(
    returns: tp.Array1d,
    bm_returns: tp.Array1d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[float] = None,
) -> float:
    """Capture ratio for periods when the benchmark return is negative."""
    if period is None:
        period = returns.shape[0]

    def _annualized_neg_return(a):
        ann_ret = np.nan
        ret_cnt = 0
        for i in range(a.shape[0]):
            if not np.isnan(a[i]):
                if log_returns:
                    _a = np.exp(a[i]) - 1
                else:
                    _a = a[i]
                if np.isnan(ann_ret):
                    ann_ret = 1.0
                if _a < 0:
                    ann_ret *= _a + 1.0
                    ret_cnt += 1
        if ret_cnt == 0:
            return np.nan
        if period == 0:
            return np.nan
        return ann_ret ** (ann_factor / period) - 1

    annualized_return = _annualized_neg_return(returns)
    annualized_bm_return = _annualized_neg_return(bm_returns)
    if annualized_bm_return == 0:
        if annualized_return == 0:
            return np.nan
        return np.inf
    return annualized_return / annualized_bm_return


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        ann_factor=None,
        log_returns=None,
        period=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def down_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    ann_factor: float,
    log_returns: bool = False,
    period: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """2-dim version of `down_capture_ratio_1d_nb`."""
    out = np.full(returns.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    if period is None:
        period_ = sim_end_ - sim_start_
    else:
        period_ = to_1d_array_nb(np.asarray(period).astype(np.int_))
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _period = flex_select_1d_pc_nb(period_, col)

        out[col] = down_capture_ratio_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            ann_factor,
            log_returns=log_returns,
            period=_period,
        )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="returns", axis=1),
    arg_take_spec=dict(
        returns=ch.ArraySlicer(axis=1),
        bm_returns=ch.ArraySlicer(axis=1),
        window=None,
        ann_factor=None,
        log_returns=None,
        minp=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(tags={"can_parallel"})
def rolling_down_capture_ratio_nb(
    returns: tp.Array2d,
    bm_returns: tp.Array2d,
    window: int,
    ann_factor: float,
    log_returns: bool = False,
    minp: tp.Optional[int] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Rolling version of `down_capture_ratio_1d_nb`."""
    out = np.full(returns.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=returns.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(returns.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        out[_sim_start:_sim_end, col] = generic_nb.rolling_reduce_two_1d_nb(
            returns[_sim_start:_sim_end, col],
            bm_returns[_sim_start:_sim_end, col],
            window,
            minp,
            down_capture_ratio_1d_nb,
            ann_factor,
            log_returns,
        )
    return out
