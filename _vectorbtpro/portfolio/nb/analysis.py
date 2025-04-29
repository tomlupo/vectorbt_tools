# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled functions for portfolio analysis."""

from numba import prange

from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.reshaping import to_1d_array_nb, to_2d_array_nb
from vectorbtpro.portfolio import chunking as portfolio_ch
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import chunking as records_ch
from vectorbtpro.registries.ch_registry import register_chunkable
from vectorbtpro.returns import nb as returns_nb_
from vectorbtpro.utils import chunking as ch
from vectorbtpro.utils.math_ import is_close_nb, add_nb
from vectorbtpro.utils.template import RepFunc


# ############# Assets ############# #


@register_jitted(cache=True)
def get_long_size_nb(position_before: float, position_now: float) -> float:
    """Get long size."""
    if position_before <= 0 and position_now <= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_before
    if position_before < 0 and position_now >= 0:
        return position_now
    return add_nb(position_now, -position_before)


@register_jitted(cache=True)
def get_short_size_nb(position_before: float, position_now: float) -> float:
    """Get short size."""
    if position_before >= 0 and position_now >= 0:
        return 0.0
    if position_before >= 0 and position_now < 0:
        return -position_now
    if position_before < 0 and position_now >= 0:
        return position_before
    return add_nb(position_before, -position_now)


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        direction=None,
        init_position=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    direction: int = Direction.Both,
    init_position: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset flow series per column.

    Returns the total transacted amount of assets at each time step."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    out = np.full(target_shape, np.nan, dtype=np.float_)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        out[_sim_start:_sim_end, col] = 0.0
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = flex_select_1d_pc_nb(init_position_, col)

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]

            if side == OrderSide.Sell:
                size *= -1
            new_position_now = add_nb(position_now, size)
            if direction == Direction.LongOnly:
                asset_flow = get_long_size_nb(position_now, new_position_now)
            elif direction == Direction.ShortOnly:
                asset_flow = get_short_size_nb(position_now, new_position_now)
            else:
                asset_flow = size
            out[i, col] = add_nb(out[i, col], asset_flow)
            position_now = new_position_now
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_flow", axis=1),
    arg_take_spec=dict(
        asset_flow=ch.ArraySlicer(axis=1),
        direction=None,
        init_position=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def assets_nb(
    asset_flow: tp.Array2d,
    direction: int = Direction.Both,
    init_position: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset series per column.

    Returns the current position at each time step."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))

    out = np.full(asset_flow.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        position_now = flex_select_1d_pc_nb(init_position_, col)

        for i in range(_sim_start, _sim_end):
            flow_value = asset_flow[i, col]
            position_now = add_nb(position_now, flow_value)
            if direction == Direction.Both:
                out[i, col] = position_now
            elif direction == Direction.LongOnly and position_now > 0:
                out[i, col] = position_now
            elif direction == Direction.ShortOnly and position_now < 0:
                out[i, col] = -position_now
            else:
                out[i, col] = 0.0
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="assets", axis=1),
    arg_take_spec=dict(
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_mask_nb(
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get position mask per column."""
    out = np.full(assets.shape, False, dtype=np.bool_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(assets.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if assets[i, col] != 0:
                out[i, col] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        assets=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_mask_grouped_nb(
    assets: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get position mask per group."""
    out = np.full((assets.shape[0], len(group_lens)), False, dtype=np.bool_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                    out[i, group] = True
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="assets", axis=1),
    arg_take_spec=dict(
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_coverage_nb(
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get position mask per column."""
    out = np.full(assets.shape[1], 0.0, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(assets.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        hit_elements = 0

        for i in range(_sim_start, _sim_end):
            if assets[i, col] != 0:
                hit_elements += 1

        out[col] = hit_elements / (_sim_end - _sim_start)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        assets=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        granular_groups=None,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def position_coverage_grouped_nb(
    assets: tp.Array2d,
    group_lens: tp.GroupLens,
    granular_groups: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get position coverage per group."""
    out = np.full(len(group_lens), 0.0, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=assets.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]
        n_elements = 0
        hit_elements = 0

        if granular_groups:
            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                n_elements += _sim_end - _sim_start

                for i in range(_sim_start, _sim_end):
                    if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                        hit_elements += 1
        else:
            min_sim_start = assets.shape[0]
            max_sim_end = 0
            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                if _sim_start < min_sim_start:
                    min_sim_start = _sim_start
                if _sim_end > max_sim_end:
                    max_sim_end = _sim_end
            if min_sim_start >= max_sim_end:
                continue
            n_elements = max_sim_end - min_sim_start

            for i in range(min_sim_start, max_sim_end):
                for col in range(from_col, to_col):
                    _sim_start = sim_start_[col]
                    _sim_end = sim_end_[col]
                    if _sim_start >= _sim_end:
                        continue
                    if not np.isnan(assets[i, col]) and assets[i, col] != 0:
                        hit_elements += 1
                        break

        if n_elements == 0:
            out[group] = np.nan
        else:
            out[group] = hit_elements / n_elements
    return out


# ############# Cash ############# #


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        split_shared=None,
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    cash_deposits_raw: tp.FlexArray2dLike = 0.0,
    split_shared: bool = False,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get cash deposit series per column."""
    cash_deposits_raw_ = to_2d_array_nb(np.asarray(cash_deposits_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    out = np.full(target_shape, np.nan, dtype=np.float_)

    if not cash_sharing:
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for col in prange(target_shape[1]):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            _weights = flex_select_1d_pc_nb(weights_, col)

            for i in range(_sim_start, _sim_end):
                _cash_deposits = flex_select_nb(cash_deposits_raw_, i, col)
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    out[i, col] = _weights * _cash_deposits
                else:
                    out[i, col] = _cash_deposits
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                _weights = flex_select_1d_pc_nb(weights_, col)

                for i in range(_sim_start, _sim_end):
                    _cash_deposits = flex_select_nb(cash_deposits_raw_, i, group)
                    if split_shared:
                        if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                            out[i, col] = _weights * _cash_deposits / (to_col - from_col)
                        else:
                            out[i, col] = _cash_deposits / (to_col - from_col)
                    else:
                        if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                            out[i, col] = _weights * _cash_deposits
                        else:
                            out[i, col] = _cash_deposits
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_sharing=None,
        cash_deposits_raw=RepFunc(portfolio_ch.get_cash_deposits_slicer),
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_deposits_grouped_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    cash_deposits_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get cash deposit series per group."""
    cash_deposits_raw_ = to_2d_array_nb(np.asarray(cash_deposits_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    out = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)

    if cash_sharing:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_grouped_sim_range_nb(
            target_shape=target_shape,
            group_lens=group_lens,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            _sim_start = sim_start_[group]
            _sim_end = sim_end_[group]
            if _sim_start >= _sim_end:
                continue
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for i in range(_sim_start, _sim_end):
                _cash_deposits = flex_select_nb(cash_deposits_raw_, i, group)
                if np.isnan(_cash_deposits) or _cash_deposits == 0:
                    out[i, group] = _cash_deposits
                    continue
                group_weight = 0.0
                for col in range(from_col, to_col):
                    _weights = flex_select_1d_pc_nb(weights_, col)
                    if not np.isnan(group_weight) and not np.isnan(_weights):
                        group_weight += _weights
                    else:
                        group_weight = np.nan
                        break
                if not np.isnan(group_weight):
                    group_weight /= group_lens[group]
                if not np.isnan(group_weight) and not is_close_nb(group_weight, 1.0):
                    out[i, group] = group_weight * _cash_deposits
                else:
                    out[i, group] = _cash_deposits
    else:
        group_end_idxs = np.cumsum(group_lens)
        group_start_idxs = group_end_idxs - group_lens
        sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
            sim_shape=target_shape,
            sim_start=sim_start,
            sim_end=sim_end,
        )
        for group in prange(len(group_lens)):
            from_col = group_start_idxs[group]
            to_col = group_end_idxs[group]

            for col in range(from_col, to_col):
                _sim_start = sim_start_[col]
                _sim_end = sim_end_[col]
                if _sim_start >= _sim_end:
                    continue
                _weights = flex_select_1d_pc_nb(weights_, col)

                for i in range(_sim_start, _sim_end):
                    _cash_deposits = flex_select_nb(cash_deposits_raw_, i, col)
                    if np.isnan(out[i, group]):
                        out[i, group] = 0.0
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[i, group] += _weights * _cash_deposits
                    else:
                        out[i, group] += _cash_deposits
    return out


@register_chunkable(
    size=ch.ShapeSizer(arg_query="target_shape", axis=1),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        cash_earnings_raw=base_ch.FlexArraySlicer(axis=1),
        weights=base_ch.FlexArraySlicer(),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_earnings_nb(
    target_shape: tp.Shape,
    cash_earnings_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get cash earning series per column."""
    cash_earnings_raw_ = to_2d_array_nb(np.asarray(cash_earnings_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    out = np.full(target_shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(target_shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _weights = flex_select_1d_pc_nb(weights_, col)

        for i in range(_sim_start, _sim_end):
            _cash_earnings = flex_select_nb(cash_earnings_raw_, i, col)
            if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                out[i, col] = _weights * _cash_earnings
            else:
                out[i, col] = _cash_earnings
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        target_shape=base_ch.shape_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        cash_earnings_raw=base_ch.flex_array_gl_slicer,
        weights=base_ch.flex_1d_array_gl_slicer,
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_earnings_grouped_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_earnings_raw: tp.FlexArray2dLike = 0.0,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get cash earning series per group."""
    cash_earnings_raw_ = to_2d_array_nb(np.asarray(cash_earnings_raw))
    if weights is None:
        weights_ = np.full(target_shape[1], np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    out = np.full((target_shape[0], len(group_lens)), np.nan, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            _weights = flex_select_1d_pc_nb(weights_, col)

            for i in range(_sim_start, _sim_end):
                _cash_earnings = flex_select_nb(cash_earnings_raw_, i, col)
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    out[i, group] += _weights * _cash_earnings
                else:
                    out[i, group] += _cash_earnings
    return out


@register_jitted(cache=True)
def get_free_cash_diff_nb(
    position_before: float,
    position_now: float,
    debt_now: float,
    price: float,
    fees: float,
) -> tp.Tuple[float, float]:
    """Get updated debt and free cash flow."""
    size = add_nb(position_now, -position_before)
    final_cash = -size * price - fees
    if is_close_nb(size, 0):
        new_debt = debt_now
        free_cash_diff = 0.0
    elif size > 0:
        if position_before < 0:
            if position_now < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_before)
            avg_entry_price = debt_now / abs(position_before)
            debt_diff = short_size * avg_entry_price
            new_debt = add_nb(debt_now, -debt_diff)
            free_cash_diff = add_nb(2 * debt_diff, final_cash)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    else:
        if position_now < 0:
            if position_before < 0:
                short_size = abs(size)
            else:
                short_size = abs(position_now)
            short_value = short_size * price
            new_debt = debt_now + short_value
            free_cash_diff = add_nb(final_cash, -2 * short_value)
        else:
            new_debt = debt_now
            free_cash_diff = final_cash
    return new_debt, free_cash_diff


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        free=None,
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_flow_nb(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    free: bool = False,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get (free) cash flow series per column."""
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    out = np.full(target_shape, np.nan, dtype=np.float_)
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in range(target_shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = flex_select_nb(cash_earnings_, i, col)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            continue
        last_id = -1
        position_now = 0.0
        debt_now = 0.0

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            i = order_record["idx"]
            side = order_record["side"]
            size = order_record["size"]
            price = order_record["price"]
            fees = order_record["fees"]

            if side == OrderSide.Sell:
                size *= -1
            position_before = position_now
            position_now = add_nb(position_now, size)
            if free:
                debt_now, cash_flow = get_free_cash_diff_nb(
                    position_before=position_before,
                    position_now=position_now,
                    debt_now=debt_now,
                    price=price,
                    fees=fees,
                )
            else:
                cash_flow = -size * price - fees
            out[i, col] = add_nb(out[i, col], cash_flow)
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        cash_flow=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_flow_grouped_nb(
    cash_flow: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get (free) cash flow series per group."""
    out = np.full((cash_flow.shape[0], len(group_lens)), np.nan, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += cash_flow[i, col]

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="free_cash_flow", axis=1),
    arg_take_spec=dict(
        init_cash_raw=None,
        free_cash_flow=ch.ArraySlicer(axis=1),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def align_init_cash_nb(
    init_cash_raw: int,
    free_cash_flow: tp.Array2d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Align initial cash to the maximum negative free cash flow per column or group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(free_cash_flow.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=free_cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(free_cash_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        free_cash = 0.0
        min_req_cash = np.inf

        for i in range(_sim_start, _sim_end):
            free_cash = add_nb(free_cash, free_cash_flow[i, col])
            free_cash = add_nb(free_cash, flex_select_nb(cash_deposits_, i, col))
            if free_cash < min_req_cash:
                min_req_cash = free_cash

        if min_req_cash < 0:
            out[col] = np.abs(min_req_cash)
        else:
            out[col] = 1.0

    if init_cash_raw == InitCashMode.AutoAlign:
        out = np.full(out.shape, np.max(out))
    return out


@register_jitted(cache=True)
def init_cash_nb(
    init_cash_raw: tp.FlexArray1d,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    split_shared: bool = False,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get initial cash per column."""
    out = np.empty(np.sum(group_lens), dtype=np.float_)
    if weights is None:
        weights_ = np.full(group_lens.sum(), np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    if not cash_sharing:
        for col in range(out.shape[0]):
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, col)
            _weights = flex_select_1d_pc_nb(weights_, col)
            if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                out[col] = _weights * _init_cash
            else:
                out[col] = _init_cash
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            group_len = to_col - from_col
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, group)
            for col in range(from_col, to_col):
                _weights = flex_select_1d_pc_nb(weights_, col)
                if split_shared:
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[col] = _weights * _init_cash / group_len
                    else:
                        out[col] = _init_cash / group_len
                else:
                    if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                        out[col] = _weights * _init_cash
                    else:
                        out[col] = _init_cash
            from_col = to_col
    return out


@register_jitted(cache=True)
def init_cash_grouped_nb(
    init_cash_raw: tp.FlexArray1d,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    weights: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get initial cash per group."""
    out = np.empty(group_lens.shape, dtype=np.float_)
    if weights is None:
        weights_ = np.full(group_lens.sum(), np.nan, dtype=np.float_)
    else:
        weights_ = to_1d_array_nb(np.asarray(weights).astype(np.float_))

    if cash_sharing:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = flex_select_1d_pc_nb(init_cash_raw, group)
            group_weight = 0.0
            for col in range(from_col, to_col):
                _weights = flex_select_1d_pc_nb(weights_, col)
                if not np.isnan(group_weight) and not np.isnan(_weights):
                    group_weight += _weights
                else:
                    group_weight = np.nan
                    break
            if not np.isnan(group_weight):
                group_weight /= group_lens[group]
            if not np.isnan(group_weight) and not is_close_nb(group_weight, 1.0):
                out[group] = group_weight * _init_cash
            else:
                out[group] = _init_cash
            from_col = to_col
    else:
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            cash_sum = 0.0
            for col in range(from_col, to_col):
                _init_cash = flex_select_1d_pc_nb(init_cash_raw, col)
                _weights = flex_select_1d_pc_nb(weights_, col)
                if not np.isnan(_weights) and not is_close_nb(_weights, 1.0):
                    cash_sum += _weights * _init_cash
                else:
                    cash_sum += _init_cash
            out[group] = cash_sum
            from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash_flow", axis=1),
    arg_take_spec=dict(
        cash_flow=ch.ArraySlicer(axis=1),
        init_cash=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def cash_nb(
    cash_flow: tp.Array2d,
    init_cash: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get cash series per column or group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(cash_flow.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash_flow.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(cash_flow.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        cash_now = flex_select_1d_pc_nb(init_cash, col)

        for i in range(_sim_start, _sim_end):
            cash_now = add_nb(cash_now, flex_select_nb(cash_deposits_, i, col))
            cash_now = add_nb(cash_now, cash_flow[i, col])
            out[i, col] = cash_now
    return out


# ############# Value ############# #


@register_jitted(cache=True)
def init_position_value_nb(
    n_cols: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get initial position value per column."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.empty(n_cols, dtype=np.float_)

    for col in range(n_cols):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position == 0:
            out[col] = 0.0
        else:
            out[col] = _init_position * _init_price
    return out


@register_jitted(cache=True)
def init_position_value_grouped_nb(
    group_lens: tp.GroupLens,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get initial position value per group."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))

    out = np.full(len(group_lens), 0.0, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _init_position = float(flex_select_1d_pc_nb(init_position_, col))
            _init_price = float(flex_select_1d_pc_nb(init_price_, col))
            if _init_position != 0:
                out[group] += _init_position * _init_price
    return out


@register_jitted(cache=True)
def init_value_nb(init_position_value: tp.Array1d, init_cash: tp.FlexArray1d) -> tp.Array1d:
    """Get initial value per column or group."""
    out = np.empty(len(init_position_value), dtype=np.float_)

    for col in range(len(init_position_value)):
        _init_cash = flex_select_1d_pc_nb(init_cash, col)
        out[col] = _init_cash + init_position_value[col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        assets=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_value_nb(
    close: tp.Array2d,
    assets: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset value series per column."""
    out = np.full(close.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(close.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if assets[i, col] == 0:
                out[i, col] = 0.0
            else:
                out[i, col] = close[i, col] * assets[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        asset_value=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_value_grouped_nb(
    asset_value: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset value series per group."""
    out = np.full((asset_value.shape[0], len(group_lens)), np.nan, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += asset_value[i, col]

    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="cash", axis=1),
    arg_take_spec=dict(
        cash=ch.ArraySlicer(axis=1),
        asset_value=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def value_nb(
    cash: tp.Array2d,
    asset_value: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get value series per column or group."""
    out = np.full(cash.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=cash.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(cash.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = cash[i, col] + asset_value[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        value=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def gross_exposure_nb(
    asset_value: tp.Array2d,
    value: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get gross exposure series per column."""
    out = np.full(asset_value.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            if value[i, col] == 0:
                out[i, col] = np.nan
            else:
                out[i, col] = abs(asset_value[i, col] / value[i, col])
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="long_exposure", axis=1),
    arg_take_spec=dict(
        long_exposure=ch.ArraySlicer(axis=1),
        short_exposure=ch.ArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def net_exposure_nb(
    long_exposure: tp.Array2d,
    short_exposure: tp.Array2d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get net exposure series per column."""
    out = np.full(long_exposure.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=long_exposure.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(long_exposure.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        for i in range(_sim_start, _sim_end):
            out[i, col] = long_exposure[i, col] - short_exposure[i, col]
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        asset_value=base_ch.array_gl_slicer,
        value=ch.ArraySlicer(axis=1),
        group_lens=ch.ArraySlicer(axis=0),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def allocations_nb(
    asset_value: tp.Array2d,
    value: tp.Array2d,
    group_lens: tp.GroupLens,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get allocations per column."""
    out = np.full(asset_value.shape, np.nan, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue

            for i in range(_sim_start, _sim_end):
                if value[i, group] == 0:
                    out[i, col] = np.nan
                else:
                    out[i, col] = asset_value[i, col] / value[i, group]
    return out


@register_chunkable(
    size=base_ch.GroupLensSizer(arg_query="col_map"),
    arg_take_spec=dict(
        target_shape=ch.ShapeSlicer(axis=1),
        close=ch.ArraySlicer(axis=1),
        order_records=ch.ArraySlicer(axis=0, mapper=records_ch.col_idxs_mapper),
        col_map=base_ch.GroupMapSlicer(),
        init_position=base_ch.FlexArraySlicer(),
        init_price=base_ch.FlexArraySlicer(),
        cash_earnings=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="concat",
)
@register_jitted(cache=True, tags={"can_parallel"})
def total_profit_nb(
    target_shape: tp.Shape,
    close: tp.Array2d,
    order_records: tp.RecordArray,
    col_map: tp.GroupMap,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
    cash_earnings: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get total profit per column.

    A much faster version than the one based on `value_nb`."""
    init_position_ = to_1d_array_nb(np.asarray(init_position))
    init_price_ = to_1d_array_nb(np.asarray(init_price))
    cash_earnings_ = to_2d_array_nb(np.asarray(cash_earnings))

    assets = np.full(target_shape[1], 0.0, dtype=np.float_)
    cash = np.full(target_shape[1], 0.0, dtype=np.float_)
    total_profit = np.full(target_shape[1], np.nan, dtype=np.float_)

    col_idxs, col_lens = col_map
    col_start_idxs = np.cumsum(col_lens) - col_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=target_shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(target_shape[1]):
        _init_position = float(flex_select_1d_pc_nb(init_position_, col))
        _init_price = float(flex_select_1d_pc_nb(init_price_, col))
        if _init_position != 0:
            assets[col] = _init_position
            cash[col] = -_init_position * _init_price
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        for i in range(_sim_start, _sim_end):
            cash[col] += flex_select_nb(cash_earnings_, i, col)

    for col in prange(col_lens.shape[0]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        col_len = col_lens[col]
        if col_len == 0:
            if assets[col] == 0 and cash[col] == 0:
                total_profit[col] = 0.0
            continue
        last_id = -1

        for c in range(col_len):
            order_record = order_records[col_idxs[col_start_idxs[col] + c]]
            if order_record["idx"] < _sim_start or order_record["idx"] >= _sim_end:
                continue

            if order_record["id"] < last_id:
                raise ValueError("Ids must come in ascending order per column")
            last_id = order_record["id"]

            # Fill assets
            if order_record["side"] == OrderSide.Buy:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], order_size)
            else:
                order_size = order_record["size"]
                assets[col] = add_nb(assets[col], -order_size)

            # Fill cash balance
            if order_record["side"] == OrderSide.Buy:
                order_cash = order_record["size"] * order_record["price"] + order_record["fees"]
                cash[col] = add_nb(cash[col], -order_cash)
            else:
                order_cash = order_record["size"] * order_record["price"] - order_record["fees"]
                cash[col] = add_nb(cash[col], order_cash)

        total_profit[col] = cash[col] + assets[col] * close[_sim_end - 1, col]
    return total_profit


@register_jitted(cache=True)
def total_profit_grouped_nb(total_profit: tp.Array1d, group_lens: tp.GroupLens) -> tp.Array1d:
    """Get total profit per group."""
    out = np.empty(len(group_lens), dtype=np.float_)

    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = np.sum(total_profit[from_col:to_col])
        from_col = to_col
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="value", axis=1),
    arg_take_spec=dict(
        value=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        cash_deposits_as_input=None,
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def returns_nb(
    value: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    cash_deposits_as_input: bool = False,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get return series per column or group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(value.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        input_value = flex_select_1d_pc_nb(init_value, col)

        for i in range(_sim_start, _sim_end):
            _cash_deposits = flex_select_nb(cash_deposits_, i, col)
            output_value = value[i, col]
            if cash_deposits_as_input:
                adj_input_value = input_value + _cash_deposits
                out[i, col] = returns_nb_.get_return_nb(adj_input_value, output_value, log_returns=log_returns)
            else:
                adj_output_value = output_value - _cash_deposits
                out[i, col] = returns_nb_.get_return_nb(input_value, adj_output_value, log_returns=log_returns)
            input_value = output_value
    return out


@register_jitted(cache=True)
def get_asset_pnl_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
) -> float:
    """Get asset PnL from the input and output asset value, and the cash flow."""
    return output_asset_value + cash_flow - input_asset_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
        init_position_value=base_ch.FlexArraySlicer(axis=0),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_pnl_nb(
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
    init_position_value: tp.FlexArray1dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset (realized and unrealized) PnL series per column or group."""
    init_position_value_ = to_1d_array_nb(np.asarray(init_position_value))

    out = np.full(asset_value.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_position_value = flex_select_1d_pc_nb(init_position_value_, col)

        for i in range(_sim_start, _sim_end):
            if i == _sim_start:
                input_asset_value = _init_position_value
            else:
                input_asset_value = asset_value[i - 1, col]
            out[i, col] = get_asset_pnl_nb(
                input_asset_value,
                asset_value[i, col],
                cash_flow[i, col],
            )
    return out


@register_jitted(cache=True)
def get_asset_return_nb(
    input_asset_value: float,
    output_asset_value: float,
    cash_flow: float,
    log_returns: bool = False,
) -> float:
    """Get asset return from the input and output asset value, and the cash flow."""
    if is_close_nb(input_asset_value, 0):
        input_value = -output_asset_value
        output_value = cash_flow
    else:
        input_value = input_asset_value
        output_value = output_asset_value + cash_flow
    if input_value < 0 and output_value < 0:
        return_value = -returns_nb_.get_return_nb(-input_value, -output_value, log_returns=False)
    else:
        return_value = returns_nb_.get_return_nb(input_value, output_value, log_returns=False)
    if log_returns:
        return np.log1p(return_value)
    return return_value


@register_chunkable(
    size=ch.ArraySizer(arg_query="asset_value", axis=1),
    arg_take_spec=dict(
        asset_value=ch.ArraySlicer(axis=1),
        cash_flow=ch.ArraySlicer(axis=1),
        init_position_value=base_ch.FlexArraySlicer(axis=0),
        log_returns=None,
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def asset_returns_nb(
    asset_value: tp.Array2d,
    cash_flow: tp.Array2d,
    init_position_value: tp.FlexArray1dLike = 0.0,
    log_returns: bool = False,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get asset return series per column or group."""
    init_position_value_ = to_1d_array_nb(np.asarray(init_position_value))

    out = np.full(asset_value.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=asset_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(asset_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        _init_position_value = flex_select_1d_pc_nb(init_position_value_, col)

        for i in range(_sim_start, _sim_end):
            if i == _sim_start:
                input_asset_value = _init_position_value
            else:
                input_asset_value = asset_value[i - 1, col]
            out[i, col] = get_asset_return_nb(
                input_asset_value,
                asset_value[i, col],
                cash_flow[i, col],
                log_returns=log_returns,
            )
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="close", axis=1),
    arg_take_spec=dict(
        close=ch.ArraySlicer(axis=1),
        init_value=base_ch.FlexArraySlicer(),
        cash_deposits=base_ch.FlexArraySlicer(axis=1),
        sim_start=base_ch.FlexArraySlicer(),
        sim_end=base_ch.FlexArraySlicer(),
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_nb(
    close: tp.Array2d,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get market value per column."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full(close.shape, np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in prange(close.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue
        curr_value = flex_select_1d_pc_nb(init_value, col)

        for i in range(_sim_start, _sim_end):
            if i > _sim_start:
                curr_value *= close[i, col] / close[i - 1, col]
            curr_value += flex_select_nb(cash_deposits_, i, col)
            out[i, col] = curr_value
    return out


@register_chunkable(
    size=ch.ArraySizer(arg_query="group_lens", axis=0),
    arg_take_spec=dict(
        close=base_ch.array_gl_slicer,
        group_lens=ch.ArraySlicer(axis=0),
        init_value=base_ch.FlexArraySlicer(mapper=base_ch.group_lens_mapper),
        cash_deposits=base_ch.FlexArraySlicer(axis=1, mapper=base_ch.group_lens_mapper),
        sim_start=base_ch.flex_1d_array_gl_slicer,
        sim_end=base_ch.flex_1d_array_gl_slicer,
    ),
    merge_func="column_stack",
)
@register_jitted(cache=True, tags={"can_parallel"})
def market_value_grouped_nb(
    close: tp.Array2d,
    group_lens: tp.GroupLens,
    init_value: tp.FlexArray1d,
    cash_deposits: tp.FlexArray2dLike = 0.0,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array2d:
    """Get market value per group."""
    cash_deposits_ = to_2d_array_nb(np.asarray(cash_deposits))

    out = np.full((close.shape[0], len(group_lens)), np.nan, dtype=np.float_)

    group_end_idxs = np.cumsum(group_lens)
    group_start_idxs = group_end_idxs - group_lens
    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=close.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for group in prange(len(group_lens)):
        from_col = group_start_idxs[group]
        to_col = group_end_idxs[group]

        for col in range(from_col, to_col):
            _sim_start = sim_start_[col]
            _sim_end = sim_end_[col]
            if _sim_start >= _sim_end:
                continue
            curr_value = prev_value = flex_select_1d_pc_nb(init_value, col)

            for i in range(_sim_start, _sim_end):
                if i > _sim_start:
                    if not np.isnan(close[i - 1, col]):
                        prev_close = close[i - 1, col]
                        prev_value = prev_close
                    else:
                        prev_close = prev_value
                    if not np.isnan(close[i, col]):
                        curr_close = close[i, col]
                        prev_value = curr_close
                    else:
                        curr_close = prev_value
                    curr_value *= curr_close / prev_close
                curr_value += flex_select_nb(cash_deposits_, i, col)
                if np.isnan(out[i, group]):
                    out[i, group] = 0.0
                out[i, group] += curr_value
    return out


@register_jitted(cache=True)
def total_market_return_nb(
    market_value: tp.Array2d,
    input_value: tp.FlexArray1d,
    sim_start: tp.Optional[tp.FlexArray1dLike] = None,
    sim_end: tp.Optional[tp.FlexArray1dLike] = None,
) -> tp.Array1d:
    """Get total market return per column or group."""
    out = np.full(market_value.shape[1], np.nan, dtype=np.float_)

    sim_start_, sim_end_ = generic_nb.prepare_sim_range_nb(
        sim_shape=market_value.shape,
        sim_start=sim_start,
        sim_end=sim_end,
    )
    for col in range(market_value.shape[1]):
        _sim_start = sim_start_[col]
        _sim_end = sim_end_[col]
        if _sim_start >= _sim_end:
            continue

        _input_value = flex_select_1d_pc_nb(input_value, col)
        if _input_value != 0:
            out[col] = (market_value[_sim_end - 1, col] - _input_value) / _input_value
    return out
