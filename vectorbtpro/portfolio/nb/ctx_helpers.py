# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Numba-compiled context helper functions for portfolio simulation."""

from vectorbtpro.base.flex_indexing import flex_select_col_nb
from vectorbtpro.portfolio.nb import records as pf_records_nb
from vectorbtpro.portfolio.nb.core import *
from vectorbtpro.records import nb as records_nb


# ############# Position ############# #


@register_jitted
def get_col_position_nb(c: tp.NamedTuple, col: int) -> float:
    """Get position of a column."""
    return c.last_position[col]


@register_jitted
def get_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get position of the current column."""
    return get_col_position_nb(c, c.col)


@register_jitted
def col_in_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Check whether a column is in a position."""
    position = get_col_position_nb(c, col)
    return position != 0


@register_jitted
def in_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether the current column is in a position."""
    return col_in_position_nb(c, c.col)


@register_jitted
def col_in_long_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Check whether a column is in a long position."""
    position = get_col_position_nb(c, col)
    return position > 0


@register_jitted
def in_long_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether the current column is in a long position."""
    return col_in_long_position_nb(c, c.col)


@register_jitted
def col_in_short_position_nb(c: tp.NamedTuple, col: int) -> bool:
    """Check whether a column is in a short position."""
    position = get_col_position_nb(c, col)
    return position < 0


@register_jitted
def in_short_position_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether the current column is in a short position."""
    return col_in_short_position_nb(c, c.col)


@register_jitted
def get_n_active_positions_nb(
    c: tp.Union[
        GroupContext,
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
        PostSignalContext,
    ],
    all_groups: bool = False,
) -> int:
    """Get the number of active positions in the current group (regardless of cash sharing).

    To calculate across all groups, set `all_groups` to True."""
    n_active_positions = 0
    if all_groups:
        for col in range(c.target_shape[1]):
            if c.last_position[col] != 0:
                n_active_positions += 1
    else:
        for col in range(c.from_col, c.to_col):
            if c.last_position[col] != 0:
                n_active_positions += 1
    return n_active_positions


# ############# Cash ############# #


@register_jitted
def get_col_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Get cash of a column."""
    if c.cash_sharing:
        raise ValueError(
            "Cannot get cash of a single column from a group with cash sharing. "
            "Use get_group_cash_nb."
        )
    return c.last_cash[col]


@register_jitted
def get_group_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Get cash of a group."""
    if c.cash_sharing:
        return c.last_cash[group]
    cash = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                cash += c.last_cash[col]
            break
        from_col = to_col
    return cash


@register_jitted
def get_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return get_group_cash_nb(c, c.group)
    return get_col_cash_nb(c, c.col)


# ############# Debt ############# #


@register_jitted
def get_col_debt_nb(c: tp.NamedTuple, col: int) -> float:
    """Get debt of a column."""
    return c.last_debt[col]


@register_jitted
def get_debt_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get debt of the current column."""
    return get_col_debt_nb(c, c.col)


# ############# Locked cash ############# #


@register_jitted
def get_col_locked_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Get locked cash of a column."""
    return c.last_locked_cash[col]


@register_jitted
def get_locked_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get locked cash of the current column."""
    return get_col_locked_cash_nb(c, c.col)


# ############# Free cash ############# #


@register_jitted
def get_col_free_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Get free cash of a column."""
    if c.cash_sharing:
        raise ValueError(
            "Cannot get free cash of a single column from a group with cash sharing. "
            "Use get_group_free_cash_nb."
        )
    return c.last_free_cash[col]


@register_jitted
def get_group_free_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Get free cash of a group."""
    if c.cash_sharing:
        return c.last_free_cash[group]
    free_cash = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                free_cash += c.last_free_cash[col]
            break
        from_col = to_col
    return free_cash


@register_jitted
def get_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get free cash of the current column or group with cash sharing."""
    if c.cash_sharing:
        return get_group_free_cash_nb(c, c.group)
    return get_col_free_cash_nb(c, c.col)


@register_jitted
def col_has_free_cash_nb(c: tp.NamedTuple, col: int) -> float:
    """Check whether a column has free cash."""
    return get_col_free_cash_nb(c, col) > 0


@register_jitted
def group_has_free_cash_nb(c: tp.NamedTuple, group: int) -> float:
    """Check whether a group has free cash."""
    return get_group_free_cash_nb(c, group) > 0


@register_jitted
def has_free_cash_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether the current column or group with cash sharing has free cash."""
    if c.cash_sharing:
        return group_has_free_cash_nb(c, c.group)
    return col_has_free_cash_nb(c, c.col)


# ############# Valuation price ############# #


@register_jitted
def get_col_val_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get valuation price of a column."""
    return c.last_val_price[col]


@register_jitted
def get_val_price_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get valuation price of the current column."""
    return get_col_val_price_nb(c, c.col)


# ############# Value ############# #


@register_jitted
def get_col_value_nb(c: tp.NamedTuple, col: int) -> float:
    """Get value of a column."""
    if c.cash_sharing:
        raise ValueError(
            "Cannot get value of a single column from a group with cash sharing. "
            "Use get_group_value_nb."
        )
    return c.last_value[col]


@register_jitted
def get_group_value_nb(c: tp.NamedTuple, group: int) -> float:
    """Get value of a group."""
    if c.cash_sharing:
        return c.last_value[group]
    value = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                value += c.last_value[col]
            break
        from_col = to_col
    return value


@register_jitted
def get_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get value of the current column or group with cash sharing."""
    if c.cash_sharing:
        return get_group_value_nb(c, c.group)
    return get_col_value_nb(c, c.col)


# ############# Leverage ############# #


@register_jitted
def get_col_leverage_nb(c: tp.NamedTuple, col: int) -> float:
    """Get leverage of a column."""
    position = get_col_position_nb(c, col)
    debt = get_col_debt_nb(c, col)
    locked_cash = get_col_locked_cash_nb(c, col)
    if locked_cash == 0:
        return np.nan
    leverage = debt / locked_cash
    if position > 0:
        leverage += 1
    return leverage


@register_jitted
def get_leverage_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get leverage of the current column."""
    return get_col_leverage_nb(c, c.col)


# ############# Allocation ############# #


@register_jitted
def get_col_position_value_nb(c: tp.NamedTuple, col: int) -> float:
    """Get position value of a column."""
    position = get_col_position_nb(c, col)
    val_price = get_col_val_price_nb(c, col)
    if position:
        return 0.0
    return position * val_price


@register_jitted
def get_group_position_value_nb(c: tp.NamedTuple, group: int) -> float:
    """Get position value of a group."""
    value = 0.0
    from_col = 0
    for g in range(len(c.group_lens)):
        to_col = from_col + c.group_lens[g]
        if g == group:
            for col in range(from_col, to_col):
                value += get_col_position_value_nb(c, col)
            break
        from_col = to_col
    return value


@register_jitted
def get_position_value_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get position value of the current column."""
    return get_col_position_value_nb(c, c.col)


@register_jitted
def get_col_allocation_nb(c: tp.NamedTuple, col: int, group: tp.Optional[int] = None) -> float:
    """Get allocation of a column in its group."""
    position_value = get_col_position_value_nb(c, col)
    if group is None:
        from_col = 0
        found = False
        for _group in range(len(c.group_lens)):
            to_col = from_col + c.group_lens[_group]
            if from_col <= col < to_col:
                found = True
                break
            from_col = to_col
        if not found:
            raise ValueError("Column out of bounds")
    else:
        _group = group
    value = get_group_value_nb(c, _group)
    if position_value == 0:
        return 0.0
    if value <= 0:
        return np.nan
    return position_value / value


@register_jitted
def get_allocation_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get allocation of the current column in the current group."""
    return get_col_allocation_nb(c, c.col, group=c.group)


# ############# Orders ############# #


@register_jitted
def get_col_order_count_nb(c: tp.NamedTuple, col: int) -> int:
    """Get number of order records for a column."""
    return c.order_counts[col]


@register_jitted
def get_order_count_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> int:
    """Get number of order records for the current column."""
    return get_col_order_count_nb(c, c.col)


@register_jitted
def get_col_order_records_nb(c: tp.NamedTuple, col: int) -> tp.RecordArray:
    """Get order records for a column."""
    order_count = get_col_order_count_nb(c, col)
    return c.order_records[:order_count, col]


@register_jitted
def get_order_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> tp.RecordArray:
    """Get order records for the current column."""
    return get_col_order_records_nb(c, c.col)


@register_jitted
def col_has_orders_nb(c: tp.NamedTuple, col: int) -> bool:
    """Check whether there is any order in a column."""
    return get_col_order_count_nb(c, col) > 0


@register_jitted
def has_orders_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> bool:
    """Check whether there is any order in the current column."""
    return col_has_orders_nb(c, c.col)


@register_jitted
def get_col_last_order_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get the last order in a column."""
    if not col_has_orders_nb(c, col):
        raise ValueError("There are no orders. Check for any orders first.")
    return get_col_order_records_nb(c, col)[-1]


@register_jitted
def get_last_order_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get the last order in the current column."""
    return get_col_last_order_nb(c, c.col)


# ############# Order result ############# #


@register_jitted
def order_filled_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order was filled."""
    return c.order_result.status == OrderStatus.Filled


@register_jitted
def order_opened_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has opened a new position."""
    position_now = get_position_nb(c)
    return order_reversed_position_nb(c) or (c.position_before == 0 and position_now != 0)


@register_jitted
def order_increased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has opened or increased an existing position."""
    position_now = get_position_nb(c)
    return order_opened_position_nb(c) or (
        np.sign(position_now) == np.sign(c.position_before) and abs(position_now) > abs(c.position_before)
    )


@register_jitted
def order_decreased_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has decreased or closed an existing position."""
    position_now = get_position_nb(c)
    return (
        order_closed_position_nb(c)
        or order_reversed_position_nb(c)
        or (np.sign(position_now) == np.sign(c.position_before) and abs(position_now) < abs(c.position_before))
    )


@register_jitted
def order_closed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has closed out an existing position."""
    position_now = get_position_nb(c)
    return c.position_before != 0 and position_now == 0


@register_jitted
def order_reversed_position_nb(
    c: tp.Union[
        PostOrderContext,
        PostSignalContext,
    ]
) -> bool:
    """Check whether the order has reversed an existing position."""
    position_now = get_position_nb(c)
    return c.position_before != 0 and position_now != 0 and np.sign(c.position_before) != np.sign(position_now)


# ############# Limit orders ############# #


@register_jitted
def get_col_limit_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get limit order information of a column."""
    return c.last_limit_info[col]


@register_jitted
def get_limit_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get limit order information of the current column."""
    return get_col_limit_info_nb(c, c.col)


@register_jitted
def get_col_limit_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of limit order in a column."""
    if not col_in_position_nb(c, col):
        return np.nan
    limit_info = get_col_limit_info_nb(c, col)
    return get_limit_info_target_price_nb(limit_info)


@register_jitted
def get_limit_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of limit order in the current column."""
    return get_col_limit_target_price_nb(c, c.col)


# ############# Stop orders ############# #


@register_jitted
def get_col_sl_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get SL order information of a column."""
    return c.last_sl_info[col]


@register_jitted
def get_sl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get SL order information of the current column."""
    return get_col_sl_info_nb(c, c.col)


@register_jitted
def get_col_sl_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of SL order in a column."""
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    sl_info = get_col_sl_info_nb(c, col)
    return get_sl_info_target_price_nb(sl_info, position)


@register_jitted
def get_sl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of SL order in the current column."""
    return get_col_sl_target_price_nb(c, c.col)


@register_jitted
def get_col_tsl_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get TSL/TTP order information of a column."""
    return c.last_tsl_info[col]


@register_jitted
def get_tsl_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get TSL/TTP order information of the current column."""
    return get_col_tsl_info_nb(c, c.col)


@register_jitted
def get_col_tsl_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of TSL/TTP order in a column."""
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    tsl_info = get_col_tsl_info_nb(c, col)
    return get_tsl_info_target_price_nb(tsl_info, position)


@register_jitted
def get_tsl_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of TSL/TTP order in the current column."""
    return get_col_tsl_target_price_nb(c, c.col)


@register_jitted
def get_col_tp_info_nb(c: tp.NamedTuple, col: int) -> tp.Record:
    """Get TP order information of a column."""
    return c.last_tp_info[col]


@register_jitted
def get_tp_info_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> tp.Record:
    """Get TP order information of the current column."""
    return get_col_tp_info_nb(c, c.col)


@register_jitted
def get_col_tp_target_price_nb(c: tp.NamedTuple, col: int) -> float:
    """Get target price of TP order in a column."""
    if not col_in_position_nb(c, col):
        return np.nan
    position = get_col_position_nb(c, col)
    tp_info = get_col_tp_info_nb(c, col)
    return get_tp_info_target_price_nb(tp_info, position)


@register_jitted
def get_tp_target_price_nb(
    c: tp.Union[
        SignalContext,
        PostSignalContext,
    ],
) -> float:
    """Get target price of TP order in the current column."""
    return get_col_tp_target_price_nb(c, c.col)


# ############# Trades ############# #


@register_jitted
def get_col_entry_trade_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get entry trade records of a column up to this point."""
    order_records = get_col_order_records_nb(c, col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, col)
    entry_trades = pf_records_nb.get_entry_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return entry_trades


@register_jitted
def get_entry_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get entry trade records of the current column up to this point."""
    return get_col_entry_trade_records_nb(c, c.col, init_position=init_position, init_price=init_price)


@register_jitted
def get_col_exit_trade_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get exit trade records of a column up to this point."""
    order_records = get_col_order_records_nb(c, col)
    col_map = records_nb.col_map_nb(order_records["col"], c.target_shape[1])
    close = flex_select_col_nb(c.close, col)
    exit_trades = pf_records_nb.get_exit_trades_nb(
        order_records,
        close[: c.i + 1],
        col_map,
        init_position=init_position,
        init_price=init_price,
    )
    return exit_trades


@register_jitted
def get_exit_trade_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get exit trade records of the current column up to this point."""
    return get_col_exit_trade_records_nb(c, c.col, init_position=init_position, init_price=init_price)


@register_jitted
def get_col_position_records_nb(
    c: tp.NamedTuple,
    col: int,
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get position records of a column up to this point."""
    exit_trade_records = get_col_exit_trade_records_nb(c, col, init_position=init_position, init_price=init_price)
    col_map = records_nb.col_map_nb(exit_trade_records["col"], c.target_shape[1])
    position_records = pf_records_nb.get_positions_nb(exit_trade_records, col_map)
    return position_records


@register_jitted
def get_position_records_nb(
    c: tp.Union[
        OrderContext,
        PostOrderContext,
        SignalContext,
        PostSignalContext,
    ],
    init_position: tp.FlexArray1dLike = 0.0,
    init_price: tp.FlexArray1dLike = np.nan,
) -> tp.Array1d:
    """Get position records of the current column up to this point."""
    return get_col_position_records_nb(c, c.col, init_position=init_position, init_price=init_price)


# ############# Simulation ############# #


@register_jitted
def stop_group_sim_nb(c: tp.NamedTuple, group: int) -> None:
    """Stop the simulation of a group."""
    c.sim_end[group] = c.i + 1


@register_jitted
def stop_sim_nb(
    c: tp.Union[
        SegmentContext,
        OrderContext,
        PostOrderContext,
        FlexOrderContext,
        SignalSegmentContext,
        SignalContext,
        PostSignalContext,
    ],
) -> None:
    """Stop the simulation of the current group."""
    stop_group_sim_nb(c, c.group)
