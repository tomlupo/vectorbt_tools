# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Core Numba-compiled functions for portfolio simulation."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.flex_indexing import flex_select_1d_pc_nb, flex_select_nb
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.portfolio.enums import *
from vectorbtpro.registries.jit_registry import register_jitted
from vectorbtpro.utils.math_ import is_close_nb, is_close_or_less_nb, is_close_or_greater_nb, is_less_nb, add_nb


@register_jitted(cache=True)
def order_not_filled_nb(status: int, status_info: int) -> OrderResult:
    """Return `OrderResult` for order that hasn't been filled."""
    return OrderResult(size=np.nan, price=np.nan, fees=np.nan, side=-1, status=status, status_info=status_info)


@register_jitted(cache=True)
def check_adj_price_nb(
    adj_price: float,
    price_area: PriceArea,
    is_closing_price: bool,
    price_area_vio_mode: int,
) -> float:
    """Check whether adjusted price is within price boundaries."""
    if price_area_vio_mode == PriceAreaVioMode.Ignore:
        return adj_price
    if adj_price > price_area.high:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is above the highest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.high
    if adj_price < price_area.low:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is below the lowest price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.low
    if is_closing_price and adj_price != price_area.close:
        if price_area_vio_mode == PriceAreaVioMode.Error:
            raise ValueError("Adjusted order price is beyond the closing price")
        elif price_area_vio_mode == PriceAreaVioMode.Cap:
            adj_price = price_area.close
    return adj_price


@register_jitted(cache=True)
def approx_long_buy_value_nb(val_price: float, size: float) -> float:
    """Approximate value of a long-buy operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def adj_size_granularity_nb(size: float, size_granularity: float) -> bool:
    """Whether to adjust the size with the size granularity."""
    if size_granularity % 1 == 0:
        return True
    adj_size = size // size_granularity * size_granularity
    return not is_close_nb(size, adj_size) and not is_close_nb(size, adj_size + size_granularity)


@register_jitted(cache=True)
def cast_account_state_nb(account_state: AccountState) -> AccountState:
    """Cast account state to float."""
    return AccountState(
        cash=float(account_state.cash),
        position=float(account_state.position),
        debt=float(account_state.debt),
        locked_cash=float(account_state.locked_cash),
        free_cash=float(account_state.free_cash),
    )


@register_jitted(cache=True)
def long_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a long position."""
    _account_state = cast_account_state_nb(account_state)

    # Get cash limit
    cash_limit = _account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state
    cash_limit = cash_limit * leverage

    # Adjust for max size
    if not np.isnan(max_size) and size > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size = max_size
    if np.isinf(size) and np.isinf(cash_limit):
        raise ValueError("Attempt to go in long direction infinitely")

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size, size_granularity):
        size = size // size_granularity * size_granularity

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if not np.isnan(size_granularity) and adj_size_granularity_nb(max_acq_size, size_granularity):
            final_size = max_acq_size // size_granularity * size_granularity
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check against size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size) and is_less_nb(final_size, size) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(final_size),
        float(adj_price),
        float(fees_paid),
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(_account_state.cash, -req_cash)
    new_position = add_nb(_account_state.position, final_size)
    if leverage_mode == LeverageMode.Lazy:
        debt_diff = max(add_nb(req_cash, -_account_state.free_cash), 0.0)
        if debt_diff > 0:
            new_debt = _account_state.debt + debt_diff
            new_locked_cash = _account_state.locked_cash + _account_state.free_cash
            new_free_cash = 0.0
        else:
            new_debt = _account_state.debt
            new_locked_cash = _account_state.locked_cash
            new_free_cash = add_nb(_account_state.free_cash, -req_cash)
    else:
        if leverage > 1:
            if np.isinf(leverage):
                raise ValueError("Leverage must be finite for LeverageMode.Eager")
            order_value = final_size * adj_price
            new_debt = _account_state.debt + order_value * (leverage - 1) / leverage
            new_locked_cash = _account_state.locked_cash + order_value / leverage
            new_free_cash = add_nb(_account_state.free_cash, -order_value / leverage - fees_paid)
        else:
            new_debt = _account_state.debt
            new_locked_cash = _account_state.locked_cash
            new_free_cash = add_nb(_account_state.free_cash, -req_cash)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_long_sell_value_nb(position: float, debt: float, val_price: float, size: float) -> float:
    """Approximate value of a long-sell operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(position, abs(size))
    order_value = size_limit * val_price
    size_fraction = size_limit / position
    released_debt = size_fraction * debt
    add_free_cash = order_value - released_debt
    return -add_free_cash


@register_jitted(cache=True)
def long_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a long position."""
    _account_state = cast_account_state_nb(account_state)

    # Check for open position
    if _account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), _account_state

    # Get size limit
    size_limit = min(_account_state.position, size)
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get acquired cash
    acq_cash = size_limit * adj_price

    # Update fees
    fees_paid = acq_cash * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(acq_cash, -fees_paid)
    if final_acq_cash < 0 and is_less_nb(_account_state.free_cash, -final_acq_cash):
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(size_limit),
        float(adj_price),
        float(fees_paid),
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = _account_state.cash + final_acq_cash
    new_position = add_nb(_account_state.position, -size_limit)
    new_pos_fraction = abs(new_position) / abs(_account_state.position)
    new_debt = new_pos_fraction * _account_state.debt
    new_locked_cash = new_pos_fraction * _account_state.locked_cash
    size_fraction = size_limit / _account_state.position
    released_debt = size_fraction * _account_state.debt
    new_free_cash = add_nb(_account_state.free_cash, final_acq_cash - released_debt)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_sell_value_nb(val_price: float, size: float) -> float:
    """Approximate value of a short-sell operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0:
        return 0.0
    order_value = abs(size) * val_price
    add_free_cash = -order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Open or increase a short position."""
    _account_state = cast_account_state_nb(account_state)

    # Get cash limit
    cash_limit = _account_state.free_cash
    if not np.isnan(percent):
        cash_limit = cash_limit * percent
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state
    cash_limit = cash_limit * leverage

    # Get price adjusted with slippage
    adj_price = price * (1 - slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get size limit
    fees_adj_price = adj_price * (1 + fees)
    if fees_adj_price == 0:
        max_size_limit = np.inf
    else:
        max_size_limit = add_nb(cash_limit, -fixed_fees) / (adj_price * (1 + fees))
    size_limit = min(size, max_size_limit)
    if size_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size
    if np.isinf(size_limit):
        raise ValueError("Attempt to go in short direction infinitely")

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Check against size of zero
    if is_close_nb(size_limit, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(size_limit, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill
    if np.isfinite(size) and is_less_nb(size_limit, size) and not allow_partial:  # np.inf doesn't count
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Get acquired cash
    order_value = size_limit * adj_price

    # Update fees
    fees_paid = order_value * fees + fixed_fees

    # Get final cash by subtracting costs
    final_acq_cash = add_nb(order_value, -fees_paid)
    if final_acq_cash < 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(size_limit),
        float(adj_price),
        float(fees_paid),
        OrderSide.Sell,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = _account_state.cash + final_acq_cash
    new_position = _account_state.position - size_limit
    new_debt = _account_state.debt + order_value
    if np.isinf(leverage):
        if np.isinf(_account_state.free_cash):
            raise ValueError("Leverage must be finite when _account_state.free_cash is infinite")
        if is_close_or_less_nb(_account_state.free_cash, fees_paid):
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state
        leverage_ = order_value / (_account_state.free_cash - fees_paid)
    else:
        leverage_ = float(leverage)
    new_locked_cash = _account_state.locked_cash + order_value / leverage_
    new_free_cash = add_nb(_account_state.free_cash, -order_value / leverage_ - fees_paid)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_short_buy_value_nb(position: float, debt: float, locked_cash: float, val_price: float, size: float) -> float:
    """Approximate value of a short-buy operation.

    Positive value means spending (for sorting reasons)."""
    if size == 0 or position == 0:
        return 0.0
    size_limit = min(abs(position), abs(size))
    order_value = size_limit * val_price
    size_fraction = size_limit / abs(position)
    released_debt = size_fraction * debt
    released_cash = size_fraction * locked_cash
    add_free_cash = released_cash + released_debt - order_value
    return -add_free_cash


@register_jitted(cache=True)
def short_buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Decrease or close a short position."""
    _account_state = cast_account_state_nb(account_state)

    # Check for open position
    if _account_state.position == 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoOpenPosition), _account_state

    # Get cash limit
    cash_limit = _account_state.free_cash + _account_state.debt + _account_state.locked_cash
    if cash_limit <= 0:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.NoCash), _account_state

    # Get size limit
    size_limit = min(abs(_account_state.position), size)
    if not np.isnan(percent):
        size_limit = size_limit * percent

    # Adjust for max size
    if not np.isnan(max_size) and size_limit > max_size:
        if not allow_partial:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.MaxSizeExceeded), _account_state

        size_limit = max_size

    # Adjust for granularity
    if not np.isnan(size_granularity) and adj_size_granularity_nb(size_limit, size_granularity):
        size_limit = size_limit // size_granularity * size_granularity

    # Get price adjusted with slippage
    adj_price = price * (1 + slippage)
    adj_price = check_adj_price_nb(adj_price, price_area, is_closing_price, price_area_vio_mode)

    # Get cash required to complete this order
    if np.isinf(size_limit):
        req_cash = np.inf
        req_fees = np.inf
    else:
        order_value = size_limit * adj_price
        req_fees = order_value * fees + fixed_fees
        req_cash = order_value + req_fees

    if is_close_or_less_nb(req_cash, cash_limit):
        # Sufficient amount of cash
        final_size = size_limit
        fees_paid = req_fees
    else:
        # Insufficient amount of cash, size will be less than requested

        # For fees of 10% and 1$ per transaction, you can buy for 90$ (new_req_cash)
        # to spend 100$ (cash_limit) in total
        max_req_cash = add_nb(cash_limit, -fixed_fees) / (1 + fees)
        if max_req_cash <= 0:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.CantCoverFees), _account_state

        max_acq_size = max_req_cash / adj_price

        # Adjust for granularity
        if not np.isnan(size_granularity) and adj_size_granularity_nb(max_acq_size, size_granularity):
            final_size = max_acq_size // size_granularity * size_granularity
            new_order_value = final_size * adj_price
            fees_paid = new_order_value * fees + fixed_fees
            req_cash = new_order_value + fees_paid
        else:
            final_size = max_acq_size
            fees_paid = cash_limit - max_req_cash
            req_cash = cash_limit

    # Check size of zero
    if is_close_nb(final_size, 0):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeZero), _account_state

    # Check against minimum size
    if not np.isnan(min_size) and is_less_nb(final_size, min_size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), _account_state

    # Check against partial fill (np.inf doesn't count)
    if np.isfinite(size_limit) and is_less_nb(final_size, size_limit) and not allow_partial:
        return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.PartialFill), _account_state

    # Create a filled order
    order_result = OrderResult(
        float(final_size),
        float(adj_price),
        float(fees_paid),
        OrderSide.Buy,
        OrderStatus.Filled,
        -1,
    )

    # Update the current account state
    new_cash = add_nb(_account_state.cash, -req_cash)
    new_position = add_nb(_account_state.position, final_size)
    new_pos_fraction = abs(new_position) / abs(_account_state.position)
    new_debt = new_pos_fraction * _account_state.debt
    new_locked_cash = new_pos_fraction * _account_state.locked_cash
    size_fraction = final_size / abs(_account_state.position)
    released_debt = size_fraction * _account_state.debt
    released_cash = size_fraction * _account_state.locked_cash
    new_free_cash = add_nb(_account_state.free_cash, released_cash + released_debt - req_cash)
    new_account_state = AccountState(
        cash=float(new_cash),
        position=float(new_position),
        debt=float(new_debt),
        locked_cash=float(new_locked_cash),
        free_cash=float(new_free_cash),
    )
    return order_result, new_account_state


@register_jitted(cache=True)
def approx_buy_value_nb(
    position: float,
    debt: float,
    locked_cash: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate value of a buy operation.

    Positive value means spending (for sorting reasons)."""
    if position <= 0 and direction == Direction.ShortOnly:
        return approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    if position >= 0:
        return approx_long_buy_value_nb(val_price, size)
    value1 = approx_short_buy_value_nb(position, debt, locked_cash, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_long_buy_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def buy_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Buy."""
    _account_state = cast_account_state_nb(account_state)

    if _account_state.position <= 0 and direction == Direction.ShortOnly:
        return short_buy_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if _account_state.position >= 0:
        return long_buy_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            leverage_mode=leverage_mode,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if not np.isnan(min_size):
        min_size1 = min(min_size, abs(_account_state.position))
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, abs(_account_state.position))
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = short_buy_nb(
        account_state=_account_state,
        size=size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, _account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(_account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - abs(_account_state.position), 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - abs(_account_state.position), 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = long_buy_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, _account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def approx_sell_value_nb(
    position: float,
    debt: float,
    val_price: float,
    size: float,
    direction: int,
) -> float:
    """Approximate value of a sell operation.

    Positive value means spending (for sorting reasons)."""
    if position >= 0 and direction == Direction.LongOnly:
        return approx_long_sell_value_nb(position, debt, val_price, size)
    if position <= 0:
        return approx_short_sell_value_nb(val_price, size)
    value1 = approx_long_sell_value_nb(position, debt, val_price, size)
    new_size = add_nb(size, -abs(position))
    if new_size <= 0:
        return value1
    value2 = approx_short_sell_value_nb(val_price, new_size)
    return value1 + value2


@register_jitted(cache=True)
def sell_nb(
    account_state: AccountState,
    size: float,
    price: float,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    percent: float = np.nan,
    price_area: PriceArea = NoPriceArea,
    is_closing_price: bool = False,
) -> tp.Tuple[OrderResult, AccountState]:
    """Sell."""
    _account_state = cast_account_state_nb(account_state)

    if _account_state.position >= 0 and direction == Direction.LongOnly:
        return long_sell_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if _account_state.position <= 0:
        return short_sell_nb(
            account_state=_account_state,
            size=size,
            price=price,
            fees=fees,
            fixed_fees=fixed_fees,
            slippage=slippage,
            min_size=min_size,
            max_size=max_size,
            size_granularity=size_granularity,
            leverage=leverage,
            price_area_vio_mode=price_area_vio_mode,
            allow_partial=allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    if not np.isnan(min_size):
        min_size1 = min(min_size, _account_state.position)
    else:
        min_size1 = np.nan
    if not np.isnan(max_size):
        max_size1 = min(max_size, _account_state.position)
    else:
        max_size1 = np.nan
    new_order_result1, new_account_state1 = long_sell_nb(
        account_state=_account_state,
        size=size,
        price=price,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size1,
        max_size=max_size1,
        size_granularity=size_granularity,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=np.nan,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result1.status != OrderStatus.Filled:
        return new_order_result1, _account_state
    if new_account_state1.position != 0:
        return new_order_result1, new_account_state1
    new_size = add_nb(size, -abs(_account_state.position))
    if new_size <= 0:
        return new_order_result1, new_account_state1
    if not np.isnan(min_size):
        min_size2 = max(min_size - _account_state.position, 0.0)
    else:
        min_size2 = np.nan
    if not np.isnan(max_size):
        max_size2 = max(max_size - _account_state.position, 0.0)
    else:
        max_size2 = np.nan
    new_order_result2, new_account_state2 = short_sell_nb(
        account_state=new_account_state1,
        size=new_size,
        price=price,
        fees=fees,
        fixed_fees=0.0,
        slippage=slippage,
        min_size=min_size2,
        max_size=max_size2,
        size_granularity=size_granularity,
        leverage=leverage,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        percent=percent,
        price_area=price_area,
        is_closing_price=is_closing_price,
    )
    if new_order_result2.status != OrderStatus.Filled:
        if allow_partial or np.isinf(new_size):
            if new_order_result2.status_info == OrderStatusInfo.SizeZero:
                return new_order_result1, new_account_state1
            if new_order_result2.status_info == OrderStatusInfo.NoCash:
                return new_order_result1, new_account_state1
        return new_order_result2, _account_state
    new_order_result = OrderResult(
        new_order_result1.size + new_order_result2.size,
        new_order_result2.price,
        new_order_result1.fees + new_order_result2.fees,
        new_order_result2.side,
        new_order_result2.status,
        new_order_result2.status_info,
    )
    return new_order_result, new_account_state2


@register_jitted(cache=True)
def update_value_nb(
    cash_before: float,
    cash_now: float,
    position_before: float,
    position_now: float,
    val_price_before: float,
    price: float,
    value_before: float,
) -> tp.Tuple[float, float]:
    """Update valuation price and value."""
    val_price_now = price
    cash_flow = cash_now - cash_before
    if position_before != 0:
        asset_value_before = position_before * val_price_before
    else:
        asset_value_before = 0.0
    if position_now != 0:
        asset_value_now = position_now * val_price_now
    else:
        asset_value_now = 0.0
    asset_value_diff = asset_value_now - asset_value_before
    value_now = value_before + cash_flow + asset_value_diff
    return val_price_now, value_now


@register_jitted(cache=True)
def get_diraware_size_nb(size: float, direction: int) -> float:
    """Get direction-aware size."""
    if direction == Direction.ShortOnly:
        return size * -1
    return size


@register_jitted(cache=True)
def resolve_size_nb(
    size: float,
    size_type: int,
    position: float,
    val_price: float,
    value: float,
    target_size_type: int = SizeType.Amount,
    as_requirement: bool = False,
) -> tp.Tuple[float, float]:
    """Resolve size into an absolute amount of assets and percentage of resources.

    Percentage is only set if the option `SizeType.Percent(100)` is used."""
    percent = np.nan
    if size_type == target_size_type:
        return float(size), percent

    if size_type == SizeType.ValuePercent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.ValuePercent

    if size_type == SizeType.TargetPercent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.TargetPercent

    if size_type == SizeType.ValuePercent or size_type == SizeType.TargetPercent:
        if size_type == target_size_type:
            return float(size), percent

        size *= value
        if size_type == SizeType.ValuePercent:
            size_type = SizeType.Value
        else:
            size_type = SizeType.TargetValue

    if size_type == SizeType.Value or size_type == SizeType.TargetValue:
        if size_type == target_size_type:
            return float(size), percent

        size /= val_price
        if size_type == SizeType.Value:
            size_type = SizeType.Amount
        else:
            size_type = SizeType.TargetAmount

    if size_type == SizeType.TargetAmount:
        if size_type == target_size_type:
            return float(size), percent

        if not as_requirement:
            size -= position
        size_type = SizeType.Amount

    if size_type == SizeType.Percent100:
        if size_type == target_size_type:
            return float(size), percent

        size /= 100
        size_type = SizeType.Percent

    if size_type == SizeType.Percent:
        if size_type == target_size_type:
            return float(size), percent

        percent = abs(size)
        if as_requirement:
            size = np.nan
        else:
            size = np.sign(size) * np.inf
        size_type = SizeType.Amount

    if size_type != target_size_type:
        raise ValueError("Cannot convert size to target size type")
    if as_requirement:
        size = abs(size)
    return float(size), percent


@register_jitted(cache=True)
def approx_order_value_nb(
    exec_state: ExecState,
    size: float,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
) -> float:
    """Approximate the value of an order.

    Assumes that cash is infinite.

    Positive value means spending (for sorting reasons)."""
    size = get_diraware_size_nb(float(size), direction)
    amount_size, _ = resolve_size_nb(
        size=size,
        size_type=size_type,
        position=exec_state.position,
        val_price=exec_state.val_price,
        value=exec_state.value,
    )
    if amount_size >= 0:
        order_value = approx_buy_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            locked_cash=exec_state.locked_cash,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    else:
        order_value = approx_sell_value_nb(
            position=exec_state.position,
            debt=exec_state.debt,
            val_price=exec_state.val_price,
            size=abs(amount_size),
            direction=direction,
        )
    return order_value


@register_jitted(cache=True)
def execute_order_nb(
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
) -> tp.Tuple[OrderResult, ExecState]:
    """Execute an order given the current state.

    Args:
        exec_state (ExecState): See `vectorbtpro.portfolio.enums.ExecState`.
        order (Order): See `vectorbtpro.portfolio.enums.Order`.
        price_area (OrderPriceArea): See `vectorbtpro.portfolio.enums.PriceArea`.
        update_value (bool): Whether to update the value.

    Error is thrown if an input has value that is not expected.
    Order is ignored if its execution has no effect on the current balance.
    Order is rejected if an input goes over a limit or against a restriction.
    """
    # numerical stability
    cash = float(exec_state.cash)
    if is_close_nb(cash, 0):
        cash = 0.0
    position = float(exec_state.position)
    if is_close_nb(position, 0):
        position = 0.0
    debt = float(exec_state.debt)
    if is_close_nb(debt, 0):
        debt = 0.0
    locked_cash = float(exec_state.locked_cash)
    if is_close_nb(locked_cash, 0):
        locked_cash = 0.0
    free_cash = float(exec_state.free_cash)
    if is_close_nb(free_cash, 0):
        free_cash = 0.0
    val_price = float(exec_state.val_price)
    if is_close_nb(val_price, 0):
        val_price = 0.0
    value = float(exec_state.value)
    if is_close_nb(value, 0):
        value = 0.0

    # Pre-fill account state
    account_state = AccountState(
        cash=cash,
        position=position,
        debt=debt,
        locked_cash=locked_cash,
        free_cash=free_cash,
    )

    # Check price area
    if np.isinf(price_area.open) or price_area.open < 0:
        raise ValueError("price_area.open must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.high) or price_area.high < 0:
        raise ValueError("price_area.high must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.low) or price_area.low < 0:
        raise ValueError("price_area.low must be either NaN, or finite and 0 or greater")
    if np.isinf(price_area.close) or price_area.close < 0:
        raise ValueError("price_area.close must be either NaN, or finite and 0 or greater")

    # Resolve price
    order_price = order.price
    is_closing_price = False
    if np.isinf(order_price):
        if order_price > 0:
            order_price = price_area.close
            is_closing_price = True
        else:
            order_price = price_area.open
    elif order_price == PriceType.NextOpen:
        raise ValueError("Next open must be handled higher in the stack")
    elif order_price == PriceType.NextClose:
        raise ValueError("Next close must be handled higher in the stack")

    # Ignore order if size or price is nan
    if np.isnan(order.size):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.SizeNaN), exec_state
    if np.isnan(order_price):
        return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.PriceNaN), exec_state

    # Check account state
    if np.isnan(cash):
        raise ValueError("exec_state.cash cannot be NaN")
    if not np.isfinite(position):
        raise ValueError("exec_state.position must be finite")
    if not np.isfinite(debt) or debt < 0:
        raise ValueError("exec_state.debt must be finite and 0 or greater")
    if not np.isfinite(locked_cash) or locked_cash < 0:
        raise ValueError("exec_state.locked_cash must be finite and 0 or greater")
    if np.isnan(free_cash):
        raise ValueError("exec_state.free_cash cannot be NaN")

    # Check order
    if not np.isfinite(order_price) or order_price < 0:
        raise ValueError("order.price must be finite and 0 or greater")
    if order.size_type < 0 or order.size_type >= len(SizeType):
        raise ValueError("order.size_type is invalid")
    if order.direction < 0 or order.direction >= len(Direction):
        raise ValueError("order.direction is invalid")
    if not np.isfinite(order.fees):
        raise ValueError("order.fees must be finite")
    if not np.isfinite(order.fixed_fees):
        raise ValueError("order.fixed_fees must be finite")
    if not np.isfinite(order.slippage) or order.slippage < 0:
        raise ValueError("order.slippage must be finite and 0 or greater")
    if np.isinf(order.min_size) or order.min_size < 0:
        raise ValueError("order.min_size must be either NaN, 0, or greater")
    if order.max_size <= 0:
        raise ValueError("order.max_size must be either NaN or greater than 0")
    if np.isinf(order.size_granularity) or order.size_granularity <= 0:
        raise ValueError("order.size_granularity must be either NaN, or finite and greater than 0")
    if np.isnan(order.leverage) or order.leverage <= 0:
        raise ValueError("order.leverage must be greater than 0")
    if order.leverage_mode < 0 or order.leverage_mode >= len(LeverageMode):
        raise ValueError("order.leverage_mode is invalid")
    if not np.isfinite(order.reject_prob) or order.reject_prob < 0 or order.reject_prob > 1:
        raise ValueError("order.reject_prob must be between 0 and 1")

    # Positive/negative size in short direction should be treated as negative/positive
    order_size = get_diraware_size_nb(order.size, order.direction)
    min_order_size = order.min_size
    max_order_size = order.max_size
    order_size_type = order.size_type

    if (
        order_size_type == SizeType.ValuePercent100
        or order_size_type == SizeType.ValuePercent
        or order_size_type == SizeType.TargetPercent100
        or order_size_type == SizeType.TargetPercent
        or order_size_type == SizeType.Value
        or order_size_type == SizeType.TargetValue
    ):
        if np.isinf(val_price) or val_price <= 0:
            raise ValueError("val_price_now must be finite and greater than 0")
        if np.isnan(val_price):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValPriceNaN), exec_state
        if (
            order_size_type == SizeType.ValuePercent100
            or order_size_type == SizeType.ValuePercent
            or order_size_type == SizeType.TargetPercent100
            or order_size_type == SizeType.TargetPercent
        ):
            if np.isnan(value):
                return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.ValueNaN), exec_state
            if value <= 0:
                return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.ValueZeroNeg), exec_state

    order_size, percent = resolve_size_nb(
        size=order_size,
        size_type=order_size_type,
        position=position,
        val_price=val_price,
        value=value,
    )
    if not np.isnan(min_order_size):
        min_order_size, min_percent = resolve_size_nb(
            size=min_order_size,
            size_type=order_size_type,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(min_percent) and is_less_nb(percent, min_percent):
            return order_not_filled_nb(OrderStatus.Ignored, OrderStatusInfo.MinSizeNotReached), exec_state
    if not np.isnan(max_order_size):
        max_order_size, max_percent = resolve_size_nb(
            size=max_order_size,
            size_type=order_size_type,
            position=position,
            val_price=val_price,
            value=value,
            as_requirement=True,
        )
        if not np.isnan(percent) and not np.isnan(max_percent) and is_less_nb(max_percent, percent):
            percent = max_percent

    if order_size >= 0:
        order_result, new_account_state = buy_nb(
            account_state=account_state,
            size=order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            leverage_mode=order.leverage_mode,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )
    else:
        order_result, new_account_state = sell_nb(
            account_state=account_state,
            size=-order_size,
            price=order_price,
            direction=order.direction,
            fees=order.fees,
            fixed_fees=order.fixed_fees,
            slippage=order.slippage,
            min_size=min_order_size,
            max_size=max_order_size,
            size_granularity=order.size_granularity,
            leverage=order.leverage,
            price_area_vio_mode=order.price_area_vio_mode,
            allow_partial=order.allow_partial,
            percent=percent,
            price_area=price_area,
            is_closing_price=is_closing_price,
        )

    if order.reject_prob > 0:
        if np.random.uniform(0, 1) < order.reject_prob:
            return order_not_filled_nb(OrderStatus.Rejected, OrderStatusInfo.RandomEvent), exec_state

    if order_result.status == OrderStatus.Rejected and order.raise_reject:
        raise_rejected_order_nb(order_result)

    is_filled = order_result.status == OrderStatus.Filled
    if is_filled and update_value:
        new_val_price, new_value = update_value_nb(
            cash,
            new_account_state.cash,
            position,
            new_account_state.position,
            val_price,
            order_result.price,
            value,
        )
    else:
        new_val_price = val_price
        new_value = value

    new_exec_state = ExecState(
        cash=new_account_state.cash,
        position=new_account_state.position,
        debt=new_account_state.debt,
        locked_cash=new_account_state.locked_cash,
        free_cash=new_account_state.free_cash,
        val_price=new_val_price,
        value=new_value,
    )

    return order_result, new_exec_state


@register_jitted(cache=True)
def fill_log_record_nb(
    records: tp.RecordArray2d,
    r: int,
    group: int,
    col: int,
    i: int,
    price_area: PriceArea,
    exec_state: ExecState,
    order: Order,
    order_result: OrderResult,
    new_exec_state: ExecState,
    order_id: int,
) -> None:
    """Fill a log record."""

    records["id"][r, col] = r
    records["group"][r, col] = group
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["price_area_open"][r, col] = price_area.open
    records["price_area_high"][r, col] = price_area.high
    records["price_area_low"][r, col] = price_area.low
    records["price_area_close"][r, col] = price_area.close
    records["st0_cash"][r, col] = exec_state.cash
    records["st0_position"][r, col] = exec_state.position
    records["st0_debt"][r, col] = exec_state.debt
    records["st0_locked_cash"][r, col] = exec_state.locked_cash
    records["st0_free_cash"][r, col] = exec_state.free_cash
    records["st0_val_price"][r, col] = exec_state.val_price
    records["st0_value"][r, col] = exec_state.value
    records["req_size"][r, col] = order.size
    records["req_price"][r, col] = order.price
    records["req_size_type"][r, col] = order.size_type
    records["req_direction"][r, col] = order.direction
    records["req_fees"][r, col] = order.fees
    records["req_fixed_fees"][r, col] = order.fixed_fees
    records["req_slippage"][r, col] = order.slippage
    records["req_min_size"][r, col] = order.min_size
    records["req_max_size"][r, col] = order.max_size
    records["req_size_granularity"][r, col] = order.size_granularity
    records["req_leverage"][r, col] = order.leverage
    records["req_leverage_mode"][r, col] = order.leverage_mode
    records["req_reject_prob"][r, col] = order.reject_prob
    records["req_price_area_vio_mode"][r, col] = order.price_area_vio_mode
    records["req_allow_partial"][r, col] = order.allow_partial
    records["req_raise_reject"][r, col] = order.raise_reject
    records["req_log"][r, col] = order.log
    records["res_size"][r, col] = order_result.size
    records["res_price"][r, col] = order_result.price
    records["res_fees"][r, col] = order_result.fees
    records["res_side"][r, col] = order_result.side
    records["res_status"][r, col] = order_result.status
    records["res_status_info"][r, col] = order_result.status_info
    records["st1_cash"][r, col] = new_exec_state.cash
    records["st1_position"][r, col] = new_exec_state.position
    records["st1_debt"][r, col] = new_exec_state.debt
    records["st1_locked_cash"][r, col] = new_exec_state.locked_cash
    records["st1_free_cash"][r, col] = new_exec_state.free_cash
    records["st1_val_price"][r, col] = new_exec_state.val_price
    records["st1_value"][r, col] = new_exec_state.value
    records["order_id"][r, col] = order_id


@register_jitted(cache=True)
def fill_order_record_nb(records: tp.RecordArray2d, r: int, col: int, i: int, order_result: OrderResult) -> None:
    """Fill an order record."""

    records["id"][r, col] = r
    records["col"][r, col] = col
    records["idx"][r, col] = i
    records["size"][r, col] = order_result.size
    records["price"][r, col] = order_result.price
    records["fees"][r, col] = order_result.fees
    records["side"][r, col] = order_result.side


@register_jitted(cache=True)
def raise_rejected_order_nb(order_result: OrderResult) -> None:
    """Raise an `vectorbtpro.portfolio.enums.RejectedOrderError`."""

    if order_result.status_info == OrderStatusInfo.SizeNaN:
        raise RejectedOrderError("Size is NaN")
    if order_result.status_info == OrderStatusInfo.PriceNaN:
        raise RejectedOrderError("Price is NaN")
    if order_result.status_info == OrderStatusInfo.ValPriceNaN:
        raise RejectedOrderError("Asset valuation price is NaN")
    if order_result.status_info == OrderStatusInfo.ValueNaN:
        raise RejectedOrderError("Asset/group value is NaN")
    if order_result.status_info == OrderStatusInfo.ValueZeroNeg:
        raise RejectedOrderError("Asset/group value is zero or negative")
    if order_result.status_info == OrderStatusInfo.SizeZero:
        raise RejectedOrderError("Size is zero")
    if order_result.status_info == OrderStatusInfo.NoCash:
        raise RejectedOrderError("Not enough cash")
    if order_result.status_info == OrderStatusInfo.NoOpenPosition:
        raise RejectedOrderError("No open position to reduce/close")
    if order_result.status_info == OrderStatusInfo.MaxSizeExceeded:
        raise RejectedOrderError("Size is greater than maximum allowed")
    if order_result.status_info == OrderStatusInfo.RandomEvent:
        raise RejectedOrderError("Random event happened")
    if order_result.status_info == OrderStatusInfo.CantCoverFees:
        raise RejectedOrderError("Not enough cash to cover fees")
    if order_result.status_info == OrderStatusInfo.MinSizeNotReached:
        raise RejectedOrderError("Final size is less than minimum allowed")
    if order_result.status_info == OrderStatusInfo.PartialFill:
        raise RejectedOrderError("Final size is less than requested")
    raise RejectedOrderError


@register_jitted(cache=True)
def process_order_nb(
    group: int,
    col: int,
    i: int,
    exec_state: ExecState,
    order: Order,
    price_area: PriceArea = NoPriceArea,
    update_value: bool = False,
    order_records: tp.Optional[tp.RecordArray2d] = None,
    order_counts: tp.Optional[tp.Array1d] = None,
    log_records: tp.Optional[tp.RecordArray2d] = None,
    log_counts: tp.Optional[tp.Array1d] = None,
) -> tp.Tuple[OrderResult, ExecState]:
    """Process an order by executing it, saving relevant information to the logs, and returning a new state."""
    # Execute the order
    order_result, new_exec_state = execute_order_nb(
        exec_state=exec_state,
        order=order,
        price_area=price_area,
        update_value=update_value,
    )

    is_filled = order_result.status == OrderStatus.Filled
    if order_records is not None and order_counts is not None:
        if is_filled and order_records.shape[0] > 0:
            # Fill order record
            if order_counts[col] >= order_records.shape[0]:
                raise IndexError("order_records index out of range. Set a higher max_order_records.")
            fill_order_record_nb(order_records, order_counts[col], col, i, order_result)
            order_counts[col] += 1

    if log_records is not None and log_counts is not None:
        if order.log and log_records.shape[0] > 0:
            # Fill log record
            if log_counts[col] >= log_records.shape[0]:
                raise IndexError("log_records index out of range. Set a higher max_log_records.")
            fill_log_record_nb(
                log_records,
                log_counts[col],
                group,
                col,
                i,
                price_area,
                exec_state,
                order,
                order_result,
                new_exec_state,
                order_counts[col] - 1 if order_counts is not None and is_filled else -1,
            )
            log_counts[col] += 1

    return order_result, new_exec_state


@register_jitted(cache=True)
def order_nb(
    size: float = np.inf,
    price: float = np.inf,
    size_type: int = SizeType.Amount,
    direction: int = Direction.Both,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Create an order.

    See `vectorbtpro.portfolio.enums.Order` for details on arguments."""

    return Order(
        size=float(size),
        price=float(price),
        size_type=int(size_type),
        direction=int(direction),
        fees=float(fees),
        fixed_fees=float(fixed_fees),
        slippage=float(slippage),
        min_size=float(min_size),
        max_size=float(max_size),
        size_granularity=float(size_granularity),
        leverage=float(leverage),
        leverage_mode=int(leverage_mode),
        reject_prob=float(reject_prob),
        price_area_vio_mode=int(price_area_vio_mode),
        allow_partial=bool(allow_partial),
        raise_reject=bool(raise_reject),
        log=bool(log),
    )


@register_jitted(cache=True)
def close_position_nb(
    price: float = np.inf,
    fees: float = 0.0,
    fixed_fees: float = 0.0,
    slippage: float = 0.0,
    min_size: float = np.nan,
    max_size: float = np.nan,
    size_granularity: float = np.nan,
    leverage: float = 1.0,
    leverage_mode: int = LeverageMode.Lazy,
    reject_prob: float = 0.0,
    price_area_vio_mode: int = PriceAreaVioMode.Ignore,
    allow_partial: bool = True,
    raise_reject: bool = False,
    log: bool = False,
) -> Order:
    """Close the current position."""

    return order_nb(
        size=0.0,
        price=price,
        size_type=SizeType.TargetAmount,
        direction=Direction.Both,
        fees=fees,
        fixed_fees=fixed_fees,
        slippage=slippage,
        min_size=min_size,
        max_size=max_size,
        size_granularity=size_granularity,
        leverage=leverage,
        leverage_mode=leverage_mode,
        reject_prob=reject_prob,
        price_area_vio_mode=price_area_vio_mode,
        allow_partial=allow_partial,
        raise_reject=raise_reject,
        log=log,
    )


@register_jitted(cache=True)
def order_nothing_nb() -> Order:
    """Convenience function to order nothing."""
    return NoOrder


@register_jitted(cache=True)
def check_group_lens_nb(group_lens: tp.GroupLens, n_cols: int) -> None:
    """Check `group_lens`."""
    if np.sum(group_lens) != n_cols:
        raise ValueError("group_lens has incorrect total number of columns")


@register_jitted(cache=True)
def is_grouped_nb(group_lens: tp.GroupLens) -> bool:
    """Check if columm,ns are grouped, that is, more than one column per group."""
    return np.any(group_lens > 1)


@register_jitted(cache=True)
def prepare_records_nb(
    target_shape: tp.Shape,
    max_order_records: tp.Optional[int] = None,
    max_log_records: tp.Optional[int] = 0,
) -> tp.Tuple[tp.RecordArray2d, tp.RecordArray2d]:
    """Prepare records."""
    if max_order_records is None:
        order_records = np.empty((target_shape[0], target_shape[1]), dtype=order_dt)
    else:
        order_records = np.empty((max_order_records, target_shape[1]), dtype=order_dt)
    if max_log_records is None:
        log_records = np.empty((target_shape[0], target_shape[1]), dtype=log_dt)
    else:
        log_records = np.empty((max_log_records, target_shape[1]), dtype=log_dt)
    return order_records, log_records


@register_jitted(cache=True)
def prepare_last_cash_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare `last_cash`."""
    if cash_sharing:
        last_cash = np.empty(len(group_lens), dtype=np.float_)
        for group in range(len(group_lens)):
            last_cash[group] = float(flex_select_1d_pc_nb(init_cash, group))
    else:
        last_cash = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            last_cash[col] = float(flex_select_1d_pc_nb(init_cash, col))
    return last_cash


@register_jitted(cache=True)
def prepare_last_position_nb(target_shape: tp.Shape, init_position: tp.FlexArray1d) -> tp.Array1d:
    """Prepare `last_position`."""
    last_position = np.empty(target_shape[1], dtype=np.float_)
    for col in range(target_shape[1]):
        last_position[col] = float(flex_select_1d_pc_nb(init_position, col))
    return last_position


@register_jitted(cache=True)
def prepare_last_value_nb(
    target_shape: tp.Shape,
    group_lens: tp.GroupLens,
    cash_sharing: bool,
    init_cash: tp.FlexArray1d,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
) -> tp.Array1d:
    """Prepare `last_value`."""
    if cash_sharing:
        last_value = np.empty(len(group_lens), dtype=np.float_)
        from_col = 0
        for group in range(len(group_lens)):
            to_col = from_col + group_lens[group]
            _init_cash = float(flex_select_1d_pc_nb(init_cash, group))
            last_value[group] = _init_cash
            for col in range(from_col, to_col):
                _init_position = float(flex_select_1d_pc_nb(init_position, col))
                _init_price = float(flex_select_1d_pc_nb(init_price, col))
                if _init_position != 0:
                    last_value[group] += _init_position * _init_price
            from_col = to_col
    else:
        last_value = np.empty(target_shape[1], dtype=np.float_)
        for col in range(target_shape[1]):
            _init_cash = float(flex_select_1d_pc_nb(init_cash, col))
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position == 0:
                last_value[col] = _init_cash
            else:
                last_value[col] = _init_cash + _init_position * _init_price
    return last_value


@register_jitted(cache=True)
def prepare_last_pos_info_nb(
    target_shape: tp.Shape,
    init_position: tp.FlexArray1d,
    init_price: tp.FlexArray1d,
    fill_pos_info: bool = True,
) -> tp.RecordArray:
    """Prepare `last_pos_info`."""
    if fill_pos_info:
        last_pos_info = np.empty(target_shape[1], dtype=trade_dt)
        last_pos_info["id"][:] = -1
        last_pos_info["col"][:] = -1
        last_pos_info["size"][:] = np.nan
        last_pos_info["entry_order_id"][:] = -1
        last_pos_info["entry_idx"][:] = -1
        last_pos_info["entry_price"][:] = np.nan
        last_pos_info["entry_fees"][:] = np.nan
        last_pos_info["exit_order_id"][:] = -1
        last_pos_info["exit_idx"][:] = -1
        last_pos_info["exit_price"][:] = np.nan
        last_pos_info["exit_fees"][:] = np.nan
        last_pos_info["pnl"][:] = np.nan
        last_pos_info["return"][:] = np.nan
        last_pos_info["direction"][:] = -1
        last_pos_info["status"][:] = -1
        last_pos_info["parent_id"][:] = -1

        for col in range(target_shape[1]):
            _init_position = float(flex_select_1d_pc_nb(init_position, col))
            _init_price = float(flex_select_1d_pc_nb(init_price, col))
            if _init_position != 0:
                fill_init_pos_info_nb(last_pos_info[col], col, _init_position, _init_price)
    else:
        last_pos_info = np.empty(0, dtype=trade_dt)
    return last_pos_info


@register_jitted
def prepare_sim_out_nb(
    order_records: tp.RecordArray2d,
    order_counts: tp.Array1d,
    log_records: tp.RecordArray2d,
    log_counts: tp.Array1d,
    cash_deposits: tp.Array2d,
    cash_earnings: tp.Array2d,
    call_seq: tp.Optional[tp.Array2d] = None,
    in_outputs: tp.Optional[tp.NamedTuple] = None,
    sim_start: tp.Optional[tp.Array1d] = None,
    sim_end: tp.Optional[tp.Array1d] = None,
) -> SimulationOutput:
    """Prepare simulation output."""
    order_records_flat = generic_nb.repartition_nb(order_records, order_counts)
    log_records_flat = generic_nb.repartition_nb(log_records, log_counts)
    return SimulationOutput(
        order_records=order_records_flat,
        log_records=log_records_flat,
        cash_deposits=cash_deposits,
        cash_earnings=cash_earnings,
        call_seq=call_seq,
        in_outputs=in_outputs,
        sim_start=sim_start,
        sim_end=sim_end,
    )


@register_jitted(cache=True)
def get_trade_stats_nb(
    size: float,
    entry_price: float,
    entry_fees: float,
    exit_price: float,
    exit_fees: float,
    direction: int,
) -> tp.Tuple[float, float]:
    """Get trade statistics."""
    entry_val = size * entry_price
    exit_val = size * exit_price
    val_diff = add_nb(exit_val, -entry_val)
    if val_diff != 0 and direction == TradeDirection.Short:
        val_diff *= -1
    pnl = val_diff - entry_fees - exit_fees
    if is_close_nb(entry_val, 0):
        ret = np.nan
    else:
        ret = pnl / entry_val
    return pnl, ret


@register_jitted(cache=True)
def update_open_pos_info_stats_nb(record: tp.Record, position_now: float, price: float) -> None:
    """Update statistics of an open position record using custom price."""
    if record["id"] >= 0 and record["status"] == TradeStatus.Open:
        if np.isnan(record["exit_price"]):
            exit_price = price
        else:
            exit_size_sum = record["size"] - abs(position_now)
            exit_gross_sum = exit_size_sum * record["exit_price"]
            exit_gross_sum += abs(position_now) * price
            exit_price = exit_gross_sum / record["size"]
        pnl, ret = get_trade_stats_nb(
            record["size"],
            record["entry_price"],
            record["entry_fees"],
            exit_price,
            record["exit_fees"],
            record["direction"],
        )
        record["pnl"] = pnl
        record["return"] = ret


@register_jitted(cache=True)
def fill_init_pos_info_nb(record: tp.Record, col: int, position_now: float, price: float) -> None:
    """Fill position record for an initial position."""
    record["id"] = 0
    record["col"] = col
    record["size"] = abs(position_now)
    record["entry_order_id"] = -1
    record["entry_idx"] = -1
    record["entry_price"] = price
    record["entry_fees"] = 0.0
    record["exit_order_id"] = -1
    record["exit_idx"] = -1
    record["exit_price"] = np.nan
    record["exit_fees"] = 0.0
    if position_now >= 0:
        record["direction"] = TradeDirection.Long
    else:
        record["direction"] = TradeDirection.Short
    record["status"] = TradeStatus.Open
    record["parent_id"] = record["id"]

    # Update open position stats
    update_open_pos_info_stats_nb(record, position_now, np.nan)


@register_jitted(cache=True)
def update_pos_info_nb(
    record: tp.Record,
    i: int,
    col: int,
    position_before: float,
    position_now: float,
    order_result: OrderResult,
    order_id: int,
) -> None:
    """Update position record after filling an order."""
    if order_result.status == OrderStatus.Filled:
        if position_before == 0 and position_now != 0:
            # New position opened
            record["id"] += 1
            record["col"] = col
            record["size"] = order_result.size
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            record["entry_fees"] = order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        elif position_before != 0 and position_now == 0:
            # Position closed
            record["exit_order_id"] = order_id
            record["exit_idx"] = i
            if np.isnan(record["exit_price"]):
                exit_price = order_result.price
            else:
                exit_size_sum = record["size"] - abs(position_before)
                exit_gross_sum = exit_size_sum * record["exit_price"]
                exit_gross_sum += abs(position_before) * order_result.price
                exit_price = exit_gross_sum / record["size"]
            record["exit_price"] = exit_price
            record["exit_fees"] += order_result.fees
            pnl, ret = get_trade_stats_nb(
                record["size"],
                record["entry_price"],
                record["entry_fees"],
                record["exit_price"],
                record["exit_fees"],
                record["direction"],
            )
            record["pnl"] = pnl
            record["return"] = ret
            record["status"] = TradeStatus.Closed
        elif np.sign(position_before) != np.sign(position_now):
            # Position reversed
            record["id"] += 1
            record["size"] = abs(position_now)
            record["entry_order_id"] = order_id
            record["entry_idx"] = i
            record["entry_price"] = order_result.price
            new_pos_fraction = abs(position_now) / abs(position_now - position_before)
            record["entry_fees"] = new_pos_fraction * order_result.fees
            record["exit_order_id"] = -1
            record["exit_idx"] = -1
            record["exit_price"] = np.nan
            record["exit_fees"] = 0.0
            if order_result.side == OrderSide.Buy:
                record["direction"] = TradeDirection.Long
            else:
                record["direction"] = TradeDirection.Short
            record["status"] = TradeStatus.Open
            record["parent_id"] = record["id"]
        else:
            # Position changed
            if abs(position_before) <= abs(position_now):
                # Position increased
                entry_gross_sum = record["size"] * record["entry_price"]
                entry_gross_sum += order_result.size * order_result.price
                entry_price = entry_gross_sum / (record["size"] + order_result.size)
                record["entry_price"] = entry_price
                record["entry_fees"] += order_result.fees
                record["size"] += order_result.size
            else:
                # Position decreased
                record["exit_order_id"] = order_id
                if np.isnan(record["exit_price"]):
                    exit_price = order_result.price
                else:
                    exit_size_sum = record["size"] - abs(position_before)
                    exit_gross_sum = exit_size_sum * record["exit_price"]
                    exit_gross_sum += order_result.size * order_result.price
                    exit_price = exit_gross_sum / (exit_size_sum + order_result.size)
                record["exit_price"] = exit_price
                record["exit_fees"] += order_result.fees

        # Update open position stats
        update_open_pos_info_stats_nb(record, position_now, order_result.price)


@register_jitted(cache=True)
def resolve_hl_nb(open, high, low, close):
    """Resolve the current high and low."""
    if np.isnan(high):
        if np.isnan(open):
            high = close
        elif np.isnan(close):
            high = open
        else:
            high = max(open, close)
    if np.isnan(low):
        if np.isnan(open):
            low = close
        elif np.isnan(close):
            low = open
        else:
            low = min(open, close)
    return high, low


@register_jitted(cache=True)
def check_price_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_price: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Check whether a target price was hit.

    If `hard_price` is False, and `can_use_ohlc` and `check_open` are True and the target price
    is hit by open, returns open. Otherwise, returns the actual target price.

    Returns the stop price, whether it was hit by open, and whether it was hit during this bar."""
    high, low = resolve_hl_nb(
        open=open,
        high=high,
        low=low,
        close=close,
    )
    if hit_below:
        if can_use_ohlc and check_open and is_close_or_less_nb(open, price):
            if hard_price:
                return price, True, True
            return open, True, True
        if is_close_or_less_nb(close, price) or (can_use_ohlc and is_close_or_less_nb(low, price)):
            return price, False, True
        return price, False, False
    if can_use_ohlc and check_open and is_close_or_greater_nb(open, price):
        if hard_price:
            return price, True, True
        return open, True, True
    if is_close_or_greater_nb(close, price) or (can_use_ohlc and is_close_or_greater_nb(high, price)):
        return price, False, True
    return price, False, False


@register_jitted(cache=True)
def resolve_stop_exit_price_nb(
    stop_price: float,
    close: float,
    stop_exit_price: float,
) -> float:
    """Resolve the exit price of a stop order."""
    if stop_exit_price == StopExitPrice.Stop or stop_exit_price == StopExitPrice.HardStop:
        return float(stop_price)
    elif stop_exit_price == StopExitPrice.Close:
        return float(close)
    elif stop_exit_price < 0:
        raise ValueError("Invalid StopExitPrice option")
    return float(stop_exit_price)


@register_jitted(cache=True)
def is_limit_active_nb(init_idx: int, init_price: float) -> bool:
    """Check whether a limit order is active."""
    return init_idx != -1 and not np.isnan(init_price)


@register_jitted(cache=True)
def is_stop_active_nb(init_idx: int, stop: float) -> bool:
    """Check whether a stop order is active."""
    return init_idx != -1 and not np.isnan(stop)


@register_jitted(cache=True)
def is_time_stop_active_nb(init_idx: int, stop: int) -> bool:
    """Check whether a time stop order is active."""
    return init_idx != -1 and stop != -1


@register_jitted(cache=True)
def should_update_stop_nb(new_stop: float, upon_stop_update: int) -> bool:
    """Whether to update stop."""
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if not np.isnan(new_stop) or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def should_update_time_stop_nb(new_stop: int, upon_stop_update: int) -> bool:
    """Whether to update time stop."""
    if upon_stop_update == StopUpdateMode.Keep:
        return False
    if upon_stop_update == StopUpdateMode.Override or upon_stop_update == StopUpdateMode.OverrideNaN:
        if new_stop != -1 or upon_stop_update == StopUpdateMode.OverrideNaN:
            return True
        return False
    raise ValueError("Invalid StopUpdateMode option")


@register_jitted(cache=True)
def check_limit_expired_nb(
    creation_idx: int,
    i: int,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether limit is expired by comparing the current index with the creation index.

    Returns whether the limit expires already on open, and whether the limit expires during this bar."""
    if tif == -1 and expiry == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if creation_idx + tif <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < creation_idx + tif < i + 1:
                is_expired = True
        if expiry != -1:
            if expiry <= i:
                is_expired_on_open = True
                is_expired = True
            elif i < expiry < i + 1:
                is_expired = True
        return is_expired_on_open, is_expired
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_expired_on_open = False
        is_expired = False
        if tif != -1:
            if index[creation_idx] + tif <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < index[creation_idx] + tif < index[i] + freq:
                is_expired = True
        if expiry != -1:
            if expiry <= index[i]:
                is_expired_on_open = True
                is_expired = True
            elif index[i] < expiry < index[i] + freq:
                is_expired = True
        return is_expired_on_open, is_expired
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def resolve_limit_price_nb(
    init_price: float,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the limit price."""
    if delta_format == DeltaFormat.Percent100:
        limit_delta /= 100
        delta_format = DeltaFormat.Percent
    if not np.isnan(limit_delta):
        if hit_below:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta > 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = init_price - limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = init_price * (1 - limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
        else:
            if np.isinf(limit_delta) and delta_format != DeltaFormat.Target:
                if limit_delta < 0:
                    limit_price = -np.inf
                else:
                    limit_price = np.inf
            else:
                if delta_format == DeltaFormat.Absolute:
                    limit_price = init_price + limit_delta
                elif delta_format == DeltaFormat.Percent:
                    limit_price = init_price * (1 + limit_delta)
                elif delta_format == DeltaFormat.Target:
                    limit_price = limit_delta
                else:
                    raise ValueError("Invalid DeltaFormat option")
    else:
        limit_price = init_price
    return limit_price


@register_jitted(cache=True)
def check_limit_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    price: float,
    size: float,
    direction: int = Direction.Both,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    limit_reverse: bool = False,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_limit: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the limit price using `resolve_limit_price_nb` and check whether it was hit.

    Returns the limit price, whether it was hit before open, and whether it was hit during this bar.

    If `can_use_ohlc` and `check_open` is True and the stop is hit before open, returns open."""
    if size == 0:
        raise ValueError("Limit order size cannot be zero")
    _size = get_diraware_size_nb(size, direction)
    hit_below = (_size > 0 and not limit_reverse) or (_size < 0 and limit_reverse)
    limit_price = resolve_limit_price_nb(
        init_price=price,
        limit_delta=limit_delta,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    hit_on_open = False

    if can_use_ohlc:
        high, low = resolve_hl_nb(
            open=open,
            high=high,
            low=low,
            close=close,
        )
        if hit_below:
            if check_open and is_close_or_less_nb(open, limit_price):
                hit_on_open = True
                hit = True
                if not hard_limit:
                    limit_price = open
            else:
                hit = is_close_or_less_nb(low, limit_price)
                if hit and np.isinf(limit_price):
                    limit_price = low
        else:
            if check_open and is_close_or_greater_nb(open, limit_price):
                hit_on_open = True
                hit = True
                if not hard_limit:
                    limit_price = open
            else:
                hit = is_close_or_greater_nb(high, limit_price)
                if hit and np.isinf(limit_price):
                    limit_price = high
    else:
        if hit_below:
            hit = is_close_or_less_nb(close, limit_price)
        else:
            hit = is_close_or_greater_nb(close, limit_price)
        if hit and np.isinf(limit_price):
            limit_price = close
    return limit_price, hit_on_open, hit


@register_jitted(cache=True)
def resolve_limit_order_price_nb(
    limit_price: float,
    close: float,
    limit_order_price: float,
) -> float:
    """Resolve the limit order price of a limit order."""
    if limit_order_price == LimitOrderPrice.Limit or limit_order_price == LimitOrderPrice.HardLimit:
        return float(limit_price)
    elif limit_order_price == LimitOrderPrice.Close:
        return float(close)
    elif limit_order_price < 0:
        raise ValueError("Invalid LimitOrderPrice option")
    return float(limit_order_price)


@register_jitted(cache=True)
def resolve_stop_price_nb(
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Resolve the stop price."""
    if delta_format == DeltaFormat.Percent100:
        stop /= 100
        delta_format = DeltaFormat.Percent
    if hit_below:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price - abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 - abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    else:
        if delta_format == DeltaFormat.Absolute:
            stop_price = init_price + abs(stop)
        elif delta_format == DeltaFormat.Percent:
            stop_price = init_price * (1 + abs(stop))
        elif delta_format == DeltaFormat.Target:
            stop_price = stop
        else:
            raise ValueError("Invalid DeltaFormat option")
    return stop_price


@register_jitted(cache=True)
def check_stop_hit_nb(
    open: float,
    high: float,
    low: float,
    close: float,
    is_position_long: bool,
    init_price: float,
    stop: float,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
    can_use_ohlc: bool = True,
    check_open: bool = True,
    hard_stop: bool = False,
) -> tp.Tuple[float, bool, bool]:
    """Resolve the stop price using `resolve_stop_price_nb` and check whether it was hit.

    See `check_price_hit_nb`."""
    hit_below = (is_position_long and hit_below) or (not is_position_long and not hit_below)
    stop_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    return check_price_hit_nb(
        open=open,
        high=high,
        low=low,
        close=close,
        price=stop_price,
        hit_below=hit_below,
        can_use_ohlc=can_use_ohlc,
        check_open=check_open,
        hard_price=hard_stop,
    )


@register_jitted(cache=True)
def check_td_stop_hit_nb(
    init_idx: int,
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether TD stop was hit by comparing the current index with the initial index.

    Returns whether the stop was hit already on open, and whether the stop was hit during this bar."""
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if init_idx + stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < init_idx + stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if index[init_idx] + stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < index[init_idx] + stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_dt_stop_hit_nb(
    i: int,
    stop: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
    freq: tp.Optional[int] = None,
) -> tp.Tuple[bool, bool]:
    """Check whether DT stop was hit by comparing the current index with the initial index.

    Returns whether the stop was hit already on open, and whether the stop was hit during this bar."""
    if stop == -1:
        return False, False
    if time_delta_format == TimeDeltaFormat.Rows:
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= i:
                is_hit_on_open = True
                is_hit = True
            elif i < stop < i + 1:
                is_hit = True
        return is_hit_on_open, is_hit
    elif time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        if freq is None:
            raise ValueError("Must provide frequency for TimeDeltaFormat.Index")
        is_hit_on_open = False
        is_hit = False
        if stop != -1:
            if stop <= index[i]:
                is_hit_on_open = True
                is_hit = True
            elif index[i] < stop < index[i] + freq:
                is_hit = True
        return is_hit_on_open, is_hit
    else:
        raise ValueError("Invalid TimeDeltaFormat option")


@register_jitted(cache=True)
def check_tsl_th_hit_nb(
    is_position_long: bool,
    init_price: float,
    peak_price: float,
    threshold: float,
    delta_format: int = DeltaFormat.Percent,
) -> bool:
    """Resolve the TSL threshold price using `resolve_stop_price_nb` and check whether it was hit."""
    hit_below = not is_position_long
    tsl_th_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=threshold,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if hit_below:
        return is_close_or_less_nb(peak_price, tsl_th_price)
    return is_close_or_greater_nb(peak_price, tsl_th_price)


@register_jitted(cache=True)
def resolve_dyn_limit_price_nb(val_price: float, price: float, limit_price: float) -> float:
    """Resolve price dynamically.

    Uses the valuation price as the left bound and order price as the right bound."""
    if np.isinf(limit_price):
        if limit_price < 0:
            return float(val_price)
        return float(price)
    return float(limit_price)


@register_jitted(cache=True)
def resolve_dyn_stop_entry_price_nb(val_price: float, price: float, stop_entry_price: float) -> float:
    """Resolve stop entry price dynamically.

    Uses the valuation/open price as the left bound and order price as the right bound."""
    if np.isinf(stop_entry_price):
        if stop_entry_price < 0:
            return float(val_price)
        return float(price)
    if stop_entry_price < 0:
        if stop_entry_price == StopEntryPrice.ValPrice:
            return float(val_price)
        if stop_entry_price == StopEntryPrice.Price:
            return float(price)
        raise ValueError("Only valuation and order price are supported when setting stop entry price dynamically")
    return float(stop_entry_price)


@register_jitted(cache=True)
def get_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_price: float,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    delta_format: int = DeltaFormat.Percent,
    hit_below: bool = True,
) -> float:
    """Get the exit size corresponding to the current step in the ladder."""
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    stop = flex_select_nb(stop_, step, col)
    if np.isnan(stop):
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if not np.isnan(flex_select_nb(stop_, i, col)):
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    hit_below = (init_position >= 0 and hit_below) or (init_position < 0 and not hit_below)
    price = resolve_stop_price_nb(
        init_price=init_price,
        stop=stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    last_stop = flex_select_nb(stop_, last_step, col)
    last_price = resolve_stop_price_nb(
        init_price=init_price,
        stop=last_stop,
        delta_format=delta_format,
        hit_below=hit_below,
    )
    if step == 0:
        prev_price = init_price
    else:
        prev_stop = flex_select_nb(stop_, step - 1, col)
        prev_price = resolve_stop_price_nb(
            init_price=init_price,
            stop=prev_stop,
            delta_format=delta_format,
            hit_below=hit_below,
        )
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (price - prev_price) / (last_price - init_price)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (price - prev_price) / (last_price - prev_price)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_jitted(cache=True)
def get_time_stop_ladder_exit_size_nb(
    stop_: tp.FlexArray2d,
    step: int,
    col: int,
    init_idx: int,
    init_position: float,
    position_now: float,
    ladder: int = StopLadderMode.Disabled,
    time_delta_format: int = TimeDeltaFormat.Index,
    index: tp.Optional[tp.Array1d] = None,
) -> float:
    """Get the exit size corresponding to the current step in the ladder."""
    if ladder == StopLadderMode.Disabled:
        raise ValueError("Stop ladder must be enabled to select exit size")
    if ladder == StopLadderMode.Dynamic:
        raise ValueError("Stop ladder must be static to select exit size")
    if init_idx == -1:
        raise ValueError("Initial index of the ladder must be known")
    if time_delta_format == TimeDeltaFormat.Index:
        if index is None:
            raise ValueError("Must provide index for TimeDeltaFormat.Index")
        init_idx = index[init_idx]
    idx = flex_select_nb(stop_, step, col)
    if idx == -1:
        return np.nan
    last_step = -1
    for i in range(step, stop_.shape[0]):
        if flex_select_nb(stop_, i, col) != -1:
            last_step = i
        else:
            break
    if last_step == -1:
        return np.nan
    if step == last_step:
        return abs(position_now)

    if ladder == StopLadderMode.Uniform:
        exit_fraction = 1 / (last_step + 1)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptUniform:
        exit_fraction = 1 / (last_step + 1 - step)
        return exit_fraction * abs(position_now)
    last_idx = flex_select_nb(stop_, last_step, col)
    if step == 0:
        prev_idx = init_idx
    else:
        prev_idx = flex_select_nb(stop_, step - 1, col)
    if ladder == StopLadderMode.Weighted:
        exit_fraction = (idx - prev_idx) / (last_idx - init_idx)
        return exit_fraction * abs(init_position)
    if ladder == StopLadderMode.AdaptWeighted:
        exit_fraction = (idx - prev_idx) / (last_idx - prev_idx)
        return exit_fraction * abs(position_now)
    raise ValueError("Invalid StopLadderMode option")


@register_jitted(cache=True)
def is_limit_info_active_nb(limit_info: tp.Record) -> bool:
    """Check whether information record for a limit order is active."""
    return is_limit_active_nb(limit_info["init_idx"], limit_info["init_price"])


@register_jitted(cache=True)
def is_stop_info_active_nb(stop_info: tp.Record) -> bool:
    """Check whether information record for a stop order is active."""
    return is_stop_active_nb(stop_info["init_idx"], stop_info["stop"])


@register_jitted(cache=True)
def is_time_stop_info_active_nb(time_stop_info: tp.Record) -> bool:
    """Check whether information record for a time stop order is active."""
    return is_time_stop_active_nb(time_stop_info["init_idx"], time_stop_info["stop"])


@register_jitted(cache=True)
def is_stop_info_ladder_active_nb(info: tp.Record) -> bool:
    """Check whether information record for a stop ladder is active."""
    return info["step"] != -1


@register_jitted(cache=True)
def set_limit_info_nb(
    limit_info: tp.Record,
    signal_idx: int,
    creation_idx: tp.Optional[int] = None,
    init_idx: tp.Optional[int] = None,
    init_price: float = -np.inf,
    init_size: float = np.inf,
    init_size_type: int = SizeType.Amount,
    init_direction: int = Direction.Both,
    init_stop_type: int = -1,
    delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    tif: int = -1,
    expiry: int = -1,
    time_delta_format: int = TimeDeltaFormat.Index,
    reverse: bool = False,
    order_price: int = LimitOrderPrice.Limit,
) -> None:
    """Set limit order information.

    See `vectorbtpro.portfolio.enums.limit_info_dt`."""
    limit_info["signal_idx"] = signal_idx
    limit_info["creation_idx"] = creation_idx if creation_idx is not None else signal_idx
    limit_info["init_idx"] = init_idx if init_idx is not None else signal_idx
    limit_info["init_price"] = init_price
    limit_info["init_size"] = init_size
    limit_info["init_size_type"] = init_size_type
    limit_info["init_direction"] = init_direction
    limit_info["init_stop_type"] = init_stop_type
    limit_info["delta"] = delta
    limit_info["delta_format"] = delta_format
    limit_info["tif"] = tif
    limit_info["expiry"] = expiry
    limit_info["time_delta_format"] = time_delta_format
    limit_info["reverse"] = reverse
    limit_info["order_price"] = order_price


@register_jitted(cache=True)
def clear_limit_info_nb(limit_info: tp.Record) -> None:
    """Clear limit order information."""
    limit_info["signal_idx"] = -1
    limit_info["creation_idx"] = -1
    limit_info["init_idx"] = -1
    limit_info["init_price"] = np.nan
    limit_info["init_size"] = np.nan
    limit_info["init_size_type"] = -1
    limit_info["init_direction"] = -1
    limit_info["init_stop_type"] = -1
    limit_info["delta"] = np.nan
    limit_info["delta_format"] = -1
    limit_info["tif"] = -1
    limit_info["expiry"] = -1
    limit_info["time_delta_format"] = -1
    limit_info["reverse"] = False
    limit_info["order_price"] = np.nan


@register_jitted(cache=True)
def set_sl_info_nb(
    sl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set SL order information.

    See `vectorbtpro.portfolio.enums.sl_info_dt`."""
    sl_info["init_idx"] = init_idx
    sl_info["init_price"] = init_price
    sl_info["init_position"] = init_position
    sl_info["stop"] = stop
    sl_info["exit_price"] = exit_price
    sl_info["exit_size"] = exit_size
    sl_info["exit_size_type"] = exit_size_type
    sl_info["exit_type"] = exit_type
    sl_info["order_type"] = order_type
    sl_info["limit_delta"] = limit_delta
    sl_info["delta_format"] = delta_format
    sl_info["ladder"] = ladder
    sl_info["step"] = step
    sl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_sl_info_nb(sl_info: tp.Record) -> None:
    """Clear SL order information."""
    sl_info["init_idx"] = -1
    sl_info["init_price"] = np.nan
    sl_info["init_position"] = np.nan
    sl_info["stop"] = np.nan
    sl_info["exit_price"] = -1
    sl_info["exit_size"] = np.nan
    sl_info["exit_size_type"] = -1
    sl_info["exit_type"] = -1
    sl_info["order_type"] = -1
    sl_info["limit_delta"] = np.nan
    sl_info["delta_format"] = -1
    sl_info["ladder"] = -1
    sl_info["step"] = -1
    sl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tsl_info_nb(
    tsl_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    peak_idx: tp.Optional[int] = None,
    peak_price: tp.Optional[float] = None,
    stop: float = np.nan,
    th: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TSL/TTP order information.

    See `vectorbtpro.portfolio.enums.tsl_info_dt`."""
    tsl_info["init_idx"] = init_idx
    tsl_info["init_price"] = init_price
    tsl_info["init_position"] = init_position
    tsl_info["peak_idx"] = peak_idx if peak_idx is not None else init_idx
    tsl_info["peak_price"] = peak_price if peak_price is not None else init_price
    tsl_info["stop"] = stop
    tsl_info["th"] = th
    tsl_info["exit_price"] = exit_price
    tsl_info["exit_size"] = exit_size
    tsl_info["exit_size_type"] = exit_size_type
    tsl_info["exit_type"] = exit_type
    tsl_info["order_type"] = order_type
    tsl_info["limit_delta"] = limit_delta
    tsl_info["delta_format"] = delta_format
    tsl_info["ladder"] = ladder
    tsl_info["step"] = step
    tsl_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tsl_info_nb(tsl_info: tp.Record) -> None:
    """Clear TSL/TTP order information."""
    tsl_info["init_idx"] = -1
    tsl_info["init_price"] = np.nan
    tsl_info["init_position"] = np.nan
    tsl_info["peak_idx"] = -1
    tsl_info["peak_price"] = np.nan
    tsl_info["stop"] = np.nan
    tsl_info["th"] = np.nan
    tsl_info["exit_price"] = -1
    tsl_info["exit_size"] = np.nan
    tsl_info["exit_size_type"] = -1
    tsl_info["exit_type"] = -1
    tsl_info["order_type"] = -1
    tsl_info["limit_delta"] = np.nan
    tsl_info["delta_format"] = -1
    tsl_info["ladder"] = -1
    tsl_info["step"] = -1
    tsl_info["step_idx"] = -1


@register_jitted(cache=True)
def set_tp_info_nb(
    tp_info: tp.Record,
    init_idx: int,
    init_price: float = -np.inf,
    init_position: float = np.nan,
    stop: float = np.nan,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set TP order information.

    See `vectorbtpro.portfolio.enums.tp_info_dt`."""
    tp_info["init_idx"] = init_idx
    tp_info["init_price"] = init_price
    tp_info["init_position"] = init_position
    tp_info["stop"] = stop
    tp_info["exit_price"] = exit_price
    tp_info["exit_size"] = exit_size
    tp_info["exit_size_type"] = exit_size_type
    tp_info["exit_type"] = exit_type
    tp_info["order_type"] = order_type
    tp_info["limit_delta"] = limit_delta
    tp_info["delta_format"] = delta_format
    tp_info["ladder"] = ladder
    tp_info["step"] = step
    tp_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_tp_info_nb(tp_info: tp.Record) -> None:
    """Clear TP order information."""
    tp_info["init_idx"] = -1
    tp_info["init_price"] = np.nan
    tp_info["init_position"] = np.nan
    tp_info["stop"] = np.nan
    tp_info["exit_price"] = -1
    tp_info["exit_size"] = np.nan
    tp_info["exit_size_type"] = -1
    tp_info["exit_type"] = -1
    tp_info["order_type"] = -1
    tp_info["limit_delta"] = np.nan
    tp_info["delta_format"] = -1
    tp_info["ladder"] = -1
    tp_info["step"] = -1
    tp_info["step_idx"] = -1


@register_jitted(cache=True)
def set_time_info_nb(
    time_info: tp.Record,
    init_idx: int,
    init_position: float = np.nan,
    stop: int = -1,
    exit_price: float = StopExitPrice.Stop,
    exit_size: float = np.nan,
    exit_size_type: int = -1,
    exit_type: int = StopExitType.Close,
    order_type: int = OrderType.Market,
    limit_delta: float = np.nan,
    delta_format: int = DeltaFormat.Percent,
    time_delta_format: int = TimeDeltaFormat.Index,
    ladder: int = StopLadderMode.Disabled,
    step: int = -1,
    step_idx: int = -1,
) -> None:
    """Set time order information.

    See `vectorbtpro.portfolio.enums.time_info_dt`."""
    time_info["init_idx"] = init_idx
    time_info["init_position"] = init_position
    time_info["stop"] = stop
    time_info["exit_price"] = exit_price
    time_info["exit_size"] = exit_size
    time_info["exit_size_type"] = exit_size_type
    time_info["exit_type"] = exit_type
    time_info["order_type"] = order_type
    time_info["limit_delta"] = limit_delta
    time_info["delta_format"] = delta_format
    time_info["time_delta_format"] = time_delta_format
    time_info["ladder"] = ladder
    time_info["step"] = step
    time_info["step_idx"] = step_idx


@register_jitted(cache=True)
def clear_time_info_nb(time_info: tp.Record) -> None:
    """Clear time order information."""
    time_info["init_idx"] = -1
    time_info["init_position"] = np.nan
    time_info["stop"] = -1
    time_info["exit_price"] = -1
    time_info["exit_size"] = np.nan
    time_info["exit_size_type"] = -1
    time_info["exit_type"] = -1
    time_info["order_type"] = -1
    time_info["limit_delta"] = np.nan
    time_info["delta_format"] = -1
    time_info["time_delta_format"] = -1
    time_info["ladder"] = -1
    time_info["step"] = -1
    time_info["step_idx"] = -1


@register_jitted(cache=True)
def get_limit_info_target_price_nb(limit_info: tp.Record) -> float:
    """Get target price from limit order information."""
    if not is_limit_info_active_nb(limit_info):
        return np.nan
    if limit_info["init_size"] == 0:
        raise ValueError("Limit order size cannot be zero")
    size = get_diraware_size_nb(limit_info["init_size"], limit_info["init_direction"])
    hit_below = (size > 0 and not limit_info["reverse"]) or (size < 0 and limit_info["reverse"])
    return resolve_limit_price_nb(
        init_price=limit_info["init_price"],
        limit_delta=limit_info["delta"],
        delta_format=limit_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_sl_info_target_price_nb(sl_info: tp.Record, position_now: float) -> float:
    """Get target price from SL order information."""
    if not is_stop_info_active_nb(sl_info):
        return np.nan
    hit_below = position_now > 0
    return resolve_stop_price_nb(
        init_price=sl_info["init_price"],
        stop=sl_info["stop"],
        delta_format=sl_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_tsl_info_target_price_nb(tsl_info: tp.Record, position_now: float) -> float:
    """Get target price from TSL/TTP order information."""
    if not is_stop_info_active_nb(tsl_info):
        return np.nan
    hit_below = position_now > 0
    return resolve_stop_price_nb(
        init_price=tsl_info["peak_price"],
        stop=tsl_info["stop"],
        delta_format=tsl_info["delta_format"],
        hit_below=hit_below,
    )


@register_jitted
def get_tp_info_target_price_nb(tp_info: tp.Record, position_now: float) -> float:
    """Get target price from TP order information."""
    if not is_stop_info_active_nb(tp_info):
        return np.nan
    hit_below = position_now < 0
    return resolve_stop_price_nb(
        init_price=tp_info["init_price"],
        stop=tp_info["stop"],
        delta_format=tp_info["delta_format"],
        hit_below=hit_below,
    )
