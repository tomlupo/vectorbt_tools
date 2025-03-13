# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Classes for preparing portfolio simulations."""

import warnings
from collections import namedtuple
from functools import cached_property as cachedproperty

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base import chunking as base_ch
from vectorbtpro.base.decorators import override_arg_config, attach_arg_properties
from vectorbtpro.base.preparing import BasePreparer
from vectorbtpro.base.reshaping import to_2d_array, broadcast_array_to, broadcast
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import OHLCDataMixin, Data
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.sim_range import SimRangeMixin
from vectorbtpro.portfolio import nb, enums
from vectorbtpro.portfolio.call_seq import require_call_seq, build_call_seq
from vectorbtpro.portfolio.orders import FSOrders
from vectorbtpro.registries.ch_registry import ch_reg
from vectorbtpro.registries.jit_registry import jit_reg, register_jitted
from vectorbtpro.signals import nb as signals_nb
from vectorbtpro.utils import checks, chunking as ch
from vectorbtpro.utils.config import Configured, merge_dicts, ReadonlyConfig
from vectorbtpro.utils.mapping import to_field_mapping
from vectorbtpro.utils.template import CustomTemplate, substitute_templates, RepFunc

__all__ = [
    "PFPrepResult",
    "BasePFPreparer",
    "FOPreparer",
    "FSPreparer",
    "FOFPreparer",
    "FDOFPreparer",
]

__pdoc__ = {}


@register_jitted(cache=True)
def valid_price_from_ago_1d_nb(price: tp.Array1d) -> tp.Array1d:
    """Parse from_ago from a valid price."""
    from_ago = np.empty(price.shape, dtype=np.int_)
    for i in range(price.shape[0] - 1, -1, -1):
        if i > 0 and not np.isnan(price[i]):
            for j in range(i - 1, -1, -1):
                if not np.isnan(price[j]):
                    break
            from_ago[i] = i - j
        else:
            from_ago[i] = 1
    return from_ago


PFPrepResultT = tp.TypeVar("PFPrepResultT", bound="PFPrepResult")


class PFPrepResult(Configured):
    """Result of preparation."""

    def __init__(
        self,
        target_func: tp.Optional[tp.Callable] = None,
        target_args: tp.Optional[tp.Kwargs] = None,
        pf_args: tp.Optional[tp.Kwargs] = None,
    ) -> None:
        Configured.__init__(self, target_func=target_func, target_args=target_args, pf_args=pf_args)

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        """Target function."""
        return self.config["target_func"]

    @cachedproperty
    def target_args(self) -> tp.Kwargs:
        """Target arguments."""
        return self.config["target_args"]

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        """Portfolio arguments."""
        return self.config["pf_args"]


base_arg_config = ReadonlyConfig(
    dict(
        data=dict(),
        open=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        high=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        low=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        bm_close=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        cash_earnings=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        init_cash=dict(map_enum_kwargs=dict(enum=enums.InitCashMode, look_for_type=str)),
        init_position=dict(),
        init_price=dict(),
        cash_deposits=dict(),
        group_by=dict(),
        cash_sharing=dict(),
        freq=dict(),
        sim_start=dict(),
        sim_end=dict(),
        call_seq=dict(map_enum_kwargs=dict(enum=enums.CallSeqType, look_for_type=str)),
        attach_call_seq=dict(),
        keep_inout_flex=dict(),
        in_outputs=dict(has_default=False),
    )
)
"""_"""

__pdoc__[
    "base_arg_config"
] = f"""Argument config for `BasePFPreparer`.

```python
{base_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(base_arg_config)
class BasePFPreparer(BasePreparer):
    """Base class for preparing portfolio simulations."""

    _settings_path: tp.SettingsPath = "portfolio"

    @classmethod
    def find_target_func(cls, target_func_name: str) -> tp.Callable:
        return getattr(nb, target_func_name)

    # ############# Ready arguments ############# #

    @cachedproperty
    def init_cash_mode(self) -> tp.Optional[int]:
        """Initial cash mode."""
        init_cash = self["init_cash"]
        if checks.is_int(init_cash) and init_cash in enums.InitCashMode:
            return init_cash
        return None

    @cachedproperty
    def group_by(self) -> tp.GroupByLike:
        """Argument `group_by`."""
        group_by = self["group_by"]
        if group_by is None and self.cash_sharing:
            return True
        return group_by

    @cachedproperty
    def auto_call_seq(self) -> bool:
        """Whether automatic call sequence is enabled."""
        call_seq = self["call_seq"]
        return checks.is_int(call_seq) and call_seq == enums.CallSeqType.Auto

    @classmethod
    def parse_data(
        cls,
        data: tp.Union[None, OHLCDataMixin, str, tp.ArrayLike],
        all_ohlc: bool = False,
    ) -> tp.Optional[OHLCDataMixin]:
        """Parse an instance with OHLC features."""
        if data is None:
            return None
        if isinstance(data, OHLCDataMixin):
            return data
        if isinstance(data, str):
            return Data.from_data_str(data)
        if isinstance(data, pd.DataFrame):
            ohlcv_acc = data.vbt.ohlcv
            if all_ohlc and ohlcv_acc.has_ohlc:
                return ohlcv_acc
            if not all_ohlc and ohlcv_acc.has_any_ohlc:
                return ohlcv_acc
        return None

    @cachedproperty
    def data(self) -> tp.Optional[OHLCDataMixin]:
        """Argument `data`."""
        return self.parse_data(self["data"])

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_open(self) -> tp.ArrayLike:
        """Argument `open` before broadcasting."""
        open = self["open"]
        if open is None:
            if self.data is not None:
                open = self.data.open
            if open is None:
                return np.nan
        return open

    @cachedproperty
    def _pre_high(self) -> tp.ArrayLike:
        """Argument `high` before broadcasting."""
        high = self["high"]
        if high is None:
            if self.data is not None:
                high = self.data.high
            if high is None:
                return np.nan
        return high

    @cachedproperty
    def _pre_low(self) -> tp.ArrayLike:
        """Argument `low` before broadcasting."""
        low = self["low"]
        if low is None:
            if self.data is not None:
                low = self.data.low
            if low is None:
                return np.nan
        return low

    @cachedproperty
    def _pre_close(self) -> tp.ArrayLike:
        """Argument `close` before broadcasting."""
        close = self["close"]
        if close is None:
            if self.data is not None:
                close = self.data.close
            if close is None:
                return np.nan
        return close

    @cachedproperty
    def _pre_bm_close(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `bm_close` before broadcasting."""
        bm_close = self["bm_close"]
        if bm_close is not None and not isinstance(bm_close, bool):
            return bm_close
        return np.nan

    @cachedproperty
    def _pre_init_cash(self) -> tp.ArrayLike:
        """Argument `init_cash` before broadcasting."""
        if self.init_cash_mode is not None:
            return np.inf
        return self["init_cash"]

    @cachedproperty
    def _pre_init_position(self) -> tp.ArrayLike:
        """Argument `init_position` before broadcasting."""
        return self["init_position"]

    @cachedproperty
    def _pre_init_price(self) -> tp.ArrayLike:
        """Argument `init_price` before broadcasting."""
        return self["init_price"]

    @cachedproperty
    def _pre_cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits` before broadcasting."""
        return self["cash_deposits"]

    @cachedproperty
    def _pre_freq(self) -> tp.Optional[tp.FrequencyLike]:
        """Argument `freq` before casting to nanosecond format."""
        freq = self["freq"]
        if freq is None and self.data is not None:
            return self.data.symbol_wrapper.freq
        return freq

    @cachedproperty
    def _pre_call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq` before broadcasting."""
        if self.auto_call_seq:
            return None
        return self["call_seq"]

    @cachedproperty
    def _pre_in_outputs(self) -> tp.Union[None, tp.NamedTuple, CustomTemplate]:
        """Argument `in_outputs` before broadcasting."""
        in_outputs = self["in_outputs"]
        if (
            in_outputs is not None
            and not isinstance(in_outputs, CustomTemplate)
            and not checks.is_namedtuple(in_outputs)
        ):
            in_outputs = to_field_mapping(in_outputs)
            in_outputs = namedtuple("InOutputs", in_outputs)(**in_outputs)
        return in_outputs

    # ############# After broadcasting ############# #

    @cachedproperty
    def cs_group_lens(self) -> tp.GroupLens:
        """Cash sharing aware group lengths."""
        cs_group_lens = self.wrapper.grouper.get_group_lens(group_by=None if self.cash_sharing else False)
        checks.assert_subdtype(cs_group_lens, np.integer, arg_name="cs_group_lens")
        return cs_group_lens

    @cachedproperty
    def group_lens(self) -> tp.GroupLens:
        """Group lengths."""
        return self.wrapper.grouper.get_group_lens(group_by=self.group_by)

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        """Simulation group lengths."""
        return self.group_lens

    def align_pc_arr(
        self,
        arr: tp.ArrayLike,
        group_lens: tp.Optional[tp.GroupLens] = None,
        check_dtype: tp.Optional[tp.DTypeLike] = None,
        cast_to_dtype: tp.Optional[tp.DTypeLike] = None,
        reduce_func: tp.Union[None, str, tp.Callable] = None,
        arg_name: tp.Optional[str] = None,
    ) -> tp.Array1d:
        """Align a per-column array."""
        arr = np.asarray(arr)
        if check_dtype is not None:
            checks.assert_subdtype(arr, check_dtype, arg_name=arg_name)
        if cast_to_dtype is not None:
            arr = np.require(arr, dtype=cast_to_dtype)
        if arr.size > 1 and group_lens is not None and reduce_func is not None:
            if len(self.group_lens) == len(arr) != len(group_lens) == len(self.wrapper.columns):
                new_arr = np.empty(len(self.wrapper.columns), dtype=np.int_)
                col_generator = self.wrapper.grouper.iter_group_idxs()
                for i, cols in enumerate(col_generator):
                    new_arr[cols] = arr[i]
                arr = new_arr
            if len(self.wrapper.columns) == len(arr) != len(group_lens):
                new_arr = np.empty(len(group_lens), dtype=np.int_)
                col_generator = self.wrapper.grouper.iter_group_lens(group_lens)
                for i, cols in enumerate(col_generator):
                    if isinstance(reduce_func, str):
                        new_arr[i] = getattr(arr[cols], reduce_func)()
                    else:
                        new_arr[i] = reduce_func(arr[cols])
                arr = new_arr
        if group_lens is not None:
            return broadcast_array_to(arr, len(group_lens))
        return broadcast_array_to(arr, len(self.wrapper.columns))

    @cachedproperty
    def init_cash(self) -> tp.Array1d:
        """Argument `init_cash`."""
        return self.align_pc_arr(
            self._pre_init_cash,
            group_lens=self.cs_group_lens,
            check_dtype=np.number,
            cast_to_dtype=np.float_,
            reduce_func="sum",
            arg_name="init_cash",
        )

    @cachedproperty
    def init_position(self) -> tp.Array1d:
        """Argument `init_position`."""
        init_position = self.align_pc_arr(
            self._pre_init_position,
            check_dtype=np.number,
            cast_to_dtype=np.float_,
            arg_name="init_position",
        )
        if (((init_position > 0) | (init_position < 0)) & np.isnan(self.init_price)).any():
            warnings.warn(f"Initial position has undefined price. Set init_price.", stacklevel=2)
        return init_position

    @cachedproperty
    def init_price(self) -> tp.Array1d:
        """Argument `init_price`."""
        return self.align_pc_arr(
            self._pre_init_price,
            check_dtype=np.number,
            cast_to_dtype=np.float_,
            arg_name="init_price",
        )

    @cachedproperty
    def cash_deposits(self) -> tp.ArrayLike:
        """Argument `cash_deposits`."""
        cash_deposits = self["cash_deposits"]
        checks.assert_subdtype(cash_deposits, np.number, arg_name="cash_deposits")
        return broadcast(
            cash_deposits,
            to_shape=(self.target_shape[0], len(self.cs_group_lens)),
            to_pd=False,
            keep_flex=self.keep_inout_flex,
            reindex_kwargs=dict(fill_value=0.0),
            require_kwargs=self.broadcast_kwargs.get("require_kwargs", {}),
        )

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        """Get automatic `sim_start`"""
        return None

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        """Get automatic `sim_end`"""
        return None

    @cachedproperty
    def sim_start(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `sim_start`."""
        sim_start = self["sim_start"]
        if sim_start is None:
            return None
        if isinstance(sim_start, str) and sim_start.lower() == "auto":
            sim_start = self.auto_sim_start
            if sim_start is None:
                return None
        sim_start_arr = np.asarray(sim_start)
        if np.issubdtype(sim_start_arr.dtype, np.integer):
            if sim_start_arr.ndim == 0:
                return sim_start
            new_sim_start = sim_start_arr
        else:
            if sim_start_arr.ndim == 0:
                return SimRangeMixin.resolve_sim_start_value(sim_start, wrapper=self.wrapper)
            new_sim_start = np.empty(len(sim_start), dtype=np.int_)
            for i in range(len(sim_start)):
                new_sim_start[i] = SimRangeMixin.resolve_sim_start_value(sim_start[i], wrapper=self.wrapper)
        return self.align_pc_arr(
            new_sim_start,
            group_lens=self.sim_group_lens,
            check_dtype=np.integer,
            cast_to_dtype=np.int_,
            reduce_func="min",
            arg_name="sim_start",
        )

    @cachedproperty
    def sim_end(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `sim_end`."""
        sim_end = self["sim_end"]
        if sim_end is None:
            return None
        if isinstance(sim_end, str) and sim_end.lower() == "auto":
            sim_end = self.auto_sim_end
            if sim_end is None:
                return None
        sim_end_arr = np.asarray(sim_end)
        if np.issubdtype(sim_end_arr.dtype, np.integer):
            if sim_end_arr.ndim == 0:
                return sim_end
            new_sim_end = sim_end_arr
        else:
            if sim_end_arr.ndim == 0:
                return SimRangeMixin.resolve_sim_end_value(sim_end, wrapper=self.wrapper)
            new_sim_end = np.empty(len(sim_end), dtype=np.int_)
            for i in range(len(sim_end)):
                new_sim_end[i] = SimRangeMixin.resolve_sim_end_value(sim_end[i], wrapper=self.wrapper)
        return self.align_pc_arr(
            new_sim_end,
            group_lens=self.sim_group_lens,
            check_dtype=np.integer,
            cast_to_dtype=np.int_,
            reduce_func="max",
            arg_name="sim_end",
        )

    @cachedproperty
    def call_seq(self) -> tp.Optional[tp.ArrayLike]:
        """Argument `call_seq`."""
        call_seq = self._pre_call_seq
        if call_seq is None and self.attach_call_seq:
            call_seq = enums.CallSeqType.Default
        if call_seq is not None:
            if checks.is_any_array(call_seq):
                call_seq = require_call_seq(broadcast(call_seq, to_shape=self.target_shape, to_pd=False))
            else:
                call_seq = build_call_seq(self.target_shape, self.group_lens, call_seq_type=call_seq)
        if call_seq is not None:
            checks.assert_subdtype(call_seq, np.integer, arg_name="call_seq")
        return call_seq

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                group_lens=self.group_lens,
                cs_group_lens=self.cs_group_lens,
                cash_sharing=self.cash_sharing,
                init_cash=self.init_cash,
                init_position=self.init_position,
                init_price=self.init_price,
                cash_deposits=self.cash_deposits,
                sim_start=self.sim_start,
                sim_end=self.sim_end,
                call_seq=self.call_seq,
                auto_call_seq=self.auto_call_seq,
                attach_call_seq=self.attach_call_seq,
                in_outputs=self._pre_in_outputs,
            ),
            BasePreparer.template_context.func(self),
        )

    @cachedproperty
    def in_outputs(self) -> tp.Optional[tp.NamedTuple]:
        """Argument `in_outputs`."""
        return substitute_templates(self._pre_in_outputs, self.template_context, eval_id="in_outputs")

    # ############# Result ############# #

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        """Arguments to be passed to the portfolio."""
        kwargs = dict()
        for k, v in self.config.items():
            if k not in self.arg_config and k != "arg_config":
                kwargs[k] = v
        return dict(
            wrapper=self.wrapper,
            open=self.open if self._pre_open is not np.nan else None,
            high=self.high if self._pre_high is not np.nan else None,
            low=self.low if self._pre_low is not np.nan else None,
            close=self.close,
            cash_sharing=self.cash_sharing,
            init_cash=self.init_cash if self.init_cash_mode is None else self.init_cash_mode,
            init_position=self.init_position,
            init_price=self.init_price,
            bm_close=(
                self.bm_close
                if (self["bm_close"] is not None and not isinstance(self["bm_close"], bool))
                else self["bm_close"]
            ),
            **kwargs,
        )

    @cachedproperty
    def result(self) -> PFPrepResult:
        """Result as an instance of `PFPrepResult`."""
        return PFPrepResult(target_func=self.target_func, target_args=self.target_args, pf_args=self.pf_args)


BasePFPreparer.override_arg_config_doc(__pdoc__)

order_arg_config = ReadonlyConfig(
    dict(
        size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
            fill_default=False,
        ),
        price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceType.Close)),
        ),
        size_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.SizeType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.SizeType.Amount)),
        ),
        direction=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.Direction),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.Direction.Both)),
        ),
        fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        fixed_fees=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        slippage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        min_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        max_size=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        size_granularity=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        leverage=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=1.0)),
        ),
        leverage_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LeverageMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LeverageMode.Lazy)),
        ),
        reject_prob=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        price_area_vio_mode=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PriceAreaVioMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PriceAreaVioMode.Ignore)),
        ),
        allow_partial=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=True)),
        ),
        raise_reject=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        log=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
    )
)
"""_"""

__pdoc__[
    "order_arg_config"
] = f"""Argument config for order-related information.

```python
{order_arg_config.prettify()}
```
"""

fo_arg_config = ReadonlyConfig(
    dict(
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
    )
)
"""_"""

__pdoc__[
    "fo_arg_config"
] = f"""Argument config for `FOPreparer`.

```python
{fo_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fo_arg_config)
@override_arg_config(order_arg_config)
class FOPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_orders`."""

    _settings_path: tp.SettingsPath = "portfolio.from_orders"

    # ############# Ready arguments ############# #

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Argument `staticized`."""
        raise ValueError("This method doesn't support staticization")

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting."""
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def _pre_max_order_records(self) -> tp.Optional[int]:
        """Argument `max_order_records` before broadcasting."""
        return self["max_order_records"]

    @cachedproperty
    def _pre_max_log_records(self) -> tp.Optional[int]:
        """Argument `max_log_records` before broadcasting."""
        return self["max_log_records"]

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        return self.cs_group_lens

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        size = to_2d_array(self.size)
        if size.shape[0] == 1:
            return None
        first_valid_idx = generic_nb.first_valid_index_nb(size, check_inf=False)
        first_valid_idx = np.where(first_valid_idx == -1, 0, first_valid_idx)
        return first_valid_idx

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        size = to_2d_array(self.size)
        if size.shape[0] == 1:
            return None
        last_valid_idx = generic_nb.last_valid_index_nb(size, check_inf=False)
        last_valid_idx = np.where(last_valid_idx == -1, len(self.wrapper.index), last_valid_idx + 1)
        return last_valid_idx

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Arguments `price` and `from_ago` after broadcasting."""
        price = self._post_price
        from_ago = self._post_from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                next_valid_open_mask = price == enums.PriceType.NextValidOpen
                next_valid_close_mask = price == enums.PriceType.NextValidClose

                if next_valid_open_mask.any() or next_valid_close_mask.any():
                    new_price = np.empty(self.wrapper.shape_2d, np.float_)
                    new_from_ago = np.empty(self.wrapper.shape_2d, np.int_)
                    if next_valid_open_mask.any():
                        open = broadcast_array_to(self.open, self.wrapper.shape_2d)
                    if next_valid_close_mask.any():
                        close = broadcast_array_to(self.close, self.wrapper.shape_2d)

                    for i in range(price.size):
                        price_item = price.item(i)
                        if price_item == enums.PriceType.NextOpen:
                            new_price[:, i] = enums.PriceType.Open
                            new_from_ago[:, i] = 1
                        elif price_item == enums.PriceType.NextClose:
                            new_price[:, i] = enums.PriceType.Close
                            new_from_ago[:, i] = 1
                        elif price_item == enums.PriceType.NextValidOpen:
                            new_price[:, i] = enums.PriceType.Open
                            new_from_ago[:, i] = valid_price_from_ago_1d_nb(open[:, i])
                        elif price_item == enums.PriceType.NextValidClose:
                            new_price[:, i] = enums.PriceType.Close
                            new_from_ago[:, i] = valid_price_from_ago_1d_nb(close[:, i])
                    price = new_price
                    from_ago = new_from_ago

                elif next_open_mask.any() or next_close_mask.any():
                    price = price.astype(np.float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=np.int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Argument `price`."""
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago`."""
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_order_records(self) -> tp.Optional[int]:
        """Argument `max_order_records`."""
        max_order_records = self._pre_max_order_records
        if max_order_records is None:
            _size = self._post_size
            if _size.size == 1:
                max_order_records = self.target_shape[0] * int(not np.isnan(_size.item(0)))
            else:
                if _size.shape[0] == 1 and self.target_shape[0] > 1:
                    max_order_records = self.target_shape[0] * int(np.any(~np.isnan(_size)))
                else:
                    max_order_records = int(np.max(np.sum(~np.isnan(_size), axis=0)))
        return max_order_records

    @cachedproperty
    def max_log_records(self) -> tp.Optional[int]:
        """Argument `max_log_records`."""
        max_log_records = self._pre_max_log_records
        if max_log_records is None:
            _log = self._post_log
            if _log.size == 1:
                max_log_records = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_log_records = self.target_shape[0] * int(np.any(_log))
                else:
                    max_log_records = int(np.max(np.sum(_log, axis=0)))
        return max_log_records

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                group_lens=self.group_lens if self.dynamic_mode else self.cs_group_lens,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        func = jit_reg.resolve_option(nb.from_orders_nb, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        target_arg_map["group_lens"] = "cs_group_lens"
        return target_arg_map


FOPreparer.override_arg_config_doc(__pdoc__)

fs_arg_config = ReadonlyConfig(
    dict(
        size=dict(
            fill_default=True,
        ),
        cash_dividends=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0.0)),
        ),
        entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        long_entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        long_exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_entries=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        short_exits=dict(
            has_default=False,
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        adjust_func_nb=dict(),
        adjust_args=dict(type=tuple, substitute_templates=True),
        signal_func_nb=dict(),
        signal_args=dict(type=tuple, substitute_templates=True),
        post_signal_func_nb=dict(),
        post_signal_args=dict(type=tuple, substitute_templates=True),
        post_segment_func_nb=dict(),
        post_segment_args=dict(type=tuple, substitute_templates=True),
        order_mode=dict(),
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        accumulate=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.AccumulationMode, ignore_type=(int, bool)),
            subdtype=(np.integer, np.bool_),
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.AccumulationMode.Disabled)),
        ),
        upon_long_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_short_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.ConflictMode.Ignore)),
        ),
        upon_dir_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DirectionConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.DirectionConflictMode.Ignore)),
        ),
        upon_opposite_entry=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OppositeEntryMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OppositeEntryMode.ReverseReduce)),
        ),
        order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        limit_tif=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_expiry=dict(
            broadcast=True,
            is_dt=True,
            last_before=False,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        limit_reverse=dict(
            broadcast=True,
            subdtype=np.bool_,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=False)),
        ),
        limit_order_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.LimitOrderPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.LimitOrderPrice.Limit)),
        ),
        upon_adj_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepIgnore)),
        ),
        upon_opp_limit_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.CancelExecute)),
        ),
        use_stops=dict(),
        stop_ladder=dict(map_enum_kwargs=dict(enum=enums.StopLadderMode, look_for_type=str)),
        sl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tsl_th=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        tp_stop=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        td_stop=dict(
            broadcast=True,
            is_td=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        dt_stop=dict(
            broadcast=True,
            is_dt=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=-1)),
        ),
        stop_entry_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopEntryPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopEntryPrice.Close)),
        ),
        stop_exit_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitPrice, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitPrice.Stop)),
        ),
        stop_exit_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopExitType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopExitType.Close)),
        ),
        stop_order_type=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.OrderType),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.OrderType.Market)),
        ),
        stop_limit_delta=dict(
            broadcast=True,
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        upon_stop_update=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.StopUpdateMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.StopUpdateMode.Override)),
        ),
        upon_adj_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)),
        ),
        upon_opp_stop_conflict=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.PendingConflictMode),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.PendingConflictMode.KeepExecute)),
        ),
        delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.DeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.DeltaFormat.Percent)),
        ),
        time_delta_format=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.TimeDeltaFormat),
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=enums.TimeDeltaFormat.Index)),
        ),
        from_ago=dict(
            broadcast=True,
            subdtype=np.integer,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=0)),
        ),
        ffill_val_price=dict(),
        update_value=dict(),
        fill_pos_info=dict(),
        save_state=dict(),
        save_value=dict(),
        save_returns=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
        records=dict(
            rename_fields=dict(
                entry="entries",
                exit="exits",
                long_entry="long_entries",
                long_exit="long_exits",
                short_entry="short_entries",
                short_exit="short_exits",
            )
        ),
    )
)
"""_"""

__pdoc__[
    "fs_arg_config"
] = f"""Argument config for `FSPreparer`.

```python
{fs_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fs_arg_config)
@override_arg_config(order_arg_config)
class FSPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_signals`."""

    _settings_path: tp.SettingsPath = "portfolio.from_signals"

    # ############# Mode resolution ############# #

    @cachedproperty
    def _pre_staticized(self) -> tp.StaticizedOption:
        """Argument `staticized` before its resolution."""
        staticized = self["staticized"]
        if isinstance(staticized, bool):
            if staticized:
                staticized = dict()
            else:
                staticized = None
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if "func" not in staticized:
                staticized["func"] = nb.from_signal_func_nb
        return staticized

    @cachedproperty
    def order_mode(self) -> bool:
        """Argument `order_mode`."""
        order_mode = self["order_mode"]
        if order_mode is None:
            order_mode = False
        return order_mode

    @cachedproperty
    def dynamic_mode(self) -> tp.StaticizedOption:
        """Whether the dynamic mode is enabled."""
        return (
            self["adjust_func_nb"] is not None
            or self["signal_func_nb"] is not None
            or self["post_signal_func_nb"] is not None
            or self["post_segment_func_nb"] is not None
            or self.order_mode
            or self._pre_staticized is not None
        )

    @cachedproperty
    def implicit_mode(self) -> bool:
        """Whether the explicit mode is enabled."""
        return self["entries"] is not None or self["exits"] is not None

    @cachedproperty
    def explicit_mode(self) -> bool:
        """Whether the explicit mode is enabled."""
        return self["long_entries"] is not None or self["long_exits"] is not None

    @cachedproperty
    def _pre_ls_mode(self) -> bool:
        """Whether direction-aware mode is enabled before resolution."""
        return self.explicit_mode or self["short_entries"] is not None or self["short_exits"] is not None

    @cachedproperty
    def _pre_signals_mode(self) -> bool:
        """Whether signals mode is enabled before resolution."""
        return self.implicit_mode or self._pre_ls_mode

    @cachedproperty
    def ls_mode(self) -> bool:
        """Whether direction-aware mode is enabled."""
        if not self._pre_signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        ls_mode = self._pre_ls_mode
        if self.config.get("direction", None) is not None and ls_mode:
            raise ValueError("Direction and short signal arrays cannot be used together")
        return ls_mode

    @cachedproperty
    def signals_mode(self) -> bool:
        """Whether signals mode is enabled."""
        if not self._pre_signals_mode and not self.order_mode and self["signal_func_nb"] is None:
            return True
        signals_mode = self._pre_signals_mode
        if signals_mode and self.order_mode:
            raise ValueError("Signal arrays and order mode cannot be used together")
        return signals_mode

    @cachedproperty
    def signal_func_mode(self) -> bool:
        """Whether signal function mode is enabled."""
        return self.dynamic_mode and not self.signals_mode and not self.order_mode

    @cachedproperty
    def adjust_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `adjust_func_nb`."""
        if self.dynamic_mode:
            if self["adjust_func_nb"] is None:
                return nb.no_adjust_func_nb
            return self["adjust_func_nb"]
        return None

    @cachedproperty
    def signal_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `signal_func_nb`."""
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return nb.ls_signal_func_nb
                if self.signals_mode:
                    return nb.dir_signal_func_nb
                if self.order_mode:
                    return nb.order_signal_func_nb
                return None
            return self["signal_func_nb"]
        return None

    @cachedproperty
    def post_signal_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `post_signal_func_nb`."""
        if self.dynamic_mode:
            if self["post_signal_func_nb"] is None:
                return nb.no_post_func_nb
            return self["post_signal_func_nb"]
        return None

    @cachedproperty
    def post_segment_func_nb(self) -> tp.Optional[tp.Callable]:
        """Argument `post_segment_func_nb`."""
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return nb.save_post_segment_func_nb
                return nb.no_post_func_nb
            return self["post_segment_func_nb"]
        return None

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Argument `staticized`."""
        staticized = self._pre_staticized
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if self.dynamic_mode:
                if self["signal_func_nb"] is None:
                    if self.ls_mode:
                        self.adapt_staticized_to_udf(staticized, "ls_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_ls_signal_func_nb"
                    elif self.signals_mode:
                        self.adapt_staticized_to_udf(staticized, "dir_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_dir_signal_func_nb"
                    elif self.order_mode:
                        self.adapt_staticized_to_udf(staticized, "order_signal_func_nb", "signal_func_nb")
                        staticized["suggest_fname"] = "from_order_signal_func_nb"
                else:
                    self.adapt_staticized_to_udf(staticized, self["signal_func_nb"], "signal_func_nb")
                if self["adjust_func_nb"] is not None:
                    self.adapt_staticized_to_udf(staticized, self["adjust_func_nb"], "adjust_func_nb")
                if self["post_signal_func_nb"] is not None:
                    self.adapt_staticized_to_udf(staticized, self["post_signal_func_nb"], "post_signal_func_nb")
                if self["post_segment_func_nb"] is not None:
                    self.adapt_staticized_to_udf(staticized, self["post_segment_func_nb"], "post_segment_func_nb")
                elif self.save_state or self.save_value or self.save_returns:
                    self.adapt_staticized_to_udf(staticized, "save_post_segment_func_nb", "post_segment_func_nb")
        return staticized

    @cachedproperty
    def _pre_chunked(self) -> tp.ChunkedOption:
        """Argument `chunked` before template substitution."""
        return self["chunked"]

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_entries(self) -> tp.ArrayLike:
        """Argument `entries` before broadcasting."""
        return self["entries"] if self["entries"] is not None else False

    @cachedproperty
    def _pre_exits(self) -> tp.ArrayLike:
        """Argument `exits` before broadcasting."""
        return self["exits"] if self["exits"] is not None else False

    @cachedproperty
    def _pre_long_entries(self) -> tp.ArrayLike:
        """Argument `long_entries` before broadcasting."""
        return self["long_entries"] if self["long_entries"] is not None else False

    @cachedproperty
    def _pre_long_exits(self) -> tp.ArrayLike:
        """Argument `long_exits` before broadcasting."""
        return self["long_exits"] if self["long_exits"] is not None else False

    @cachedproperty
    def _pre_short_entries(self) -> tp.ArrayLike:
        """Argument `short_entries` before broadcasting."""
        return self["short_entries"] if self["short_entries"] is not None else False

    @cachedproperty
    def _pre_short_exits(self) -> tp.ArrayLike:
        """Argument `short_exits` before broadcasting."""
        return self["short_exits"] if self["short_exits"] is not None else False

    @cachedproperty
    def _pre_from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago` before broadcasting."""
        from_ago = self["from_ago"]
        if from_ago is not None:
            return from_ago
        return 0

    @cachedproperty
    def _pre_max_log_records(self) -> tp.Optional[int]:
        """Argument `max_log_records` before broadcasting."""
        return self["max_log_records"]

    @classmethod
    def init_in_outputs(
        cls,
        wrapper: ArrayWrapper,
        group_lens: tp.Optional[tp.GroupLens] = None,
        cash_sharing: bool = False,
        save_state: bool = True,
        save_value: bool = True,
        save_returns: bool = True,
    ) -> enums.FSInOutputs:
        """Initialize `vectorbtpro.portfolio.enums.FSInOutputs`."""
        if cash_sharing:
            if group_lens is None:
                group_lens = wrapper.grouper.get_group_lens()
        return nb.init_FSInOutputs_nb(
            wrapper.shape_2d,
            group_lens,
            cash_sharing=cash_sharing,
            save_state=save_state,
            save_value=save_value,
            save_returns=save_returns,
        )

    @cachedproperty
    def _pre_in_outputs(self) -> tp.Union[None, tp.NamedTuple, CustomTemplate]:
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return RepFunc(self.init_in_outputs)
            return BasePFPreparer._pre_in_outputs.func(self)
        if self["in_outputs"] is not None:
            raise ValueError("Argument in_outputs cannot be used in fixed mode")
        return None

    # ############# Broadcasting ############# #

    @cachedproperty
    def def_broadcast_kwargs(self) -> tp.Kwargs:
        def_broadcast_kwargs = dict(BasePFPreparer.def_broadcast_kwargs.func(self))
        new_def_broadcast_kwargs = dict()
        if self.order_mode:
            new_def_broadcast_kwargs["keep_flex"] = dict(
                size=False,
                size_type=False,
                min_size=False,
                max_size=False,
            )
            new_def_broadcast_kwargs["min_ndim"] = dict(
                size=2,
                size_type=2,
                min_size=2,
                max_size=2,
            )
            new_def_broadcast_kwargs["require_kwargs"] = dict(
                size=dict(requirements="O"),
                size_type=dict(requirements="O"),
                min_size=dict(requirements="O"),
                max_size=dict(requirements="O"),
            )
        if self.stop_ladder:
            new_def_broadcast_kwargs["axis"] = dict(
                sl_stop=1,
                tsl_stop=1,
                tp_stop=1,
                td_stop=1,
                dt_stop=1,
            )
            new_def_broadcast_kwargs["merge_kwargs"] = dict(
                sl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tsl_stop=dict(reset_index="from_start", fill_value=np.nan),
                tp_stop=dict(reset_index="from_start", fill_value=np.nan),
                td_stop=dict(reset_index="from_start", fill_value=-1),
                dt_stop=dict(reset_index="from_start", fill_value=-1),
            )
        return merge_dicts(def_broadcast_kwargs, new_def_broadcast_kwargs)

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_group_lens(self) -> tp.GroupLens:
        if not self.dynamic_mode:
            return self.cs_group_lens
        return self.group_lens

    @cachedproperty
    def signals(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike, tp.ArrayLike, tp.ArrayLike]:
        """Arguments `entries`, `exits`, `short_entries`, and `short_exits` after broadcasting."""
        if not self.dynamic_mode and not self.ls_mode:
            entries = self._post_entries
            exits = self._post_exits
            direction = self._post_direction
            if direction.size == 1:
                _direction = direction.item(0)
                if _direction == enums.Direction.LongOnly:
                    long_entries = entries
                    long_exits = exits
                    short_entries = np.array([[False]])
                    short_exits = np.array([[False]])
                elif _direction == enums.Direction.ShortOnly:
                    long_entries = np.array([[False]])
                    long_exits = np.array([[False]])
                    short_entries = entries
                    short_exits = exits
                else:
                    long_entries = entries
                    long_exits = np.array([[False]])
                    short_entries = exits
                    short_exits = np.array([[False]])
            else:
                return nb.dir_to_ls_signals_nb(
                    target_shape=self.target_shape,
                    entries=entries,
                    exits=exits,
                    direction=direction,
                )
        else:
            if self.explicit_mode and self.implicit_mode:
                long_entries = self._post_entries | self._post_long_entries
                long_exits = self._post_exits | self._post_long_exits
                short_entries = self._post_entries | self._post_short_entries
                short_exits = self._post_exits | self._post_short_exits
            elif self.explicit_mode:
                long_entries = self._post_long_entries
                long_exits = self._post_long_exits
                short_entries = self._post_short_entries
                short_exits = self._post_short_exits
            else:
                long_entries = self._post_entries
                long_exits = self._post_exits
                short_entries = self._post_short_entries
                short_exits = self._post_short_exits
        return long_entries, long_exits, short_entries, short_exits

    @cachedproperty
    def long_entries(self) -> tp.ArrayLike:
        """Argument `long_entries`."""
        return self.signals[0]

    @cachedproperty
    def long_exits(self) -> tp.ArrayLike:
        """Argument `long_exits`."""
        return self.signals[1]

    @cachedproperty
    def short_entries(self) -> tp.ArrayLike:
        """Argument `short_entries`."""
        return self.signals[2]

    @cachedproperty
    def short_exits(self) -> tp.ArrayLike:
        """Argument `short_exits`."""
        return self.signals[3]

    @cachedproperty
    def combined_mask(self) -> tp.Array2d:
        """Signals combined using the OR rule into a mask."""
        long_entries = to_2d_array(self.long_entries)
        long_exits = to_2d_array(self.long_exits)
        short_entries = to_2d_array(self.short_entries)
        short_exits = to_2d_array(self.short_exits)
        return long_entries | long_exits | short_entries | short_exits

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        if self.combined_mask.shape[0] == 1:
            return None
        first_signal_idx = signals_nb.nth_index_nb(self.combined_mask, 0)
        return np.where(first_signal_idx == -1, 0, first_signal_idx)

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        if self.combined_mask.shape[0] == 1:
            return None
        last_signal_idx = signals_nb.nth_index_nb(self.combined_mask, -1)
        return np.where(last_signal_idx == -1, len(self.wrapper.index), last_signal_idx + 1)

    @cachedproperty
    def price_and_from_ago(self) -> tp.Tuple[tp.ArrayLike, tp.ArrayLike]:
        """Arguments `price` and `from_ago` after broadcasting."""
        price = self._post_price
        from_ago = self._post_from_ago
        if self["from_ago"] is None:
            if price.size == 1 or price.shape[0] == 1:
                next_open_mask = price == enums.PriceType.NextOpen
                next_close_mask = price == enums.PriceType.NextClose
                if next_open_mask.any() or next_close_mask.any():
                    price = price.astype(np.float_)
                    price[next_open_mask] = enums.PriceType.Open
                    price[next_close_mask] = enums.PriceType.Close
                    from_ago = np.full(price.shape, 0, dtype=np.int_)
                    from_ago[next_open_mask] = 1
                    from_ago[next_close_mask] = 1
        return price, from_ago

    @cachedproperty
    def price(self) -> tp.ArrayLike:
        """Argument `price`."""
        return self.price_and_from_ago[0]

    @cachedproperty
    def from_ago(self) -> tp.ArrayLike:
        """Argument `from_ago`."""
        return self.price_and_from_ago[1]

    @cachedproperty
    def max_log_records(self) -> tp.Optional[int]:
        """Argument `max_log_records`."""
        max_log_records = self._pre_max_log_records
        if max_log_records is None:
            _log = self._post_log
            if _log.size == 1:
                max_log_records = self.target_shape[0] * int(_log.item(0))
            else:
                if _log.shape[0] == 1 and self.target_shape[0] > 1:
                    max_log_records = self.target_shape[0] * int(np.any(_log))
                else:
                    max_log_records = int(np.max(np.sum(_log, axis=0)))
        return max_log_records

    @cachedproperty
    def use_stops(self) -> bool:
        """Argument `use_stops`."""
        if self["use_stops"] is None:
            if self.stop_ladder:
                use_stops = True
            else:
                if self.dynamic_mode:
                    use_stops = True
                else:
                    if (
                        not np.all(np.isnan(self.sl_stop))
                        or not np.all(np.isnan(self.tsl_stop))
                        or not np.all(np.isnan(self.tp_stop))
                        or np.any(self.td_stop != -1)
                        or np.any(self.dt_stop != -1)
                    ):
                        use_stops = True
                    else:
                        use_stops = False
        else:
            use_stops = self["use_stops"]
        return use_stops

    @cachedproperty
    def use_limit_orders(self) -> bool:
        """Whether to use limit orders."""
        if np.any(self.order_type == enums.OrderType.Limit):
            return True
        if self.use_stops and np.any(self.stop_order_type == enums.OrderType.Limit):
            return True
        return False

    @cachedproperty
    def basic_mode(self) -> bool:
        """Whether the basic mode is enabled."""
        return not self.use_stops and not self.use_limit_orders

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                order_mode=self.order_mode,
                use_stops=self.use_stops,
                stop_ladder=self.stop_ladder,
                adjust_func_nb=self.adjust_func_nb,
                adjust_args=self._pre_adjust_args,
                signal_func_nb=self.signal_func_nb,
                signal_args=self._pre_signal_args,
                post_signal_func_nb=self.post_signal_func_nb,
                post_signal_args=self._pre_post_signal_args,
                post_segment_func_nb=self.post_segment_func_nb,
                post_segment_args=self._pre_post_segment_args,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                fill_pos_info=self.fill_pos_info,
                save_state=self.save_state,
                save_value=self.save_value,
                save_returns=self.save_returns,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    @cachedproperty
    def signal_args(self) -> tp.Args:
        """Argument `signal_args`."""
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return (
                        self.long_entries,
                        self.long_exits,
                        self.short_entries,
                        self.short_exits,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
                if self.signals_mode:
                    return (
                        self.entries,
                        self.exits,
                        self.direction,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
                if self.order_mode:
                    return (
                        self.size,
                        self.price,
                        self.size_type,
                        self.direction,
                        self.min_size,
                        self.max_size,
                        self.val_price,
                        self.from_ago,
                        *((self.adjust_func_nb,) if self.staticized is None else ()),
                        self.adjust_args,
                    )
        return self._post_signal_args

    @cachedproperty
    def post_segment_args(self) -> tp.Args:
        """Argument `post_segment_args`."""
        if self.dynamic_mode:
            if self["post_segment_func_nb"] is None:
                if self.save_state or self.save_value or self.save_returns:
                    return (
                        self.save_state,
                        self.save_value,
                        self.save_returns,
                    )
        return self._post_post_segment_args

    @cachedproperty
    def chunked(self) -> tp.ChunkedOption:
        if self.dynamic_mode:
            if self["signal_func_nb"] is None:
                if self.ls_mode:
                    return ch.specialize_chunked_option(
                        self._pre_chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
                if self.signals_mode:
                    return ch.specialize_chunked_option(
                        self._pre_chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
                if self.order_mode:
                    return ch.specialize_chunked_option(
                        self._pre_chunked,
                        arg_take_spec=dict(
                            signal_args=ch.ArgsTaker(
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                base_ch.flex_array_gl_slicer,
                                *((None,) if self.staticized is None else ()),
                                ch.ArgsTaker(),
                            )
                        ),
                    )
        return self._pre_chunked

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        if self.dynamic_mode:
            func = self.resolve_dynamic_target_func("from_signal_func_nb", self.staticized)
        elif not self.basic_mode:
            func = nb.from_signals_nb
        else:
            func = nb.from_basic_signals_nb
        func = jit_reg.resolve_option(func, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        if self.dynamic_mode:
            if self.staticized is not None:
                target_arg_map["signal_func_nb"] = None
                target_arg_map["post_signal_func_nb"] = None
                target_arg_map["post_segment_func_nb"] = None
        else:
            target_arg_map["group_lens"] = "cs_group_lens"
        return target_arg_map

    @cachedproperty
    def pf_args(self) -> tp.Optional[tp.Kwargs]:
        pf_args = dict(BasePFPreparer.pf_args.func(self))
        pf_args["orders_cls"] = FSOrders
        return pf_args


FSPreparer.override_arg_config_doc(__pdoc__)

fof_arg_config = ReadonlyConfig(
    dict(
        segment_mask=dict(),
        call_pre_segment=dict(),
        call_post_segment=dict(),
        pre_sim_func_nb=dict(),
        pre_sim_args=dict(type=tuple, substitute_templates=True),
        post_sim_func_nb=dict(),
        post_sim_args=dict(type=tuple, substitute_templates=True),
        pre_group_func_nb=dict(),
        pre_group_args=dict(type=tuple, substitute_templates=True),
        post_group_func_nb=dict(),
        post_group_args=dict(type=tuple, substitute_templates=True),
        pre_row_func_nb=dict(),
        pre_row_args=dict(type=tuple, substitute_templates=True),
        post_row_func_nb=dict(),
        post_row_args=dict(type=tuple, substitute_templates=True),
        pre_segment_func_nb=dict(),
        pre_segment_args=dict(type=tuple, substitute_templates=True),
        post_segment_func_nb=dict(),
        post_segment_args=dict(type=tuple, substitute_templates=True),
        order_func_nb=dict(),
        order_args=dict(type=tuple, substitute_templates=True),
        flex_order_func_nb=dict(),
        flex_order_args=dict(type=tuple, substitute_templates=True),
        post_order_func_nb=dict(),
        post_order_args=dict(type=tuple, substitute_templates=True),
        ffill_val_price=dict(),
        update_value=dict(),
        fill_pos_info=dict(),
        track_value=dict(),
        row_wise=dict(),
        max_order_records=dict(),
        max_log_records=dict(),
    )
)
"""_"""

__pdoc__[
    "fof_arg_config"
] = f"""Argument config for `FOFPreparer`.

```python
{fof_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fof_arg_config)
class FOFPreparer(BasePFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_order_func`."""

    _settings_path: tp.SettingsPath = "portfolio.from_order_func"

    # ############# Mode resolution ############# #

    @cachedproperty
    def _pre_staticized(self) -> tp.StaticizedOption:
        """Argument `staticized` before its resolution."""
        staticized = self["staticized"]
        if isinstance(staticized, bool):
            if staticized:
                staticized = dict()
            else:
                staticized = None
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if "func" not in staticized:
                if not self.flexible and not self.row_wise:
                    staticized["func"] = nb.from_order_func_nb
                elif not self.flexible and self.row_wise:
                    staticized["func"] = nb.from_order_func_rw_nb
                elif self.flexible and not self.row_wise:
                    staticized["func"] = nb.from_flex_order_func_nb
                else:
                    staticized["func"] = nb.from_flex_order_func_rw_nb
        return staticized

    @cachedproperty
    def flexible(self) -> bool:
        """Whether the flexible mode is enabled."""
        return self["flex_order_func_nb"] is not None

    @cachedproperty
    def pre_sim_func_nb(self) -> tp.Callable:
        """Argument `pre_sim_func_nb`."""
        pre_sim_func_nb = self["pre_sim_func_nb"]
        if pre_sim_func_nb is None:
            pre_sim_func_nb = nb.no_pre_func_nb
        return pre_sim_func_nb

    @cachedproperty
    def post_sim_func_nb(self) -> tp.Callable:
        """Argument `post_sim_func_nb`."""
        post_sim_func_nb = self["post_sim_func_nb"]
        if post_sim_func_nb is None:
            post_sim_func_nb = nb.no_post_func_nb
        return post_sim_func_nb

    @cachedproperty
    def pre_group_func_nb(self) -> tp.Callable:
        """Argument `pre_group_func_nb`."""
        pre_group_func_nb = self["pre_group_func_nb"]
        if self.row_wise and pre_group_func_nb is not None:
            raise ValueError("Cannot use pre_group_func_nb in a row-wise simulation")
        if pre_group_func_nb is None:
            pre_group_func_nb = nb.no_pre_func_nb
        return pre_group_func_nb

    @cachedproperty
    def post_group_func_nb(self) -> tp.Callable:
        """Argument `post_group_func_nb`."""
        post_group_func_nb = self["post_group_func_nb"]
        if self.row_wise and post_group_func_nb is not None:
            raise ValueError("Cannot use post_group_func_nb in a row-wise simulation")
        if post_group_func_nb is None:
            post_group_func_nb = nb.no_post_func_nb
        return post_group_func_nb

    @cachedproperty
    def pre_row_func_nb(self) -> tp.Callable:
        """Argument `pre_row_func_nb`."""
        pre_row_func_nb = self["pre_row_func_nb"]
        if not self.row_wise and pre_row_func_nb is not None:
            raise ValueError("Cannot use pre_row_func_nb in a column-wise simulation")
        if pre_row_func_nb is None:
            pre_row_func_nb = nb.no_pre_func_nb
        return pre_row_func_nb

    @cachedproperty
    def post_row_func_nb(self) -> tp.Callable:
        """Argument `post_row_func_nb`."""
        post_row_func_nb = self["post_row_func_nb"]
        if not self.row_wise and post_row_func_nb is not None:
            raise ValueError("Cannot use post_row_func_nb in a column-wise simulation")
        if post_row_func_nb is None:
            post_row_func_nb = nb.no_post_func_nb
        return post_row_func_nb

    @cachedproperty
    def pre_segment_func_nb(self) -> tp.Callable:
        """Argument `pre_segment_func_nb`."""
        pre_segment_func_nb = self["pre_segment_func_nb"]
        if pre_segment_func_nb is None:
            pre_segment_func_nb = nb.no_pre_func_nb
        return pre_segment_func_nb

    @cachedproperty
    def post_segment_func_nb(self) -> tp.Callable:
        """Argument `post_segment_func_nb`."""
        post_segment_func_nb = self["post_segment_func_nb"]
        if post_segment_func_nb is None:
            post_segment_func_nb = nb.no_post_func_nb
        return post_segment_func_nb

    @cachedproperty
    def order_func_nb(self) -> tp.Callable:
        """Argument `order_func_nb`."""
        order_func_nb = self["order_func_nb"]
        if self.flexible and order_func_nb is not None:
            raise ValueError("Must provide either order_func_nb or flex_order_func_nb")
        if not self.flexible and order_func_nb is None:
            raise ValueError("Must provide either order_func_nb or flex_order_func_nb")
        if order_func_nb is None:
            order_func_nb = nb.no_order_func_nb
        return order_func_nb

    @cachedproperty
    def flex_order_func_nb(self) -> tp.Callable:
        """Argument `flex_order_func_nb`."""
        flex_order_func_nb = self["flex_order_func_nb"]
        if flex_order_func_nb is None:
            flex_order_func_nb = nb.no_flex_order_func_nb
        return flex_order_func_nb

    @cachedproperty
    def post_order_func_nb(self) -> tp.Callable:
        """Argument `post_order_func_nb`."""
        post_order_func_nb = self["post_order_func_nb"]
        if post_order_func_nb is None:
            post_order_func_nb = nb.no_post_func_nb
        return post_order_func_nb

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        """Argument `staticized`."""
        staticized = self._pre_staticized
        if isinstance(staticized, dict):
            staticized = dict(staticized)
            if self["pre_sim_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_sim_func_nb"], "pre_sim_func_nb")
            if self["post_sim_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["post_sim_func_nb"], "post_sim_func_nb")
            if self["pre_group_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_group_func_nb"], "pre_group_func_nb")
            if self["post_group_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["post_group_func_nb"], "post_group_func_nb")
            if self["pre_row_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_row_func_nb"], "pre_row_func_nb")
            if self["post_row_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["post_row_func_nb"], "post_row_func_nb")
            if self["pre_segment_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["pre_segment_func_nb"], "pre_segment_func_nb")
            if self["post_segment_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["post_segment_func_nb"], "post_segment_func_nb")
            if self["order_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["order_func_nb"], "order_func_nb")
            if self["flex_order_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["flex_order_func_nb"], "flex_order_func_nb")
            if self["post_order_func_nb"] is not None:
                self.adapt_staticized_to_udf(staticized, self["post_order_func_nb"], "post_order_func_nb")
        return staticized

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_call_seq(self) -> tp.Optional[tp.ArrayLike]:
        if self.auto_call_seq:
            raise ValueError(
                "CallSeqType.Auto must be implemented manually. Use sort_call_seq_nb in pre_segment_func_nb."
            )
        return self["call_seq"]

    @cachedproperty
    def _pre_segment_mask(self) -> tp.ArrayLike:
        """Argument `segment_mask` before broadcasting."""
        return self["segment_mask"]

    # ############# After broadcasting ############# #

    @cachedproperty
    def sim_start(self) -> tp.Optional[tp.ArrayLike]:
        sim_start = self["sim_start"]
        if sim_start is None:
            return None
        return BasePFPreparer.sim_start.func(self)

    @cachedproperty
    def sim_end(self) -> tp.Optional[tp.ArrayLike]:
        sim_end = self["sim_end"]
        if sim_end is None:
            return None
        return BasePFPreparer.sim_end.func(self)

    @cachedproperty
    def segment_mask(self) -> tp.ArrayLike:
        """Argument `segment_mask`."""
        segment_mask = self._pre_segment_mask
        if checks.is_int(segment_mask):
            if self.keep_inout_flex:
                _segment_mask = np.full((self.target_shape[0], 1), False)
            else:
                _segment_mask = np.full((self.target_shape[0], len(self.group_lens)), False)
            _segment_mask[0::segment_mask] = True
            segment_mask = _segment_mask
        else:
            segment_mask = broadcast(
                segment_mask,
                to_shape=(self.target_shape[0], len(self.group_lens)),
                to_pd=False,
                keep_flex=self.keep_inout_flex,
                reindex_kwargs=dict(fill_value=False),
                require_kwargs=self.broadcast_kwargs.get("require_kwargs", {}),
            )
        checks.assert_subdtype(segment_mask, np.bool_, arg_name="segment_mask")
        return segment_mask

    # ############# Template substitution ############# #

    @cachedproperty
    def template_context(self) -> tp.Kwargs:
        return merge_dicts(
            dict(
                segment_mask=self.segment_mask,
                call_pre_segment=self.call_pre_segment,
                call_post_segment=self.call_post_segment,
                pre_sim_func_nb=self.pre_sim_func_nb,
                pre_sim_args=self._pre_pre_sim_args,
                post_sim_func_nb=self.post_sim_func_nb,
                post_sim_args=self._pre_post_sim_args,
                pre_group_func_nb=self.pre_group_func_nb,
                pre_group_args=self._pre_pre_group_args,
                post_group_func_nb=self.post_group_func_nb,
                post_group_args=self._pre_post_group_args,
                pre_row_func_nb=self.pre_row_func_nb,
                pre_row_args=self._pre_pre_row_args,
                post_row_func_nb=self.post_row_func_nb,
                post_row_args=self._pre_post_row_args,
                pre_segment_func_nb=self.pre_segment_func_nb,
                pre_segment_args=self._pre_pre_segment_args,
                post_segment_func_nb=self.post_segment_func_nb,
                post_segment_args=self._pre_post_segment_args,
                order_func_nb=self.order_func_nb,
                order_args=self._pre_order_args,
                flex_order_func_nb=self.flex_order_func_nb,
                flex_order_args=self._pre_flex_order_args,
                post_order_func_nb=self.post_order_func_nb,
                post_order_args=self._pre_post_order_args,
                ffill_val_price=self.ffill_val_price,
                update_value=self.update_value,
                fill_pos_info=self.fill_pos_info,
                track_value=self.track_value,
                max_order_records=self.max_order_records,
                max_log_records=self.max_log_records,
            ),
            BasePFPreparer.template_context.func(self),
        )

    # ############# Result ############# #

    @cachedproperty
    def target_func(self) -> tp.Optional[tp.Callable]:
        if not self.row_wise and not self.flexible:
            func = self.resolve_dynamic_target_func("from_order_func_nb", self.staticized)
        elif not self.row_wise and self.flexible:
            func = self.resolve_dynamic_target_func("from_flex_order_func_nb", self.staticized)
        elif self.row_wise and not self.flexible:
            func = self.resolve_dynamic_target_func("from_order_func_rw_nb", self.staticized)
        else:
            func = self.resolve_dynamic_target_func("from_flex_order_func_rw_nb", self.staticized)
        func = jit_reg.resolve_option(func, self.jitted)
        func = ch_reg.resolve_option(func, self.chunked)
        return func

    @cachedproperty
    def target_arg_map(self) -> tp.Kwargs:
        target_arg_map = dict(BasePFPreparer.target_arg_map.func(self))
        if self.staticized is not None:
            target_arg_map["pre_sim_func_nb"] = None
            target_arg_map["post_sim_func_nb"] = None
            target_arg_map["pre_group_func_nb"] = None
            target_arg_map["post_group_func_nb"] = None
            target_arg_map["pre_row_func_nb"] = None
            target_arg_map["post_row_func_nb"] = None
            target_arg_map["pre_segment_func_nb"] = None
            target_arg_map["post_segment_func_nb"] = None
            target_arg_map["order_func_nb"] = None
            target_arg_map["flex_order_func_nb"] = None
            target_arg_map["post_order_func_nb"] = None
        return target_arg_map


fdof_arg_config = ReadonlyConfig(
    dict(
        val_price=dict(
            broadcast=True,
            map_enum_kwargs=dict(enum=enums.ValPriceType, ignore_type=(int, float)),
            subdtype=np.number,
            broadcast_kwargs=dict(reindex_kwargs=dict(fill_value=np.nan)),
        ),
        flexible=dict(),
    )
)
"""_"""

__pdoc__[
    "fdof_arg_config"
] = f"""Argument config for `FDOFPreparer`.

```python
{fdof_arg_config.prettify()}
```
"""


@attach_arg_properties
@override_arg_config(fdof_arg_config)
@override_arg_config(order_arg_config)
class FDOFPreparer(FOFPreparer):
    """Class for preparing `vectorbtpro.portfolio.base.Portfolio.from_def_order_func`."""

    _settings_path: tp.SettingsPath = "portfolio.from_def_order_func"

    # ############# Mode resolution ############# #

    @cachedproperty
    def flexible(self) -> bool:
        return self["flexible"]

    @cachedproperty
    def pre_segment_func_nb(self) -> tp.Callable:
        """Argument `pre_segment_func_nb`."""
        pre_segment_func_nb = self["pre_segment_func_nb"]
        if pre_segment_func_nb is None:
            if self.flexible:
                pre_segment_func_nb = nb.def_flex_pre_segment_func_nb
            else:
                pre_segment_func_nb = nb.def_pre_segment_func_nb
        return pre_segment_func_nb

    @cachedproperty
    def order_func_nb(self) -> tp.Callable:
        """Argument `order_func_nb`."""
        order_func_nb = self["order_func_nb"]
        if self.flexible and order_func_nb is not None:
            raise ValueError("Argument order_func_nb cannot be provided when flexible=True")
        if order_func_nb is None:
            order_func_nb = nb.def_order_func_nb
        return order_func_nb

    @cachedproperty
    def flex_order_func_nb(self) -> tp.Callable:
        """Argument `flex_order_func_nb`."""
        flex_order_func_nb = self["flex_order_func_nb"]
        if not self.flexible and flex_order_func_nb is not None:
            raise ValueError("Argument flex_order_func_nb cannot be provided when flexible=False")
        if flex_order_func_nb is None:
            flex_order_func_nb = nb.def_flex_order_func_nb
        return flex_order_func_nb

    @cachedproperty
    def _pre_chunked(self) -> tp.ChunkedOption:
        """Argument `chunked` before template substitution."""
        return self["chunked"]

    @cachedproperty
    def staticized(self) -> tp.StaticizedOption:
        staticized = FOFPreparer.staticized.func(self)
        if isinstance(staticized, dict):
            if "pre_segment_func_nb" not in staticized:
                self.adapt_staticized_to_udf(staticized, self.pre_segment_func_nb, "pre_segment_func_nb")
            if "order_func_nb" not in staticized:
                self.adapt_staticized_to_udf(staticized, self.order_func_nb, "order_func_nb")
            if "flex_order_func_nb" not in staticized:
                self.adapt_staticized_to_udf(staticized, self.flex_order_func_nb, "flex_order_func_nb")
        return staticized

    # ############# Before broadcasting ############# #

    @cachedproperty
    def _pre_call_seq(self) -> tp.Optional[tp.ArrayLike]:
        return BasePFPreparer._pre_call_seq.func(self)

    # ############# After broadcasting ############# #

    @cachedproperty
    def auto_sim_start(self) -> tp.Optional[tp.ArrayLike]:
        return FOPreparer.auto_sim_start.func(self)

    @cachedproperty
    def auto_sim_end(self) -> tp.Optional[tp.ArrayLike]:
        return FOPreparer.auto_sim_end.func(self)

    # ############# Template substitution ############# #

    @cachedproperty
    def pre_segment_args(self) -> tp.Args:
        """Argument `pre_segment_args`."""
        return (
            self.val_price,
            self.price,
            self.size,
            self.size_type,
            self.direction,
            self.auto_call_seq,
        )

    @cachedproperty
    def _order_args(self) -> tp.Args:
        """Either `order_args` or `flex_order_args`."""
        return (
            self.size,
            self.price,
            self.size_type,
            self.direction,
            self.fees,
            self.fixed_fees,
            self.slippage,
            self.min_size,
            self.max_size,
            self.size_granularity,
            self.leverage,
            self.leverage_mode,
            self.reject_prob,
            self.price_area_vio_mode,
            self.allow_partial,
            self.raise_reject,
            self.log,
        )

    @cachedproperty
    def order_args(self) -> tp.Args:
        """Argument `order_args`."""
        if self.flexible:
            return self._post_order_args
        return self._order_args

    @cachedproperty
    def flex_order_args(self) -> tp.Args:
        """Argument `flex_order_args`."""
        if not self.flexible:
            return self._post_flex_order_args
        return self._order_args

    @cachedproperty
    def chunked(self) -> tp.ChunkedOption:
        arg_take_spec = dict()
        arg_take_spec["pre_segment_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 5, None)
        if self.flexible:
            arg_take_spec["flex_order_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 17)
        else:
            arg_take_spec["order_args"] = ch.ArgsTaker(*[base_ch.flex_array_gl_slicer] * 17)
        return ch.specialize_chunked_option(self._pre_chunked, arg_take_spec=arg_take_spec)
