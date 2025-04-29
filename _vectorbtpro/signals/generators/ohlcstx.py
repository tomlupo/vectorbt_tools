# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `OHLCSTX`."""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.signals.enums import StopType
from vectorbtpro.signals.factory import SignalFactory
from vectorbtpro.signals.nb import ohlc_stop_place_nb
from vectorbtpro.utils.config import ReadonlyConfig, merge_dicts

__all__ = [
    "OHLCSTX",
]

ohlcstx_config = ReadonlyConfig(
    dict(
        class_name="OHLCSTX",
        module_name=__name__,
        short_name="ohlcstx",
        mode="exits",
        input_names=["entry_price", "open", "high", "low", "close"],
        in_output_names=["stop_price", "stop_type"],
        param_names=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
        attr_settings=dict(
            stop_type=dict(dtype=StopType),
        ),
    )
)
"""Factory config for `OHLCSTX`."""

ohlcstx_func_config = ReadonlyConfig(
    dict(
        exit_place_func_nb=ohlc_stop_place_nb,
        exit_settings=dict(
            pass_inputs=["entry_price", "open", "high", "low", "close"],
            pass_in_outputs=["stop_price", "stop_type"],
            pass_params=["sl_stop", "tsl_th", "tsl_stop", "tp_stop", "reverse"],
            pass_kwargs=["is_entry_open"],
        ),
        in_output_settings=dict(
            stop_price=dict(dtype=np.float_),
            stop_type=dict(dtype=np.int_),
        ),
        param_settings=dict(
            sl_stop=flex_elem_param_config,
            tsl_th=flex_elem_param_config,
            tsl_stop=flex_elem_param_config,
            tp_stop=flex_elem_param_config,
            reverse=flex_elem_param_config,
        ),
        open=np.nan,
        high=np.nan,
        low=np.nan,
        close=np.nan,
        stop_price=np.nan,
        stop_type=-1,
        sl_stop=np.nan,
        tsl_th=np.nan,
        tsl_stop=np.nan,
        tp_stop=np.nan,
        reverse=False,
        is_entry_open=False,
    )
)
"""Exit function config for `OHLCSTX`."""

OHLCSTX = SignalFactory(**ohlcstx_config).with_place_func(**ohlcstx_func_config)


def bind_ohlcstx_plot(base_cls: type, entries_attr: str) -> tp.Callable:
    base_cls_plot = base_cls.plot

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        ohlc_kwargs: tp.KwargsLike = None,
        entry_price_kwargs: tp.KwargsLike = None,
        entry_trace_kwargs: tp.KwargsLike = None,
        exit_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        _base_cls_plot: tp.Callable = base_cls_plot,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        self_col = self.select_col(column=column, group_by=False)

        if ohlc_kwargs is None:
            ohlc_kwargs = {}
        if entry_price_kwargs is None:
            entry_price_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        open_any = not self_col.open.isnull().all()
        high_any = not self_col.high.isnull().all()
        low_any = not self_col.low.isnull().all()
        close_any = not self_col.close.isnull().all()
        if open_any and high_any and low_any and close_any:
            ohlc_df = pd.concat((self_col.open, self_col.high, self_col.low, self_col.close), axis=1)
            ohlc_df.columns = ["Open", "High", "Low", "Close"]
            ohlc_kwargs = merge_dicts(layout_kwargs, dict(ohlc_trace_kwargs=dict(opacity=0.5)), ohlc_kwargs)
            fig = ohlc_df.vbt.ohlcv.plot(fig=fig, **ohlc_kwargs)
        else:
            entry_price_kwargs = merge_dicts(layout_kwargs, entry_price_kwargs)
            fig = self_col.entry_price.rename("Entry price").vbt.lineplot(fig=fig, **entry_price_kwargs)

        _base_cls_plot(
            self_col,
            entry_y=self_col.entry_price,
            exit_y=self_col.stop_price,
            exit_types=self_col.stop_type_readable,
            entry_trace_kwargs=entry_trace_kwargs,
            exit_trace_kwargs=exit_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )
        return fig

    plot.__doc__ = """Plot OHLC, `{0}.{1}` and `{0}.exits`.

    Args:
        ohlc_kwargs (dict): Keyword arguments passed to
            `vectorbtpro.ohlcv.accessors.OHLCVDFAccessor.plot`.
        entry_trace_kwargs (dict): Keyword arguments passed to
            `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_entries` for `{0}.{1}`.
        exit_trace_kwargs (dict): Keyword arguments passed to
            `vectorbtpro.signals.accessors.SignalsSRAccessor.plot_as_exits` for `{0}.exits`.
        fig (Figure or FigureWidget): Figure to add traces to.
        **layout_kwargs: Keyword arguments for layout.""".format(
        base_cls.__name__,
        entries_attr,
    )

    if entries_attr == "entries":
        plot.__doc__ += """
    Usage:
        ```pycon
        >>> ohlcstx.iloc[:, 0].plot().show()
        ```

        ![](/assets/images/api/OHLCSTX.light.svg#only-light){: .iimg loading=lazy }
        ![](/assets/images/api/OHLCSTX.dark.svg#only-dark){: .iimg loading=lazy }
    """
    return plot


class _OHLCSTX(OHLCSTX):
    """Exit signal generator based on OHLC and stop values.

    Generates `exits` based on `entries` and `vectorbtpro.signals.nb.ohlc_stop_place_nb`.

    !!! hint
        All parameters can be either a single value (per frame) or a NumPy array (per row, column,
        or element). To generate multiple combinations, pass them as lists.

    !!! warning
        Searches for an exit after each entry. If two entries come one after another, no exit can be placed.
        Consider either cleaning up entry signals prior to passing, or using `OHLCSTCX`.

    Usage:
        Test each stop type:

        ```pycon
        >>> from vectorbtpro import *

        >>> entries = pd.Series([True, False, False, False, False, False])
        >>> price = pd.DataFrame({
        ...     'open': [10, 11, 12, 11, 10, 9],
        ...     'high': [11, 12, 13, 12, 11, 10],
        ...     'low': [9, 10, 11, 10, 9, 8],
        ...     'close': [10, 11, 12, 11, 10, 9]
        ... })
        >>> ohlcstx = vbt.OHLCSTX.run(
        ...     entries,
        ...     price['open'],
        ...     price['open'],
        ...     price['high'],
        ...     price['low'],
        ...     price['close'],
        ...     sl_stop=[0.1, np.nan, np.nan, np.nan],
        ...     tsl_th=[np.nan, np.nan, 0.2, np.nan],
        ...     tsl_stop=[np.nan, 0.1, 0.3, np.nan],
        ...     tp_stop=[np.nan, np.nan, np.nan, 0.1],
        ...     is_entry_open=True
        ... )

        >>> ohlcstx.entries
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                   True   True   True   True
        1                  False  False  False  False
        2                  False  False  False  False
        3                  False  False  False  False
        4                  False  False  False  False
        5                  False  False  False  False

        >>> ohlcstx.exits
        ohlcstx_sl_stop      0.1    NaN    NaN    NaN
        ohlcstx_tsl_th       NaN    NaN    0.2    NaN
        ohlcstx_tsl_stop     NaN    0.1    0.3    NaN
        ohlcstx_tp_stop      NaN    NaN    NaN    0.1
        0                  False  False  False  False
        1                  False  False  False   True
        2                  False  False  False  False
        3                  False   True  False  False
        4                   True  False   True  False
        5                  False  False  False  False

        >>> ohlcstx.stop_price
        ohlcstx_sl_stop    0.1   NaN  NaN   NaN
        ohlcstx_tsl_th     NaN   NaN  0.2   NaN
        ohlcstx_tsl_stop   NaN   0.1  0.3   NaN
        ohlcstx_tp_stop    NaN   NaN  NaN   0.1
        0                  NaN   NaN  NaN   NaN
        1                  NaN   NaN  NaN  11.0
        2                  NaN   NaN  NaN   NaN
        3                  NaN  11.7  NaN   NaN
        4                  9.0   NaN  9.1   NaN
        5                  NaN   NaN  NaN   NaN

        >>> ohlcstx.stop_type_readable
        ohlcstx_sl_stop     0.1   NaN   NaN   NaN
        ohlcstx_tsl_th      NaN   NaN   0.2   NaN
        ohlcstx_tsl_stop    NaN   0.1   0.3   NaN
        ohlcstx_tp_stop     NaN   NaN   NaN   0.1
        0                  None  None  None  None
        1                  None  None  None    TP
        2                  None  None  None  None
        3                  None   TSL  None  None
        4                    SL  None   TTP  None
        5                  None  None  None  None
        ```
    """

    plot = bind_ohlcstx_plot(OHLCSTX, "entries")


setattr(OHLCSTX, "__doc__", _OHLCSTX.__doc__)
setattr(OHLCSTX, "plot", _OHLCSTX.plot)
