# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `VWAP`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts
from vectorbtpro.utils.template import RepFunc

__all__ = [
    "VWAP",
]


def substitute_anchor(wrapper: ArrayWrapper, anchor: tp.Optional[tp.FrequencyLike]) -> tp.Array1d:
    """Substitute reset frequency by group lens."""
    if anchor is None:
        return np.array([wrapper.shape[0]])
    return wrapper.get_index_grouper(anchor).get_group_lens()


VWAP = IndicatorFactory(
    class_name="VWAP",
    module_name=__name__,
    short_name="vwap",
    input_names=["high", "low", "close", "volume"],
    param_names=["anchor"],
    output_names=["vwap"],
).with_apply_func(
    nb.vwap_nb,
    param_settings=dict(
        anchor=dict(template=RepFunc(substitute_anchor)),
    ),
    anchor="D",
)


class _VWAP(VWAP):
    """Volume-Weighted Average Price (VWAP).

    VWAP is a technical analysis indicator used on intraday charts that resets at the start
    of every new trading session.

    See [Volume-Weighted Average Price (VWAP)](https://www.investopedia.com/terms/v/vwap.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        vwap_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `VWAP.vwap` against `VWAP.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `VWAP.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `VWAP.close`.
            vwap_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `VWAP.vwap`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.VWAP.run(
            ...    ohlcv['High'],
            ...    ohlcv['Low'],
            ...    ohlcv['Close'],
            ...    ohlcv['Volume'],
            ...    anchor="W"
            ... ).plot().show()
            ```

            ![](/assets/images/api/VWAP.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/VWAP.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if vwap_trace_kwargs is None:
            vwap_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        vwap_trace_kwargs = merge_dicts(
            dict(name="VWAP", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            vwap_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.vwap.vbt.lineplot(
            trace_kwargs=vwap_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(VWAP, "__doc__", _VWAP.__doc__)
setattr(VWAP, "plot", _VWAP.plot)
