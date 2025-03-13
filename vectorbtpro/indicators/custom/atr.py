# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `ATR`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ATR",
]

ATR = IndicatorFactory(
    class_name="ATR",
    module_name=__name__,
    input_names=["high", "low", "close"],
    param_names=["window", "wtype"],
    output_names=["tr", "atr"],
).with_apply_func(
    nb.atr_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="wilder",
    minp=None,
    adjust=False,
)


class _ATR(ATR):
    """Average True Range (ATR).

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        tr_trace_kwargs: tp.KwargsLike = None,
        atr_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            column (str): Name of the column to plot.
            tr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.atr`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.ATR.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/ATR.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ATR.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if tr_trace_kwargs is None:
            tr_trace_kwargs = {}
        if atr_trace_kwargs is None:
            atr_trace_kwargs = {}
        tr_trace_kwargs = merge_dicts(
            dict(name="TR", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            tr_trace_kwargs,
        )
        atr_trace_kwargs = merge_dicts(
            dict(name="ATR", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            atr_trace_kwargs,
        )

        fig = self_col.tr.vbt.lineplot(
            trace_kwargs=tr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.atr.vbt.lineplot(
            trace_kwargs=atr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(ATR, "__doc__", _ATR.__doc__)
setattr(ATR, "plot", _ATR.plot)
