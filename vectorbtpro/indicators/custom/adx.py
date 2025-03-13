# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `ADX`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "ADX",
]

ADX = IndicatorFactory(
    class_name="ADX",
    module_name=__name__,
    input_names=["high", "low", "close"],
    param_names=["window", "wtype"],
    output_names=["plus_di", "minus_di", "dx", "adx"],
).with_apply_func(
    nb.adx_nb,
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


class _ADX(ADX):
    """Average Directional Movement Index (ADX).

    The indicator is used by some traders to determine the strength of a trend.

    See [Average Directional Index (ADX)](https://www.investopedia.com/terms/a/adx.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plus_di_trace_kwargs: tp.KwargsLike = None,
        minus_di_trace_kwargs: tp.KwargsLike = None,
        adx_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `ADX.plus_di`, `ADX.minus_di`, and `ADX.adx`.

        Args:
            column (str): Name of the column to plot.
            plus_di_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ADX.plus_di`.
            minus_di_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ADX.minus_di`.
            adx_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ADX.adx`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.ADX.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/ADX.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ADX.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if plus_di_trace_kwargs is None:
            plus_di_trace_kwargs = {}
        if minus_di_trace_kwargs is None:
            minus_di_trace_kwargs = {}
        if adx_trace_kwargs is None:
            adx_trace_kwargs = {}
        plus_di_trace_kwargs = merge_dicts(
            dict(name="+DI", line=dict(color=plotting_cfg["color_schema"]["green"], dash="dot")),
            plus_di_trace_kwargs,
        )
        minus_di_trace_kwargs = merge_dicts(
            dict(name="-DI", line=dict(color=plotting_cfg["color_schema"]["red"], dash="dot")),
            minus_di_trace_kwargs,
        )
        adx_trace_kwargs = merge_dicts(
            dict(name="ADX", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            adx_trace_kwargs,
        )

        fig = self_col.plus_di.vbt.lineplot(
            trace_kwargs=plus_di_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.minus_di.vbt.lineplot(
            trace_kwargs=minus_di_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.adx.vbt.lineplot(
            trace_kwargs=adx_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(ADX, "__doc__", _ADX.__doc__)
setattr(ADX, "plot", _ADX.plot)
