# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `MA`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MA",
]

MA = IndicatorFactory(
    class_name="MA",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["ma"],
).with_apply_func(
    nb.ma_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="simple",
    minp=None,
    adjust=False,
)


class _MA(MA):
    """Moving Average (MA).

    A moving average is a widely used indicator in technical analysis that helps smooth out
    price action by filtering out the “noise” from random short-term price fluctuations.

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        ma_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `MA.ma` against `MA.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `MA.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.close`.
            ma_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ma`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.MA.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MA.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MA.dark.svg#only-dark){: .iimg loading=lazy }
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
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        ma_trace_kwargs = merge_dicts(
            dict(name="MA", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            ma_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.ma.vbt.lineplot(
            trace_kwargs=ma_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(MA, "__doc__", _MA.__doc__)
setattr(MA, "plot", _MA.plot)
