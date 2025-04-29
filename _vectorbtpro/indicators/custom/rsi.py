# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `RSI`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "RSI",
]

RSI = IndicatorFactory(
    class_name="RSI",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["rsi"],
).with_apply_func(
    nb.rsi_nb,
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


class _RSI(RSI):
    """Relative Strength Index (RSI).

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Tuple[float, float] = (30, 70),
        rsi_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `RSI.rsi`.

        Args:
            column (str): Name of the column to plot.
            limits (tuple of float): Tuple of the lower and upper limit.
            rsi_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `RSI.rsi`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.RSI.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/RSI.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/RSI.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_dicts(
            dict(name="RSI", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            rsi_trace_kwargs,
        )

        fig = self_col.rsi.vbt.lineplot(
            trace_kwargs=rsi_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        default_layout = dict()
        default_layout[yaxis.replace("y", "yaxis")] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        # Fill void between limits
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xaxis,
                yref=yaxis,
                x0=self_col.wrapper.index[0],
                y0=limits[0],
                x1=self_col.wrapper.index[-1],
                y1=limits[1],
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(RSI, "__doc__", _RSI.__doc__)
setattr(RSI, "plot", _RSI.plot)
