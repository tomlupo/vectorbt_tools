# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `OLS`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "OLS",
]

OLS = IndicatorFactory(
    class_name="OLS",
    module_name=__name__,
    short_name="ols",
    input_names=["x", "y"],
    param_names=["window"],
    output_names=["slope", "intercept", "zscore"],
    lazy_outputs=dict(
        pred=lambda self: self.wrapper.wrap(
            nb.ols_pred_nb(
                to_2d_array(self.x),
                to_2d_array(self.slope),
                to_2d_array(self.intercept),
            ),
        ),
        error=lambda self: self.wrapper.wrap(
            nb.ols_error_nb(
                to_2d_array(self.y),
                to_2d_array(self.pred),
            ),
        ),
        angle=lambda self: self.wrapper.wrap(
            nb.ols_angle_nb(
                to_2d_array(self.slope),
            ),
        ),
    ),
).with_apply_func(
    nb.ols_nb,
    kwargs_as_args=["minp", "ddof", "with_zscore"],
    window=14,
    minp=None,
    ddof=0,
    with_zscore=True,
)


class _OLS(OLS):
    """Rolling Ordinary Least Squares (OLS).

    The indicator can be used to detect changes in the behavior of the stocks against the market or each other.

    See [The Linear Regression of Time and Price](https://www.investopedia.com/articles/trading/09/linear-regression-time-price.asp).
    """

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_y: bool = True,
        y_trace_kwargs: tp.KwargsLike = None,
        pred_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `OLS.pred` against `OLS.y`.

        Args:
            column (str): Name of the column to plot.
            plot_y (bool): Whether to plot `OLS.y`.
            y_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.y`.
            pred_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.pred`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/OLS.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OLS.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if y_trace_kwargs is None:
            y_trace_kwargs = {}
        if pred_trace_kwargs is None:
            pred_trace_kwargs = {}
        y_trace_kwargs = merge_dicts(
            dict(name="Y", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            y_trace_kwargs,
        )
        pred_trace_kwargs = merge_dicts(
            dict(name="Pred", line=dict(color=plotting_cfg["color_schema"]["lightpurple"])),
            pred_trace_kwargs,
        )

        if plot_y:
            fig = self_col.y.vbt.lineplot(
                trace_kwargs=y_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.pred.vbt.lineplot(
            trace_kwargs=pred_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig

    def plot_zscore(
        self,
        column: tp.Optional[tp.Label] = None,
        alpha: float = 0.05,
        zscore_trace_kwargs: tp.KwargsLike = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `OLS.zscore` with confidence intervals.

        Args:
            column (str): Name of the column to plot.
            alpha (float): The alpha level for the confidence interval.

                The default alpha = .05 returns a 95% confidence interval.
            zscore_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OLS.zscore`.
            add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape`
                when adding the range between both confidence intervals.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.OLS.run(np.arange(len(ohlcv)), ohlcv['Close']).plot_zscore().show()
            ```

            ![](/assets/images/api/OLS_zscore.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OLS_zscore.dark.svg#only-dark){: .iimg loading=lazy }
        """
        import scipy.stats as st
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        zscore_trace_kwargs = merge_dicts(
            dict(name="Z-score", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            zscore_trace_kwargs,
        )
        fig = self_col.zscore.vbt.lineplot(
            trace_kwargs=zscore_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        # Fill void between limits
        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        add_shape_kwargs = merge_dicts(
            dict(
                type="rect",
                xref=xaxis,
                yref=yaxis,
                x0=self_col.wrapper.index[0],
                y0=st.norm.ppf(1 - alpha / 2),
                x1=self_col.wrapper.index[-1],
                y1=st.norm.ppf(alpha / 2),
                fillcolor="mediumslateblue",
                opacity=0.2,
                layer="below",
                line_width=0,
            ),
            add_shape_kwargs,
        )
        fig.add_shape(**add_shape_kwargs)

        return fig


setattr(OLS, "__doc__", _OLS.__doc__)
setattr(OLS, "plot", _OLS.plot)
setattr(OLS, "plot_zscore", _OLS.plot_zscore)
