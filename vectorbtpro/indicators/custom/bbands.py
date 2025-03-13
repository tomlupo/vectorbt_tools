# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `BBANDS`."""

from vectorbtpro import _typing as tp
from vectorbtpro.base.reshaping import to_2d_array
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "BBANDS",
]

BBANDS = IndicatorFactory(
    class_name="BBANDS",
    module_name=__name__,
    short_name="bb",
    input_names=["close"],
    param_names=["window", "wtype", "alpha"],
    output_names=["upper", "middle", "lower"],
    lazy_outputs=dict(
        percent_b=lambda self: self.wrapper.wrap(
            nb.bbands_percent_b_nb(
                to_2d_array(self.close),
                to_2d_array(self.upper),
                to_2d_array(self.lower),
            ),
        ),
        bandwidth=lambda self: self.wrapper.wrap(
            nb.bbands_bandwidth_nb(
                to_2d_array(self.upper),
                to_2d_array(self.middle),
                to_2d_array(self.lower),
            ),
        ),
    ),
).with_apply_func(
    nb.bbands_nb,
    kwargs_as_args=["minp", "adjust", "ddof"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="simple",
    alpha=2,
    minp=None,
    adjust=False,
    ddof=0,
)


class _BBANDS(BBANDS):
    """Bollinger Bands (BBANDS).

    A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        upper_trace_kwargs: tp.KwargsLike = None,
        middle_trace_kwargs: tp.KwargsLike = None,
        lower_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `BBANDS.upper`, `BBANDS.middle`, and `BBANDS.lower` against `BBANDS.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `BBANDS.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.close`.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.upper`.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.middle`.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.lower`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.BBANDS.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/BBANDS.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/BBANDS.dark.svg#only-dark){: .iimg loading=lazy }
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
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        lower_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_trace_kwargs,
        )
        upper_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_trace_kwargs,
        )  # default kwargs
        middle_trace_kwargs = merge_dicts(
            dict(name="Middle band", line=dict(color=plotting_cfg["color_schema"]["lightblue"])), middle_trace_kwargs
        )
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower.vbt.lineplot(
            trace_kwargs=lower_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper.vbt.lineplot(
            trace_kwargs=upper_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.middle.vbt.lineplot(
            trace_kwargs=middle_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )

        return fig


setattr(BBANDS, "__doc__", _BBANDS.__doc__)
setattr(BBANDS, "plot", _BBANDS.plot)
