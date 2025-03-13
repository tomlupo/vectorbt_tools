# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `SIGDET`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "SIGDET",
]

SIGDET = IndicatorFactory(
    class_name="SIGDET",
    module_name=__name__,
    short_name="sigdet",
    input_names=["close"],
    param_names=["lag", "factor", "influence", "up_factor", "down_factor", "mean_influence", "std_influence"],
    output_names=["signal", "upper_band", "lower_band"],
).with_apply_func(
    nb.signal_detection_nb,
    param_settings=dict(
        factor=flex_elem_param_config,
        influence=flex_elem_param_config,
        up_factor=flex_elem_param_config,
        down_factor=flex_elem_param_config,
        mean_influence=flex_elem_param_config,
        std_influence=flex_elem_param_config,
    ),
    lag=14,
    factor=1.0,
    influence=1.0,
    up_factor=None,
    down_factor=None,
    mean_influence=None,
    std_influence=None,
)


class _SIGDET(SIGDET):
    """Robust peak detection algorithm (using z-scores).

    See https://stackoverflow.com/a/22640362"""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        signal_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `SIGDET.signal` against `SIGDET.close`.

        Args:
            column (str): Name of the column to plot.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.signal`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/SIGDET.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SIGDET.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        signal_trace_kwargs = merge_dicts(
            dict(name="Signal", line=dict(color=plotting_cfg["color_schema"]["lightblue"], shape="hv")),
            signal_trace_kwargs,
        )
        fig = self_col.signal.vbt.lineplot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig

    def plot_bands(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        upper_band_trace_kwargs: tp.KwargsLike = None,
        lower_band_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `SIGDET.upper_band` and `SIGDET.lower_band` against `SIGDET.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `SIGDET.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.close`.
            upper_band_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.upper_band`.
            lower_band_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `SIGDET.lower_band`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.SIGDET.run(ohlcv['Close']).plot_bands().show()
            ```

            ![](/assets/images/api/SIGDET_plot_bands.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/SIGDET_plot_bands.dark.svg#only-dark){: .iimg loading=lazy }
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
        if upper_band_trace_kwargs is None:
            upper_band_trace_kwargs = {}
        if lower_band_trace_kwargs is None:
            lower_band_trace_kwargs = {}
        lower_band_trace_kwargs = merge_dicts(
            dict(
                name="Lower band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
            ),
            lower_band_trace_kwargs,
        )
        upper_band_trace_kwargs = merge_dicts(
            dict(
                name="Upper band",
                line=dict(color=adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.5)),
                fill="tonexty",
                fillcolor="rgba(128, 128, 128, 0.2)",
            ),
            upper_band_trace_kwargs,
        )  # default kwargs
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )

        fig = self_col.lower_band.vbt.lineplot(
            trace_kwargs=lower_band_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        fig = self_col.upper_band.vbt.lineplot(
            trace_kwargs=upper_band_trace_kwargs,
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


setattr(SIGDET, "__doc__", _SIGDET.__doc__)
setattr(SIGDET, "plot", _SIGDET.plot)
setattr(SIGDET, "plot_bands", _SIGDET.plot_bands)
