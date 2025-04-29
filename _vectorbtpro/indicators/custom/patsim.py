# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `PATSIM`."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.generic import nb as generic_nb, enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "PATSIM",
]

PATSIM = IndicatorFactory(
    class_name="PATSIM",
    module_name=__name__,
    short_name="patsim",
    input_names=["close"],
    param_names=[
        "pattern",
        "window",
        "max_window",
        "row_select_prob",
        "window_select_prob",
        "interp_mode",
        "rescale_mode",
        "vmin",
        "vmax",
        "pmin",
        "pmax",
        "invert",
        "error_type",
        "distance_measure",
        "max_error",
        "max_error_interp_mode",
        "max_error_as_maxdist",
        "max_error_strict",
        "min_pct_change",
        "max_pct_change",
        "min_similarity",
    ],
    output_names=["similarity"],
).with_apply_func(
    generic_nb.rolling_pattern_similarity_nb,
    param_settings=dict(
        pattern=dict(is_array_like=True, min_one_dim=True),
        interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
        ),
        rescale_mode=dict(
            dtype=generic_enums.RescaleMode,
            post_index_func=lambda index: index.str.lower(),
        ),
        error_type=dict(
            dtype=generic_enums.ErrorType,
            post_index_func=lambda index: index.str.lower(),
        ),
        distance_measure=dict(
            dtype=generic_enums.DistanceMeasure,
            post_index_func=lambda index: index.str.lower(),
        ),
        max_error=dict(is_array_like=True, min_one_dim=True),
        max_error_interp_mode=dict(
            dtype=generic_enums.InterpMode,
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    window=None,
    max_window=None,
    row_select_prob=1.0,
    window_select_prob=1.0,
    interp_mode="mixed",
    rescale_mode="minmax",
    vmin=np.nan,
    vmax=np.nan,
    pmin=np.nan,
    pmax=np.nan,
    invert=False,
    error_type="absolute",
    distance_measure="mae",
    max_error=np.nan,
    max_error_interp_mode=None,
    max_error_as_maxdist=False,
    max_error_strict=False,
    min_pct_change=np.nan,
    max_pct_change=np.nan,
    min_similarity=np.nan,
)


class _PATSIM(PATSIM):
    """Rolling pattern similarity.

    Based on `vectorbtpro.generic.nb.rolling.rolling_pattern_similarity_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        similarity_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `PATSIM.similarity` against `PATSIM.close`.

        Args:
            column (str): Name of the column to plot.
            similarity_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PATSIM.similarity`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).plot().show()
            ```

            ![](/assets/images/api/PATSIM.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PATSIM.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        similarity_trace_kwargs = merge_dicts(
            dict(name="Similarity", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            similarity_trace_kwargs,
        )
        fig = self_col.similarity.vbt.lineplot(
            trace_kwargs=similarity_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        yaxis = getattr(fig.data[-1], "yaxis", None)
        if yaxis is None:
            yaxis = "y"
        default_layout = dict()
        default_layout[yaxis.replace("y", "yaxis")] = dict(tickformat=",.0%")
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        return fig

    def overlay_with_heatmap(
        self,
        column: tp.Optional[tp.Label] = None,
        close_trace_kwargs: tp.KwargsLike = None,
        similarity_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Overlay `PATSIM.similarity` as a heatmap on top of `PATSIM.close`.

        Args:
            column (str): Name of the column to plot.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `PATSIM.close`.
            similarity_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Heatmap` for `PATSIM.similarity`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.PATSIM.run(ohlcv['Close'], np.array([1, 2, 3, 2, 1]), 30).overlay_with_heatmap().show()
            ```

            ![](/assets/images/api/PATSIM_heatmap.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PATSIM_heatmap.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if similarity_trace_kwargs is None:
            similarity_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        similarity_trace_kwargs = merge_dicts(
            dict(
                colorbar=dict(tickformat=",.0%"),
                colorscale=[
                    [0.0, "rgba(0, 0, 0, 0)"],
                    [1.0, plotting_cfg["color_schema"]["lightpurple"]],
                ],
                zmin=0,
                zmax=1,
            ),
            similarity_trace_kwargs,
        )
        fig = self_col.close.vbt.overlay_with_heatmap(
            self_col.similarity,
            trace_kwargs=close_trace_kwargs,
            heatmap_kwargs=dict(y_labels=["Similarity"], trace_kwargs=similarity_trace_kwargs),
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


setattr(PATSIM, "__doc__", _PATSIM.__doc__)
setattr(PATSIM, "plot", _PATSIM.plot)
setattr(PATSIM, "overlay_with_heatmap", _PATSIM.overlay_with_heatmap)
