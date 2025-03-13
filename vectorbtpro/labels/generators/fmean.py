"""Module with `FMEAN`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMEAN",
]

FMEAN = IndicatorFactory(
    class_name="FMEAN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype", "wait"],
    output_names=["fmean"],
).with_apply_func(
    nb.future_mean_nb,
    kwargs_as_args=["minp", "adjust"],
    param_settings=dict(
        wtype=dict(
            dtype=generic_enums.WType,
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=14,
    wtype="simple",
    wait=1,
    minp=None,
    adjust=False,
)


class _FMEAN(FMEAN):
    """Look-ahead indicator based on `vectorbtpro.labels.nb.future_mean_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmean_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMEAN.fmean` against `FMEAN.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `FMEAN.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMEAN.close`.
            fmean_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMEAN.fmean`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.FMEAN.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMEAN.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMEAN.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmean_trace_kwargs is None:
            fmean_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmean_trace_kwargs = merge_dicts(
            dict(name="Future mean", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmean_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmean.vbt.lineplot(
            trace_kwargs=fmean_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(FMEAN, "__doc__", _FMEAN.__doc__)
setattr(FMEAN, "plot", _FMEAN.plot)
