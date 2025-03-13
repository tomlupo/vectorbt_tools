"""Module with `FMIN`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMIN",
]

FMIN = IndicatorFactory(
    class_name="FMIN",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wait"],
    output_names=["fmin"],
).with_apply_func(
    nb.future_min_nb,
    window=14,
    wait=1,
)


class _FMIN(FMIN):
    """Look-ahead indicator based on `vectorbtpro.labels.nb.future_min_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmin_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMIN.fmin` against `FMIN.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `FMIN.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMIN.close`.
            fmin_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMIN.fmin`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.FMIN.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMIN.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMIN.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmin_trace_kwargs is None:
            fmin_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmin_trace_kwargs = merge_dicts(
            dict(name="Future min", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmin_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmin.vbt.lineplot(
            trace_kwargs=fmin_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(FMIN, "__doc__", _FMIN.__doc__)
setattr(FMIN, "plot", _FMIN.plot)
