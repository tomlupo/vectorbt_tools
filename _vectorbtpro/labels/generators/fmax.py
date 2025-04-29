"""Module with `FMAX`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FMAX",
]

FMAX = IndicatorFactory(
    class_name="FMAX",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wait"],
    output_names=["fmax"],
).with_apply_func(
    nb.future_max_nb,
    window=14,
    wait=1,
)


class _FMAX(FMAX):
    """Look-ahead indicator based on `vectorbtpro.labels.nb.future_max_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        plot_close: bool = True,
        close_trace_kwargs: tp.KwargsLike = None,
        fmax_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FMAX.fmax` against `FMAX.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `FMAX.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMAX.close`.
            fmax_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FMAX.fmax`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.FMAX.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FMAX.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FMAX.dark.svg#only-dark){: .iimg loading=lazy }
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
        if fmax_trace_kwargs is None:
            fmax_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(
            dict(name="Close", line=dict(color=plotting_cfg["color_schema"]["blue"])),
            close_trace_kwargs,
        )
        fmax_trace_kwargs = merge_dicts(
            dict(name="Future max", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fmax_trace_kwargs,
        )

        if plot_close:
            fig = self_col.close.vbt.lineplot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs,
                fig=fig,
            )
        fig = self_col.fmax.vbt.lineplot(
            trace_kwargs=fmax_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(FMAX, "__doc__", _FMAX.__doc__)
setattr(FMAX, "plot", _FMAX.plot)
