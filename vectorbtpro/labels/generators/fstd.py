"""Module with `FSTD`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "FSTD",
]

FSTD = IndicatorFactory(
    class_name="FSTD",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype", "wait"],
    output_names=["fstd"],
).with_apply_func(
    nb.future_std_nb,
    kwargs_as_args=["minp", "adjust", "ddof"],
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
    ddof=0,
)


class _FSTD(FSTD):
    """Look-ahead indicator based on `vectorbtpro.labels.nb.future_std_nb`."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        fstd_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `FSTD.fstd`.

        Args:
            column (str): Name of the column to plot.
            fstd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `FSTD.fstd`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.FSTD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FSTD.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FSTD.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        if fstd_trace_kwargs is None:
            fstd_trace_kwargs = {}
        fstd_trace_kwargs = merge_dicts(
            dict(name="Future STD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            fstd_trace_kwargs,
        )
        fig = self_col.fstd.vbt.lineplot(
            trace_kwargs=fstd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        return fig


setattr(FSTD, "__doc__", _FSTD.__doc__)
setattr(FSTD, "plot", _FSTD.plot)
