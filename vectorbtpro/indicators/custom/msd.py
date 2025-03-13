# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `MSD`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "MSD",
]

MSD = IndicatorFactory(
    class_name="MSD",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype"],
    output_names=["msd"],
).with_apply_func(
    nb.msd_nb,
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
    minp=None,
    adjust=False,
    ddof=0,
)


class _MSD(MSD):
    """Moving Standard Deviation (MSD).

    Standard deviation is an indicator that measures the size of an assets recent price moves
    in order to predict how volatile the price may be in the future."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        msd_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `MSD.msd`.

        Args:
            column (str): Name of the column to plot.
            msd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MSD.msd`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> vbt.MSD.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MSD.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MSD.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if msd_trace_kwargs is None:
            msd_trace_kwargs = {}
        msd_trace_kwargs = merge_dicts(
            dict(name="MSD", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            msd_trace_kwargs,
        )

        fig = self_col.msd.vbt.lineplot(
            trace_kwargs=msd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


setattr(MSD, "__doc__", _MSD.__doc__)
setattr(MSD, "plot", _MSD.plot)
