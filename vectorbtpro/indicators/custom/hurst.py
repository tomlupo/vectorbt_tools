# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `HURST`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.enums import HurstMethod
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "HURST",
]

HURST = IndicatorFactory(
    class_name="HURST",
    module_name=__name__,
    input_names=["close"],
    param_names=[
        "window",
        "method",
        "max_lag",
        "min_log",
        "max_log",
        "log_step",
        "min_chunk",
        "max_chunk",
        "num_chunks",
    ],
    output_names=["hurst"],
).with_apply_func(
    nb.rolling_hurst_nb,
    kwargs_as_args=["minp", "stabilize"],
    param_settings=dict(
        method=dict(
            dtype=HurstMethod,
            dtype_kwargs=dict(enum_unkval=None),
            post_index_func=lambda index: index.str.lower(),
        )
    ),
    window=200,
    method="standard",
    max_lag=20,
    min_log=1,
    max_log=2,
    log_step=0.25,
    min_chunk=8,
    max_chunk=100,
    num_chunks=5,
    minp=None,
    stabilize=False,
)


class _HURST(HURST):
    """Moving Hurst exponent (HURST)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        hurst_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `HURST.hurst`.

        Args:
            column (str): Name of the column to plot.
            hurst_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `HURST.hurst`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```pycon
            >>> ohlcv = vbt.YFData.pull(
            ...     "BTC-USD",
            ...     start="2020-01-01",
            ...     end="2024-01-01"
            ... ).get()
            >>> vbt.HURST.run(ohlcv["Close"]).plot().show()
            ```

            ![](/assets/images/api/HURST.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/HURST.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if hurst_trace_kwargs is None:
            hurst_trace_kwargs = {}
        hurst_trace_kwargs = merge_dicts(
            dict(name="HURST", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            hurst_trace_kwargs,
        )

        fig = self_col.hurst.vbt.lineplot(
            trace_kwargs=hurst_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
            **layout_kwargs,
        )

        return fig


setattr(HURST, "__doc__", _HURST.__doc__)
setattr(HURST, "plot", _HURST.plot)
