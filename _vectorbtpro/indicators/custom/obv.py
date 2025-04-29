# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Module with `OBV`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators import nb
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.utils.config import merge_dicts

__all__ = [
    "OBV",
]

OBV = IndicatorFactory(
    class_name="OBV",
    module_name=__name__,
    short_name="obv",
    input_names=["close", "volume"],
    param_names=[],
    output_names=["obv"],
).with_custom_func(nb.obv_nb)


class _OBV(OBV):
    """On-balance volume (OBV).

    It relates price and volume in the stock market. OBV is based on a cumulative total volume.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp)."""

    def plot(
        self,
        column: tp.Optional[tp.Label] = None,
        obv_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot `OBV.obv`.

        Args:
            column (str): Name of the column to plot.
            obv_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OBV.obv`.
            add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments passed to `fig.update_layout`.

        Usage:
            ```py
            >>> vbt.OBV.run(ohlcv['Close'], ohlcv['Volume']).plot().show()
            ```

            ![](/assets/images/api/OBV.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/OBV.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        self_col = self.select_col(column=column)

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_dicts(
            dict(name="OBV", line=dict(color=plotting_cfg["color_schema"]["lightblue"])),
            obv_trace_kwargs,
        )

        fig = self_col.obv.vbt.lineplot(
            trace_kwargs=obv_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )

        return fig


setattr(OBV, "__doc__", _OBV.__doc__)
setattr(OBV, "plot", _OBV.plot)
