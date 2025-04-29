"""Module with `TRENDLB`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb
from vectorbtpro.labels.enums import TrendLabelMode

__all__ = [
    "TRENDLB",
]

TRENDLB = IndicatorFactory(
    class_name="TRENDLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["up_th", "down_th", "mode"],
    output_names=["labels"],
).with_apply_func(
    nb.trend_labels_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
        mode=dict(
            dtype=TrendLabelMode,
            post_index_func=lambda index: index.str.lower(),
        ),
    ),
    mode=TrendLabelMode.Binary,
)


class _TRENDLB(TRENDLB):
    """Label generator based on `vectorbtpro.labels.nb.trend_labels_nb`."""

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `TRENDLB.high` and `TRENDLB.low`, and overlay it with the heatmap of `TRENDLB.labels`.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Usage:
            ```pycon
            >>> vbt.TRENDLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/TRENDLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/TRENDLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(TRENDLB, "__doc__", _TRENDLB.__doc__)
setattr(TRENDLB, "plot", _TRENDLB.plot)
