"""Module with `PIVOTLB`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.configs import flex_elem_param_config
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb

__all__ = [
    "PIVOTLB",
]

PIVOTLB = IndicatorFactory(
    class_name="PIVOTLB",
    module_name=__name__,
    input_names=["high", "low"],
    param_names=["up_th", "down_th"],
    output_names=["labels"],
).with_apply_func(
    nb.pivots_nb,
    param_settings=dict(
        up_th=flex_elem_param_config,
        down_th=flex_elem_param_config,
    ),
)


class _PIVOTLB(PIVOTLB):
    """Label generator based on `vectorbtpro.labels.nb.pivots_nb`."""

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot the median of `PIVOTLB.high` and `PIVOTLB.low`, and overlay it with the heatmap of `PIVOTLB.labels`.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Usage:
            ```pycon
            >>> vbt.PIVOTLB.run(ohlcv['High'], ohlcv['Low'], up_th=0.2, down_th=0.2).plot().show()
            ```

            ![](/assets/images/api/PIVOTLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/PIVOTLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        median = (self_col.high + self_col.low) / 2
        return median.rename("Median").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(PIVOTLB, "__doc__", _PIVOTLB.__doc__)
setattr(PIVOTLB, "plot", _PIVOTLB.plot)
