"""Module with `FIXLB`."""

from vectorbtpro import _typing as tp
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb

__all__ = [
    "FIXLB",
]

FIXLB = IndicatorFactory(
    class_name="FIXLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["n"],
    output_names=["labels"],
).with_apply_func(
    nb.fixed_labels_nb,
    n=1,
)


class _FIXLB(FIXLB):
    """Label generator based on `vectorbtpro.labels.nb.fixed_labels_nb`."""

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot `FIXLB.close` and overlay it with the heatmap of `FIXLB.labels`.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Usage:
            ```pycon
            >>> vbt.FIXLB.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/FIXLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/FIXLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        return self_col.close.rename("Close").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(FIXLB, "__doc__", _FIXLB.__doc__)
setattr(FIXLB, "plot", _FIXLB.plot)
