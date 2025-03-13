"""Module with `MEANLB`."""

from vectorbtpro import _typing as tp
from vectorbtpro.generic import enums as generic_enums
from vectorbtpro.indicators.factory import IndicatorFactory
from vectorbtpro.labels import nb

__all__ = [
    "MEANLB",
]

MEANLB = IndicatorFactory(
    class_name="MEANLB",
    module_name=__name__,
    input_names=["close"],
    param_names=["window", "wtype", "wait"],
    output_names=["labels"],
).with_apply_func(
    nb.mean_labels_nb,
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


class _MEANLB(MEANLB):
    """Label generator based on `vectorbtpro.labels.nb.mean_labels_nb`."""

    def plot(self, column: tp.Optional[tp.Label] = None, **kwargs) -> tp.BaseFigure:
        """Plot `close` and overlay it with the heatmap of `MEANLB.labels`.

        `**kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.overlay_with_heatmap`.

        Usage:
            ```pycon
            >>> vbt.MEANLB.run(ohlcv['Close']).plot().show()
            ```

            ![](/assets/images/api/MEANLB.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/MEANLB.dark.svg#only-dark){: .iimg loading=lazy }
        """
        self_col = self.select_col(column=column, group_by=False)
        return self_col.close.rename("Close").vbt.overlay_with_heatmap(self_col.labels.rename("Labels"), **kwargs)


setattr(MEANLB, "__doc__", _MEANLB.__doc__)
setattr(MEANLB, "plot", _MEANLB.plot)
