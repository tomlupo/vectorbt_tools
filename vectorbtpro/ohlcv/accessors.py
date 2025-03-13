# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Custom Pandas accessors for OHLC(V) data.

Methods can be accessed as follows:

* `OHLCVDFAccessor` -> `pd.DataFrame.vbt.ohlcv.*`

The accessors inherit `vectorbtpro.generic.accessors`.

!!! note
    Accessors do not utilize caching.

## Column names

By default, vectorbt searches for columns with names 'open', 'high', 'low', 'close', and 'volume'
(case doesn't matter). You can change the naming either using `feature_map` in
`vectorbtpro._settings.ohlcv`, or by providing `feature_map` directly to the accessor.

```pycon
>>> from vectorbtpro import *

>>> df = pd.DataFrame({
...     'my_open1': [2, 3, 4, 3.5, 2.5],
...     'my_high2': [3, 4, 4.5, 4, 3],
...     'my_low3': [1.5, 2.5, 3.5, 2.5, 1.5],
...     'my_close4': [2.5, 3.5, 4, 3, 2],
...     'my_volume5': [10, 11, 10, 9, 10]
... })
>>> df.vbt.ohlcv.get_feature('open')
None

>>> my_feature_map = {
...     "my_open1": "Open",
...     "my_high2": "High",
...     "my_low3": "Low",
...     "my_close4": "Close",
...     "my_volume5": "Volume",
... }
>>> ohlcv_acc = df.vbt.ohlcv(freq='d', feature_map=my_feature_map)
>>> ohlcv_acc.get_feature('open')
0    2.0
1    3.0
2    4.0
3    3.5
4    2.5
Name: my_open1, dtype: float64
```

## Stats

!!! hint
    See `vectorbtpro.generic.stats_builder.StatsBuilderMixin.stats` and `OHLCVDFAccessor.metrics`.

```pycon
>>> ohlcv_acc.stats()
Start                           0
End                             4
Period            5 days 00:00:00
First Price                   2.0
Lowest Price                  1.5
Highest Price                 4.5
Last Price                    2.0
First Volume                   10
Lowest Volume                   9
Highest Volume                 11
Last Volume                    10
Name: agg_stats, dtype: object
```

## Plots

!!! hint
    See `vectorbtpro.generic.plots_builder.PlotsBuilderMixin.plots` and `OHLCVDFAccessor.subplots`.

`OHLCVDFAccessor` class has a single subplot based on `OHLCVDFAccessor.plot` (without volume):

```pycon
>>> ohlcv_acc.plots(settings=dict(ohlc_type='candlestick')).show()
```

![](/assets/images/api/ohlcv_plots.light.svg#only-light){: .iimg loading=lazy }
![](/assets/images/api/ohlcv_plots.dark.svg#only-dark){: .iimg loading=lazy }
"""

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.accessors import register_df_vbt_accessor
from vectorbtpro.base.wrapping import ArrayWrapper
from vectorbtpro.data.base import OHLCDataMixin
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor, GenericDFAccessor
from vectorbtpro.utils.config import merge_dicts, Config, HybridConfig
from vectorbtpro.utils.decorators import class_or_instanceproperty

if tp.TYPE_CHECKING:
    from vectorbtpro.data.base import Data as DataT
else:
    DataT = tp.Any

__all__ = [
    "OHLCVDFAccessor",
]

__pdoc__ = {}

OHLCVDFAccessorT = tp.TypeVar("OHLCVDFAccessorT", bound="OHLCVDFAccessor")


@register_df_vbt_accessor("ohlcv")
class OHLCVDFAccessor(OHLCDataMixin, GenericDFAccessor):
    """Accessor on top of OHLCV data. For DataFrames only.

    Accessible via `pd.DataFrame.vbt.ohlcv`."""

    _expected_keys: tp.ExpectedKeys = (GenericDFAccessor._expected_keys or set()) | {
        "feature_map",
    }

    def __init__(
        self,
        wrapper: tp.Union[ArrayWrapper, tp.ArrayLike],
        obj: tp.Optional[tp.ArrayLike] = None,
        feature_map: tp.KwargsLike = None,
        **kwargs,
    ) -> None:
        GenericDFAccessor.__init__(self, wrapper, obj=obj, feature_map=feature_map, **kwargs)

        self._feature_map = feature_map

    @class_or_instanceproperty
    def df_accessor_cls(cls_or_self) -> tp.Type["OHLCVDFAccessor"]:
        """Accessor class for `pd.DataFrame`."""
        return OHLCVDFAccessor

    @property
    def feature_map(self) -> tp.Kwargs:
        """Column names."""
        from vectorbtpro._settings import settings

        ohlcv_cfg = settings["ohlcv"]

        return merge_dicts(ohlcv_cfg["feature_map"], self._feature_map)

    @property
    def feature_wrapper(self) -> ArrayWrapper:
        new_columns = self.wrapper.columns.map(lambda x: self.feature_map[x] if x in self.feature_map else x)
        return self.wrapper.replace(columns=new_columns)

    @property
    def symbol_wrapper(self) -> ArrayWrapper:
        return ArrayWrapper(self.wrapper.index, [None], 1)

    def select_feature_idxs(self: OHLCVDFAccessorT, idxs: tp.MaybeSequence[int], **kwargs) -> OHLCVDFAccessorT:
        return self.iloc[:, idxs]

    def select_symbol_idxs(self: OHLCVDFAccessorT, idxs: tp.MaybeSequence[int], **kwargs) -> OHLCVDFAccessorT:
        raise NotImplementedError

    def get(
        self,
        features: tp.Union[None, tp.MaybeFeatures] = None,
        symbols: tp.Union[None, tp.MaybeSymbols] = None,
        feature: tp.Optional[tp.Feature] = None,
        symbol: tp.Optional[tp.Symbol] = None,
        **kwargs,
    ) -> tp.SeriesFrame:
        if features is not None and feature is not None:
            raise ValueError("Must provide either features or feature, not both")
        if symbols is not None or symbol is not None:
            raise ValueError("Cannot provide symbols")
        if feature is not None:
            features = feature
            single_feature = True
        else:
            if features is None:
                return self.obj
            single_feature = not self.has_multiple_keys(features)
        if not single_feature:
            feature_idxs = [self.get_feature_idx(k, raise_error=True) for k in features]
        else:
            feature_idxs = self.get_feature_idx(features, raise_error=True)
        return self.obj.iloc[:, feature_idxs]

    # ############# Conversion ############# #

    def to_data(self, data_cls: tp.Optional[tp.Type[DataT]] = None, **kwargs) -> DataT:
        """Convert to a `vectorbtpro.data.base.Data` instance."""
        if data_cls is None:
            from vectorbtpro.data.base import Data

            data_cls = Data

        return data_cls.from_data(self.obj, columns_are_symbols=False, **kwargs)

    # ############# Resampling ############# #

    def resample(self: OHLCVDFAccessorT, *args, wrapper_meta: tp.DictLike = None, **kwargs) -> OHLCVDFAccessorT:
        """Perform resampling on `OHLCVDFAccessor`."""
        if wrapper_meta is None:
            wrapper_meta = self.wrapper.resample_meta(*args, **kwargs)
        sr_dct = {}
        for feature in self.feature_wrapper.columns:
            if isinstance(feature, str) and feature.lower() == "open":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.first_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "high":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.max_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "low":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.min_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "close":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.last_reduce_nb,
                )
            elif isinstance(feature, str) and feature.lower() == "volume":
                sr_dct[feature] = self.obj[feature].vbt.resample_apply(
                    wrapper_meta["resampler"],
                    generic_nb.sum_reduce_nb,
                )
            else:
                raise ValueError(f"Cannot match feature '{feature}'")
        new_obj = pd.DataFrame(sr_dct)
        return self.replace(
            wrapper=wrapper_meta["new_wrapper"],
            obj=new_obj,
        )

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `OHLCVDFAccessor.stats`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.stats_defaults` and
        `stats` from `vectorbtpro._settings.ohlcv`."""
        from vectorbtpro._settings import settings

        ohlcv_stats_cfg = settings["ohlcv"]["stats"]

        return merge_dicts(GenericAccessor.stats_defaults.__get__(self), ohlcv_stats_cfg)

    _metrics: tp.ClassVar[Config] = HybridConfig(
        dict(
            start_index=dict(
                title="Start Index",
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags="wrapper",
            ),
            end_index=dict(
                title="End Index",
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags="wrapper",
            ),
            total_duration=dict(
                title="Total Duration",
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags="wrapper",
            ),
            first_price=dict(
                title="First Price",
                calc_func=lambda ohlc: generic_nb.bfill_1d_nb(ohlc.values.flatten())[0],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            lowest_price=dict(
                title="Lowest Price",
                calc_func=lambda ohlc: ohlc.values.min(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            highest_price=dict(
                title="Highest Price",
                calc_func=lambda ohlc: ohlc.values.max(),
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            last_price=dict(
                title="Last Price",
                calc_func=lambda ohlc: generic_nb.ffill_1d_nb(ohlc.values.flatten())[-1],
                resolve_ohlc=True,
                tags=["ohlcv", "ohlc"],
            ),
            first_volume=dict(
                title="First Volume",
                calc_func=lambda volume: generic_nb.bfill_1d_nb(volume.values)[0],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            lowest_volume=dict(
                title="Lowest Volume",
                calc_func=lambda volume: volume.values.min(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            highest_volume=dict(
                title="Highest Volume",
                calc_func=lambda volume: volume.values.max(),
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
            last_volume=dict(
                title="Last Volume",
                calc_func=lambda volume: generic_nb.ffill_1d_nb(volume.values)[-1],
                resolve_volume=True,
                tags=["ohlcv", "volume"],
            ),
        )
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

    # ############# Plotting ############# #

    def plot_ohlc(
        self,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot OHLC data.

        Args:
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]
        ohlcv_cfg = settings["ohlcv"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # Set up figure
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        if ohlc_type is None:
            ohlc_type = ohlcv_cfg["ohlc_type"]
        if isinstance(ohlc_type, str):
            if ohlc_type.lower() == "ohlc":
                plot_obj = go.Ohlc
            elif ohlc_type.lower() == "candlestick":
                plot_obj = go.Candlestick
            else:
                raise ValueError("Plot type can be either 'OHLC' or 'Candlestick'")
        else:
            plot_obj = ohlc_type
        def_trace_kwargs = dict(
            x=self.wrapper.index,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            name="OHLC",
            increasing=dict(
                fillcolor=plotting_cfg["color_schema"]["increasing"],
                line=dict(color=plotting_cfg["color_schema"]["increasing"]),
            ),
            decreasing=dict(
                fillcolor=plotting_cfg["color_schema"]["decreasing"],
                line=dict(color=plotting_cfg["color_schema"]["decreasing"]),
            ),
            opacity=0.75,
        )
        if plot_obj is go.Ohlc:
            del def_trace_kwargs["increasing"]["fillcolor"]
            del def_trace_kwargs["decreasing"]["fillcolor"]
        _trace_kwargs = merge_dicts(def_trace_kwargs, trace_kwargs)
        ohlc = plot_obj(**_trace_kwargs)
        fig.add_trace(ohlc, **add_trace_kwargs)
        xaxis = getattr(fig.data[-1], "xaxis", None)
        if xaxis is None:
            xaxis = "x"
        if "rangeslider_visible" not in layout_kwargs.get(xaxis.replace("x", "xaxis"), {}):
            fig.update_layout({xaxis.replace("x", "xaxis"): dict(rangeslider_visible=False)})
        return fig

    def plot_volume(
        self,
        trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot volume data.

        Args:
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        import plotly.graph_objects as go
        from vectorbtpro.utils.figure import make_figure
        from vectorbtpro._settings import settings

        plotting_cfg = settings["plotting"]

        if trace_kwargs is None:
            trace_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        # Set up figure
        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        marker_colors = np.empty(self.volume.shape, dtype=object)
        mask_greater = (self.close.values - self.open.values) > 0
        mask_less = (self.close.values - self.open.values) < 0
        marker_colors[mask_greater] = plotting_cfg["color_schema"]["increasing"]
        marker_colors[mask_less] = plotting_cfg["color_schema"]["decreasing"]
        marker_colors[~(mask_greater | mask_less)] = plotting_cfg["color_schema"]["gray"]
        _trace_kwargs = merge_dicts(
            dict(
                x=self.wrapper.index,
                y=self.volume,
                marker=dict(color=marker_colors, line_width=0),
                opacity=0.5,
                name="Volume",
            ),
            trace_kwargs,
        )
        volume_bar = go.Bar(**_trace_kwargs)
        fig.add_trace(volume_bar, **add_trace_kwargs)
        return fig

    def plot(
        self,
        ohlc_type: tp.Union[None, str, tp.BaseTraceType] = None,
        plot_volume: tp.Optional[bool] = None,
        ohlc_trace_kwargs: tp.KwargsLike = None,
        volume_trace_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        volume_add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **layout_kwargs,
    ) -> tp.BaseFigure:
        """Plot OHLC(V) data.

        Args:
            ohlc_type: Either 'OHLC', 'Candlestick' or Plotly trace.

                Pass None to use the default.
            plot_volume (bool): Whether to plot volume beneath.
            ohlc_trace_kwargs (dict): Keyword arguments passed to `ohlc_type`.
            volume_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace` for OHLC.
            volume_add_trace_kwargs (dict): Keyword arguments passed to `add_trace` for volume.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        Usage:
            ```pycon
            >>> vbt.YFData.pull("BTC-USD").get().vbt.ohlcv.plot().show()
            ```

            [=100% "100%"]{: .candystripe .candystripe-animate }

            ![](/assets/images/api/ohlcv_plot.light.svg#only-light){: .iimg loading=lazy }
            ![](/assets/images/api/ohlcv_plot.dark.svg#only-dark){: .iimg loading=lazy }
        """
        from vectorbtpro.utils.module_ import assert_can_import

        assert_can_import("plotly")
        from vectorbtpro.utils.figure import make_figure, make_subplots

        if plot_volume is None:
            plot_volume = self.volume is not None
        if plot_volume:
            add_trace_kwargs = merge_dicts(dict(row=1, col=1), add_trace_kwargs)
            volume_add_trace_kwargs = merge_dicts(dict(row=2, col=1), volume_add_trace_kwargs)

        # Set up figure
        if fig is None:
            if plot_volume:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0,
                    row_heights=[0.7, 0.3],
                )
            else:
                fig = make_figure()
            fig.update_layout(
                showlegend=True,
                xaxis=dict(showgrid=True),
                yaxis=dict(showgrid=True),
            )
            if plot_volume:
                fig.update_layout(
                    xaxis2=dict(showgrid=True),
                    yaxis2=dict(showgrid=True),
                )
        fig.update_layout(**layout_kwargs)

        fig = self.plot_ohlc(
            ohlc_type=ohlc_type,
            trace_kwargs=ohlc_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs,
            fig=fig,
        )
        if plot_volume:
            fig = self.plot_volume(
                trace_kwargs=volume_trace_kwargs,
                add_trace_kwargs=volume_add_trace_kwargs,
                fig=fig,
            )
        return fig

    @property
    def plots_defaults(self) -> tp.Kwargs:
        """Defaults for `OHLCVDFAccessor.plots`.

        Merges `vectorbtpro.generic.accessors.GenericAccessor.plots_defaults` and
        `plots` from `vectorbtpro._settings.ohlcv`."""
        from vectorbtpro._settings import settings

        ohlcv_plots_cfg = settings["ohlcv"]["plots"]

        return merge_dicts(GenericAccessor.plots_defaults.__get__(self), ohlcv_plots_cfg)

    _subplots: tp.ClassVar[Config] = HybridConfig(
        dict(
            plot=dict(
                title="OHLC",
                xaxis_kwargs=dict(showgrid=True, rangeslider_visible=False),
                yaxis_kwargs=dict(showgrid=True),
                check_is_not_grouped=True,
                plot_func="plot",
                plot_volume=False,
                tags="ohlcv",
            )
        )
    )

    @property
    def subplots(self) -> Config:
        return self._subplots


OHLCVDFAccessor.override_metrics_doc(__pdoc__)
OHLCVDFAccessor.override_subplots_doc(__pdoc__)
