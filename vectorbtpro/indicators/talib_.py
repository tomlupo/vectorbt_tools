# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Helper functions for TA-Lib."""

import inspect
import warnings

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.base.merging import column_stack_arrays
from vectorbtpro.base.reshaping import to_pd_array, broadcast_arrays, broadcast
from vectorbtpro.base.wrapping import ArrayWrapper, Wrapping
from vectorbtpro.generic import nb as generic_nb
from vectorbtpro.generic.accessors import GenericAccessor
from vectorbtpro.utils.array_ import build_nan_mask, squeeze_nan, unsqueeze_nan
from vectorbtpro.utils.colors import adjust_opacity
from vectorbtpro.utils.config import merge_dicts, resolve_dict

__all__ = [
    "talib_func",
    "talib_plot_func",
]


def talib_func(func_name: str) -> tp.Callable:
    """Get the TA-Lib indicator function."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("talib")
    import talib
    from talib import abstract

    func_name = func_name.upper()
    talib_func = getattr(talib, func_name)
    info = abstract.Function(func_name).info
    input_names = []
    for in_names in info["input_names"].values():
        if isinstance(in_names, (list, tuple)):
            input_names.extend(list(in_names))
        else:
            input_names.append(in_names)
    output_names = info["output_names"]
    one_output = len(output_names) == 1
    param_names = list(info["parameters"].keys())

    def run_talib_func(
        *args,
        timeframe: tp.Optional[tp.FrequencyLike] = None,
        resample_map: tp.KwargsLike = None,
        resample_kwargs: tp.KwargsLikeSequence = None,
        realign_kwargs: tp.KwargsLikeSequence = None,
        wrapper: tp.Optional[ArrayWrapper] = None,
        skipna: bool = False,
        silence_warnings: bool = False,
        broadcast_kwargs: tp.KwargsLike = None,
        wrap_kwargs: tp.KwargsLike = None,
        wrap: tp.Optional[bool] = None,
        unpack_to: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Union[tp.MaybeTuple[tp.AnyArray], tp.Dict[str, tp.AnyArray]]:
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrap_kwargs is None:
            wrap_kwargs = {}

        inputs = []
        other_args = []
        for k in range(len(args)):
            if k < len(input_names) and len(inputs) < len(input_names):
                inputs.append(args[k])
            else:
                other_args.append(args[k])
        if len(inputs) < len(input_names):
            for k in input_names:
                if k in kwargs:
                    inputs.append(kwargs.pop(k))

        is_pandas = False
        common_type = None
        common_shape = None
        broadcasting_needed = False
        new_inputs = []
        for input in inputs:
            if isinstance(input, (pd.Series, pd.DataFrame)):
                is_pandas = True
            elif not isinstance(input, np.ndarray):
                input = np.asarray(input)
            if common_type is None:
                common_type = type(input)
            elif type(input) != common_type:
                broadcasting_needed = True
            if common_shape is None:
                common_shape = input.shape
            elif input.shape != common_shape:
                broadcasting_needed = True
            new_inputs.append(input)
        inputs = new_inputs
        if broadcasting_needed:
            if is_pandas:
                if wrapper is None:
                    inputs, wrapper = broadcast(
                        dict(zip(input_names, inputs)),
                        return_wrapper=True,
                        **broadcast_kwargs,
                    )
                else:
                    inputs = broadcast(dict(zip(input_names, inputs)), **broadcast_kwargs)
                inputs = [inputs[k].values for k in input_names]
            else:
                inputs = broadcast_arrays(*inputs)
        else:
            if is_pandas:
                if wrapper is None:
                    wrapper = ArrayWrapper.from_obj(inputs[0])
                inputs = [input.values for input in inputs]
        input_shape = inputs[0].shape

        def _run_talib_func(inputs, *_args, **_kwargs):
            target_index = None
            if timeframe is not None:
                if wrapper is None:
                    raise ValueError("Resampling requires a wrapper")
                if wrapper.freq is None:
                    if not silence_warnings:
                        warnings.warn(
                            (
                                "Couldn't parse the frequency of index. "
                                "Set freq in wrapper_kwargs via broadcast_kwargs, or globally."
                            ),
                            stacklevel=2,
                        )
                new_inputs = ()
                _resample_map = merge_dicts(
                    resample_map,
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    },
                )
                source_wrapper = ArrayWrapper(index=wrapper.index, freq=wrapper.freq)
                for i, input in enumerate(inputs):
                    _resample_kwargs = resolve_dict(resample_kwargs, i=i)
                    new_input = GenericAccessor(source_wrapper, input).resample_apply(
                        timeframe,
                        _resample_map[input_names[i]],
                        **_resample_kwargs,
                    )
                    target_index = new_input.index
                    new_inputs += (new_input.values,)
                inputs = new_inputs

            def _build_nan_outputs():
                nan_outputs = []
                for i in range(len(output_names)):
                    nan_outputs.append(np.full(input_shape, np.nan, dtype=np.double))
                if len(nan_outputs) == 1:
                    return nan_outputs[0]
                return nan_outputs

            all_nan = False
            if skipna:
                nan_mask = build_nan_mask(*inputs)
                if nan_mask.all():
                    all_nan = True
                else:
                    inputs = squeeze_nan(*inputs, nan_mask=nan_mask)
            else:
                nan_mask = None
            if all_nan:
                outputs = _build_nan_outputs()
            else:
                inputs = tuple([arr.astype(np.double) for arr in inputs])
                try:
                    outputs = talib_func(*inputs, *_args, **_kwargs)
                except Exception as e:
                    if "inputs are all NaN" in str(e):
                        outputs = _build_nan_outputs()
                        all_nan = True
                    else:
                        raise e
                if not all_nan:
                    if one_output:
                        outputs = unsqueeze_nan(outputs, nan_mask=nan_mask)
                    else:
                        outputs = unsqueeze_nan(*outputs, nan_mask=nan_mask)
                    if timeframe is not None:
                        new_outputs = ()
                        target_wrapper = ArrayWrapper(index=target_index)
                        for i, output in enumerate(outputs):
                            _realign_kwargs = merge_dicts(
                                dict(
                                    source_rbound=True,
                                    target_rbound=True,
                                    nan_value=np.nan,
                                    ffill=True,
                                    silence_warnings=True,
                                ),
                                resolve_dict(realign_kwargs, i=i),
                            )
                            new_output = GenericAccessor(target_wrapper, output).realign(
                                wrapper.index,
                                freq=wrapper.freq,
                                **_realign_kwargs,
                            )
                            new_outputs += (new_output.values,)
                        outputs = new_outputs
            return outputs

        if inputs[0].ndim == 1:
            outputs = _run_talib_func(inputs, *other_args, **kwargs)
        else:
            outputs = []
            for col in range(inputs[0].shape[1]):
                col_inputs = [input[:, col] for input in inputs]
                col_outputs = _run_talib_func(col_inputs, *other_args, **kwargs)
                outputs.append(col_outputs)
            outputs = list(zip(*outputs))
            outputs = tuple(map(column_stack_arrays, outputs))
        if wrap is None:
            wrap = is_pandas
        if wrap:
            outputs = [wrapper.wrap(output, **wrap_kwargs) for output in outputs]
        if unpack_to is not None:
            if unpack_to.lower() in ("dict", "frame"):
                dct = {name: outputs[i] for i, name in enumerate(output_names)}
                if unpack_to.lower() == "dict":
                    return dct
                return pd.concat(list(dct.values()), axis=1, keys=pd.Index(list(dct.keys()), name="output"))
            raise ValueError(f"Invalid option unpack_to='{unpack_to}'")
        if one_output:
            return outputs[0]
        return outputs

    signature = inspect.signature(run_talib_func)
    new_parameters = list(signature.parameters.values())[1:]
    k = 0
    for input_name in input_names:
        new_parameters.insert(
            k,
            inspect.Parameter(
                input_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=tp.ArrayLike,
            ),
        )
        k += 1
    for param_name in param_names:
        new_parameters.insert(
            k,
            inspect.Parameter(
                param_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=info["parameters"][param_name],
                annotation=tp.Scalar,
            ),
        )
        k += 1
    run_talib_func.__signature__ = signature.replace(parameters=new_parameters)
    run_talib_func.__name__ = "run_" + func_name.lower()
    run_talib_func.__qualname__ = run_talib_func.__name__
    run_talib_func.__doc__ = f"""Run `talib.{func_name}` on NumPy arrays, Series, and DataFrames.
    
Requires [TA-Lib](https://github.com/mrjbq7/ta-lib) installed.

Set `timeframe` to a frequency to resample the input arrays to this frequency, run the function,
and then resample the output arrays back to the original frequency. Optionally, provide `resample_map`
as a dictionary that maps input names to resample-apply function names. Keyword arguments 
`resample_kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.resample_apply`
while `realign_kwargs` are passed to `vectorbtpro.generic.accessors.GenericAccessor.realign`.
Both can be also provided as sequences of dictionaries - one dictionary per input and output respectively.

Set `skipna` to True to run the TA-Lib function on non-NA values only.

Broadcasts the input arrays if they have different types or shapes.

If one of the input arrays is a Series/DataFrame, wraps the output arrays into a Pandas format.
To enable or disable wrapping, set `wrap` to True and False respectively."""

    return run_talib_func


def talib_plot_func(func_name: str) -> tp.Callable:
    """Get the TA-Lib indicator plotting function."""
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("talib")
    from talib import abstract
    from vectorbtpro._settings import settings

    plotting_cfg = settings["plotting"]

    func_name = func_name.upper()
    info = abstract.Function(func_name).info
    output_names = info["output_names"]
    output_flags = info["output_flags"]

    def run_talib_plot_func(
        *outputs,
        wrapper: tp.Optional[ArrayWrapper] = None,
        wrap_kwargs: tp.KwargsLike = None,
        column: tp.Optional[tp.Label] = None,
        limits: tp.Optional[tp.Tuple[float, float]] = None,
        add_shape_kwargs: tp.KwargsLike = None,
        add_trace_kwargs: tp.KwargsLike = None,
        fig: tp.Optional[tp.BaseFigure] = None,
        **kwargs,
    ) -> tp.BaseFigure:
        if wrap_kwargs is None:
            wrap_kwargs = {}

        new_outputs = []
        for output in outputs:
            if not isinstance(output, (pd.Series, pd.DataFrame)):
                if wrapper is not None:
                    output = wrapper.wrap(output, **wrap_kwargs)
                else:
                    output = to_pd_array(output)
            if wrapper is not None:
                output = Wrapping.select_col_from_obj(
                    output,
                    column=column,
                    wrapper=wrapper,
                )
            else:
                output = Wrapping.select_col_from_obj(
                    output,
                    column=column,
                    wrapper=ArrayWrapper.from_obj(output),
                )
            new_outputs.append(output)
        outputs = dict(zip(output_names, new_outputs))

        output_trace_kwargs = {}
        for output_name in output_names:
            output_trace_kwargs[output_name] = kwargs.pop(output_name + "_trace_kwargs", {})
        priority_outputs = []
        other_outputs = []
        for output_name in output_names:
            flags = set(output_flags.get(output_name))
            found_priority = False
            if abstract.TA_OUTPUT_FLAGS[2048] in flags:
                priority_outputs = priority_outputs + [output_name]
                found_priority = True
            if abstract.TA_OUTPUT_FLAGS[4096] in flags:
                priority_outputs = [output_name] + priority_outputs
                found_priority = True
            if not found_priority:
                other_outputs.append(output_name)

        for output_name in priority_outputs + other_outputs:
            output = outputs[output_name].rename(output_name)
            flags = set(output_flags.get(output_name))
            trace_kwargs = {}
            plot_func_name = "lineplot"

            if abstract.TA_OUTPUT_FLAGS[2] in flags:
                # Dotted Line
                if "line" not in trace_kwargs:
                    trace_kwargs["line"] = dict()
                trace_kwargs["line"]["dash"] = "dashdot"
            if abstract.TA_OUTPUT_FLAGS[4] in flags:
                # Dashed Line
                if "line" not in trace_kwargs:
                    trace_kwargs["line"] = dict()
                trace_kwargs["line"]["dash"] = "dash"
            if abstract.TA_OUTPUT_FLAGS[8] in flags:
                # Dot
                if "line" not in trace_kwargs:
                    trace_kwargs["line"] = dict()
                trace_kwargs["line"]["dash"] = "dot"
            if abstract.TA_OUTPUT_FLAGS[16] in flags:
                # Histogram
                hist = np.asarray(output)
                hist_diff = generic_nb.diff_1d_nb(hist)
                marker_colors = np.full(hist.shape, adjust_opacity("silver", 0.75), dtype=object)
                marker_colors[(hist > 0) & (hist_diff > 0)] = adjust_opacity("green", 0.75)
                marker_colors[(hist > 0) & (hist_diff <= 0)] = adjust_opacity("lightgreen", 0.75)
                marker_colors[(hist < 0) & (hist_diff < 0)] = adjust_opacity("red", 0.75)
                marker_colors[(hist < 0) & (hist_diff >= 0)] = adjust_opacity("lightcoral", 0.75)
                if "marker" not in trace_kwargs:
                    trace_kwargs["marker"] = {}
                trace_kwargs["marker"]["color"] = marker_colors
                if "line" not in trace_kwargs["marker"]:
                    trace_kwargs["marker"]["line"] = {}
                trace_kwargs["marker"]["line"]["width"] = 0
                kwargs["bargap"] = 0
                plot_func_name = "barplot"
            if abstract.TA_OUTPUT_FLAGS[2048] in flags:
                # Values represent an upper limit
                if "line" not in trace_kwargs:
                    trace_kwargs["line"] = {}
                trace_kwargs["line"]["color"] = adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.75)
                trace_kwargs["fill"] = "tonexty"
                trace_kwargs["fillcolor"] = "rgba(128, 128, 128, 0.2)"
            if abstract.TA_OUTPUT_FLAGS[4096] in flags:
                # Values represent a lower limit
                if "line" not in trace_kwargs:
                    trace_kwargs["line"] = {}
                trace_kwargs["line"]["color"] = adjust_opacity(plotting_cfg["color_schema"]["gray"], 0.75)

            trace_kwargs = merge_dicts(trace_kwargs, output_trace_kwargs[output_name])
            plot_func = getattr(output.vbt, plot_func_name)
            fig = plot_func(trace_kwargs=trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig, **kwargs)

        if limits is not None:
            xaxis = getattr(fig.data[-1], "xaxis", None)
            if xaxis is None:
                xaxis = "x"
            yaxis = getattr(fig.data[-1], "yaxis", None)
            if yaxis is None:
                yaxis = "y"
            add_shape_kwargs = merge_dicts(
                dict(
                    type="rect",
                    xref=xaxis,
                    yref=yaxis,
                    x0=outputs[output_names[0]].index[0],
                    y0=limits[0],
                    x1=outputs[output_names[0]].index[-1],
                    y1=limits[1],
                    fillcolor="mediumslateblue",
                    opacity=0.2,
                    layer="below",
                    line_width=0,
                ),
                add_shape_kwargs,
            )
            fig.add_shape(**add_shape_kwargs)

        return fig

    signature = inspect.signature(run_talib_plot_func)
    new_parameters = list(signature.parameters.values())[1:-1]
    k = 0
    for output_name in output_names:
        new_parameters.insert(
            k,
            inspect.Parameter(
                output_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=tp.ArrayLike,
            ),
        )
        k += 1
    for output_name in output_names:
        new_parameters.insert(
            -3,
            inspect.Parameter(
                output_name + "_trace_kwargs",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=tp.KwargsLike,
            ),
        )
    new_parameters.append(inspect.Parameter("layout_kwargs", inspect.Parameter.VAR_KEYWORD))
    run_talib_plot_func.__signature__ = signature.replace(parameters=new_parameters)
    output_trace_kwargs_docstring = "\n    ".join(
        [
            f"{output_name}_trace_kwargs (dict): Keyword arguments passed to the trace of `{output_name}`."
            for output_name in output_names
        ]
    )
    run_talib_plot_func.__name__ = "plot_" + func_name.lower()
    run_talib_plot_func.__qualname__ = run_talib_plot_func.__name__
    run_talib_plot_func.__doc__ = f"""Plot output arrays of `talib.{func_name}`.

Args:
    column (str): Name of the column to plot.
    limits (tuple of float): Tuple of the lower and upper limit.
    {output_trace_kwargs_docstring}
    add_shape_kwargs (dict): Keyword arguments passed to `fig.add_shape` when adding the range between both limits.
    add_trace_kwargs (dict): Keyword arguments passed to `fig.add_trace` when adding each trace.
    fig (Figure or FigureWidget): Figure to add the traces to.
    **layout_kwargs: Keyword arguments passed to `fig.update_layout`."""

    return run_talib_plot_func
