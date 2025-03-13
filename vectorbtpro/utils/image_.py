# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for images."""

import numpy as np

from vectorbtpro import _typing as tp
from vectorbtpro.utils.pbar import ProgressBar

__all__ = [
    "save_animation",
]


def hstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """Stack NumPy images horizontally."""
    h1, w1, d = a.shape
    h2, w2, _ = b.shape
    c = np.full((max(h1, h2), w1 + w2, d), 255, np.uint8)
    c[:h1, :w1, :] = a
    c[:h2, w1 : w1 + w2, :] = b
    return c


def vstack_image_arrays(a: tp.Array3d, b: tp.Array3d) -> tp.Array3d:
    """Stack NumPy images vertically."""
    h1, w1, d = a.shape
    h2, w2, _ = b.shape
    c = np.full((h1 + h2, max(w1, w2), d), 255, np.uint8)
    c[:h1, :w1, :] = a
    c[h1 : h1 + h2, :w2, :] = b
    return c


def save_animation(
    fname: str,
    index: tp.Sequence,
    plot_func: tp.Callable,
    *args,
    delta: tp.Optional[int] = None,
    step: int = 1,
    fps: int = 3,
    writer_kwargs: dict = None,
    show_progress: bool = True,
    pbar_kwargs: tp.KwargsLike = None,
    to_image_kwargs: tp.KwargsLike = None,
    **kwargs,
) -> None:
    """Save animation to a file.

    Args:
        fname (str): File name.
        index (sequence): Index to iterate over.
        plot_func (callable): Plotting function.

            Must take subset of `index`, `*args`, and `**kwargs`, and return either a Plotly figure,
            image that can be read by `imageio.imread`, or a NumPy array.
        *args: Positional arguments passed to `plot_func`.
        delta (int): Window size of each iteration.
        step (int): Step of each iteration.
        fps (int): Frames per second.

            Will be translated to `duration` by `1000 / fps`.
        writer_kwargs (dict): Keyword arguments passed to `imageio.get_writer`.
        show_progress (bool): Whether to show the progress bar.
        pbar_kwargs (dict): Keyword arguments passed to `vectorbtpro.utils.pbar.ProgressBar`.
        to_image_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.to_image`.
        **kwargs: Keyword arguments passed to `plot_func`.

    Usage:
        ```pycon
        >>> from vectorbtpro import *

        >>> def plot_data_window(index, data):
        ...     return data.loc[index].plot()

        >>> data = vbt.YFData.pull("BTC-USD", start="2020", end="2021")
        >>> vbt.save_animation(
        ...     "plot_data_window.gif",
        ...     data.index,
        ...     plot_data_window,
        ...     data,
        ...     delta=90,
        ...     step=10
        ... )
        ```
    """
    from vectorbtpro.utils.module_ import assert_can_import

    assert_can_import("plotly")
    import plotly.graph_objects as go
    import imageio

    if writer_kwargs is None:
        writer_kwargs = {}
    if "duration" not in writer_kwargs:
        writer_kwargs["duration"] = 1000 / fps
    if pbar_kwargs is None:
        pbar_kwargs = {}
    if "bar_id" not in pbar_kwargs:
        pbar_kwargs["bar_id"] = "save_animation"
    if to_image_kwargs is None:
        to_image_kwargs = {}
    if delta is None:
        delta = len(index) // 2

    with imageio.get_writer(fname, **writer_kwargs) as writer:
        index_steps = range(0, len(index) - delta + 1, step)
        with ProgressBar(index_steps, show_progress=show_progress, **pbar_kwargs) as pbar:
            pbar.set_description("{} → {}".format(str(index[0]), str(index[0 + delta - 1])))

            for i in range(len(index_steps)):
                j = index_steps[i]
                fig = plot_func(index[j : j + delta], *args, **kwargs)
                if fig is None:
                    continue
                if isinstance(fig, (go.Figure, go.FigureWidget)):
                    fig = fig.to_image(format="png", **to_image_kwargs)
                if not isinstance(fig, np.ndarray):
                    fig = imageio.imread(fig)
                writer.append_data(fig)

                if i + 1 < len(index_steps):
                    next_j = index_steps[i + 1]
                    pbar.set_description("{} → {}".format(str(index[next_j]), str(index[next_j + delta - 1])))
                pbar.update()
