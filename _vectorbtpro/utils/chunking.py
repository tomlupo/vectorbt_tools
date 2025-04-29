# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for chunking."""

import inspect
import multiprocessing
import uuid
import warnings
from functools import wraps

import numpy as np
import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.annotations import get_annotations, flatten_annotations, Annotatable, Union
from vectorbtpro.utils.attr_ import DefineMixin, define, MISSING
from vectorbtpro.utils.config import merge_dicts, FrozenConfig, Configured
from vectorbtpro.utils.eval_ import Evaluable
from vectorbtpro.utils.execution import Task, execute
from vectorbtpro.utils.merging import MergeFunc, parse_merge_func
from vectorbtpro.utils.parsing import annotate_args, ann_args_to_args, match_ann_arg, get_func_arg_names, Regex
from vectorbtpro.utils.template import substitute_templates, Rep

__all__ = [
    "ChunkMeta",
    "ArgChunkMeta",
    "LenChunkMeta",
    "yield_chunk_meta",
    "Sizer",
    "ArgSizer",
    "CountSizer",
    "LenSizer",
    "ShapeSizer",
    "ArraySizer",
    "ChunkMapper",
    "NotChunked",
    "ChunkTaker",
    "ChunkSelector",
    "ChunkSlicer",
    "CountAdapter",
    "ShapeSelector",
    "ShapeSlicer",
    "ArraySelector",
    "ArraySlicer",
    "ContainerTaker",
    "SequenceTaker",
    "MappingTaker",
    "ArgsTaker",
    "KwargsTaker",
    "Chunkable",
    "Chunked",
    "ChunkedCount",
    "ChunkedShape",
    "ChunkedArray",
    "Chunker",
    "chunked",
]

__pdoc__ = {}


# ############# Universal ############# #


@define
class ArgGetter(DefineMixin):
    """Class for getting an argument from annotated arguments."""

    arg_query: tp.Optional[tp.AnnArgQuery] = define.field(default=None)
    """Query for annotated argument to derive the size from."""

    def get_arg(self, ann_args: tp.AnnArgs) -> tp.Any:
        """Get argument using `vectorbtpro.utils.parsing.match_ann_arg`."""
        if self.arg_query is None:
            raise ValueError("Please provide arg_query")
        return match_ann_arg(ann_args, self.arg_query)


@define
class AxisSpecifier(DefineMixin):
    """Class with an attribute for specifying an axis."""

    axis: tp.Optional[int] = define.field(default=None)
    """Axis of the argument to take from."""


@define
class DimRetainer(DefineMixin):
    """Class with an attribute for retaining dimensions."""

    keep_dims: bool = define.field(default=False)
    """Whether to retain dimensions."""


# ############# Chunk sizing ############# #


class Sizer(Evaluable, Annotatable):
    """Abstract class for getting the size from annotated arguments.

    !!! note
        Use `Sizer.apply` instead of `Sizer.get_size`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        """Get the size given the annotated arguments."""
        raise NotImplementedError

    def apply(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        """Apply the sizer."""
        return self.get_size(ann_args, **kwargs)


@define
class ArgSizer(Sizer, ArgGetter, DefineMixin):
    """Class for getting the size from an argument."""

    single_type: tp.Optional[tp.TypeLike] = define.field(default=None)
    """One or multiple types to consider as a single value."""

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_arg(ann_args)

    def apply(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        arg = self.get_arg(ann_args)
        if self.single_type is not None:
            if checks.is_instance_of(arg, self.single_type):
                return 1
        return self.get_size(ann_args, **kwargs)


class CountSizer(ArgSizer):
    """Class for getting the size from a count."""

    @classmethod
    def get_obj_size(cls, obj: int, single_type: tp.Optional[type] = None) -> int:
        """Get size of an object."""
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        return obj

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), single_type=self.single_type)


class LenSizer(ArgSizer):
    """Class for getting the size from the length of an argument."""

    @classmethod
    def get_obj_size(cls, obj: tp.Sequence, single_type: tp.Optional[type] = None) -> int:
        """Get size of an object."""
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        return len(obj)

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), single_type=self.single_type)


@define
class ShapeSizer(ArgSizer, AxisSpecifier, DefineMixin):
    """Class for getting the size from the length of an axis in a shape."""

    @classmethod
    def get_obj_size(cls, obj: tp.ShapeLike, axis: int, single_type: tp.Optional[type] = None) -> int:
        """Get size of an object."""
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        if checks.is_int(obj):
            obj = (obj,)
        if len(obj) == 0:
            return 0
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis <= len(obj) - 1:
            return obj[axis]
        return 0

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), self.axis, single_type=self.single_type)


class ArraySizer(ShapeSizer):
    """Class for getting the size from the length of an axis in an array."""

    @classmethod
    def get_obj_size(cls, obj: tp.AnyArray, axis: int, single_type: tp.Optional[type] = None) -> int:
        """Get size of an object."""
        if single_type is not None:
            if checks.is_instance_of(obj, single_type):
                return 1
        if len(obj.shape) == 0:
            return 0
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis <= len(obj.shape) - 1:
            return obj.shape[axis]
        return 0

    def get_size(self, ann_args: tp.AnnArgs, **kwargs) -> int:
        return self.get_obj_size(self.get_arg(ann_args), self.axis, single_type=self.single_type)


# ############# Chunk generation ############# #


@define
class ChunkMeta(DefineMixin):
    """Class that represents a chunk metadata."""

    uuid: str = define.field()
    """Unique identifier of the chunk.

    Used for caching."""

    idx: int = define.field()
    """Chunk index."""

    start: tp.Optional[int] = define.field()
    """Start of the chunk range (including). Can be None."""

    end: tp.Optional[int] = define.field()
    """End of the chunk range (excluding). Can be None."""

    indices: tp.Optional[tp.Sequence[int]] = define.field()
    """Indices included in the chunk range. Can be None.

    Has priority over `ChunkMeta.start` and `ChunkMeta.end`."""


class ChunkMetaGenerator:
    """Abstract class for generating chunk metadata from annotated arguments."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        """Get chunk metadata."""
        raise NotImplementedError


class ArgChunkMeta(ChunkMetaGenerator, ArgGetter):
    """Class for generating chunk metadata from an argument."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        return self.get_arg(ann_args)


class LenChunkMeta(ArgChunkMeta):
    """Class for generating chunk metadata from a sequence of chunk lengths."""

    def get_chunk_meta(self, ann_args: tp.AnnArgs, **kwargs) -> tp.Iterable[ChunkMeta]:
        arg = self.get_arg(ann_args)
        start = 0
        end = 0
        for i, chunk_len in enumerate(arg):
            end += chunk_len
            yield ChunkMeta(uuid=str(uuid.uuid4()), idx=i, start=start, end=end, indices=None)
            start = end


def yield_chunk_meta(
    size: tp.Optional[int] = None,
    min_size: tp.Optional[int] = None,
    n_chunks: tp.Union[None, int, str] = None,
    chunk_len: tp.Union[None, int, str] = None,
) -> tp.Generator[ChunkMeta, None, None]:
    """Yield meta of each successive chunk from a sequence with a number of elements.

    Args:
        size (int): Size of the space to split.
        min_size (int): Minimum size.

            If `size` is lower than this number, returns a single chunk.
        n_chunks (int or str): Number of chunks.

            If "auto", becomes the number of cores.
        chunk_len (int or str): Length of each chunk.

            If "auto", becomes the number of cores.

    If `size`, `n_chunks`, and `chunk_len` are None (after resolving them from settings),
    returns a single chunk. If only `n_chunks` and `chunk_len` are None, sets `n_chunks` to "auto"."""
    if size is not None and min_size is not None and size < min_size:
        yield ChunkMeta(uuid=str(uuid.uuid4()), idx=0, start=0, end=size, indices=None)
    else:
        if n_chunks is None and chunk_len is None and size is None:
            n_chunks = 1
        if n_chunks is None and chunk_len is None:
            n_chunks = "auto"
        if n_chunks is not None and chunk_len is not None:
            raise ValueError("Must provide either n_chunks or chunk_len, not both")
        if n_chunks is not None:
            if isinstance(n_chunks, str):
                if n_chunks.lower() == "auto":
                    n_chunks = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid option n_chunks='{n_chunks}'")
            if n_chunks == 0:
                raise ValueError("Chunk count cannot be zero")
            if size is not None:
                if n_chunks > size:
                    n_chunks = size
                d, r = divmod(size, n_chunks)
                for i in range(n_chunks):
                    si = (d + 1) * (i if i < r else r) + d * (0 if i < r else i - r)
                    yield ChunkMeta(
                        uuid=str(uuid.uuid4()),
                        idx=i,
                        start=si,
                        end=si + (d + 1 if i < r else d),
                        indices=None,
                    )
            else:
                for i in range(n_chunks):
                    yield ChunkMeta(uuid=str(uuid.uuid4()), idx=i, start=None, end=None, indices=None)
        if chunk_len is not None:
            checks.assert_not_none(size, arg_name="size")
            if isinstance(chunk_len, str):
                if chunk_len.lower() == "auto":
                    chunk_len = multiprocessing.cpu_count()
                else:
                    raise ValueError(f"Invalid option chunk_len='{chunk_len}'")
            if chunk_len == 0:
                raise ValueError("Chunk length cannot be zero")
            for chunk_i, i in enumerate(range(0, size, chunk_len)):
                yield ChunkMeta(
                    uuid=str(uuid.uuid4()),
                    idx=chunk_i,
                    start=i,
                    end=min(i + chunk_len, size),
                    indices=None,
                )


# ############# Chunk mapping ############# #


@define
class ChunkMapper(DefineMixin):
    """Abstract class for mapping chunk metadata.

    Implements the abstract `ChunkMapper.map` method.

    Supports caching of each pair of incoming and outgoing `ChunkMeta` instances.

    !!! note
        Use `ChunkMapper.apply` instead of `ChunkMapper.map`."""

    should_cache: bool = define.field(default=True)
    """Whether should cache."""

    chunk_meta_cache: tp.Dict[str, ChunkMeta] = define.field(factory=dict)
    """Cache for outgoing `ChunkMeta` instances keyed by UUID of the incoming ones."""

    def apply(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Apply the mapper."""
        if not self.should_cache:
            return self.map(chunk_meta, **kwargs)
        if chunk_meta.uuid not in self.chunk_meta_cache:
            new_chunk_meta = self.map(chunk_meta, **kwargs)
            self.chunk_meta_cache[chunk_meta.uuid] = new_chunk_meta
            return new_chunk_meta
        return self.chunk_meta_cache[chunk_meta.uuid]

    def map(self, chunk_meta: ChunkMeta, **kwargs) -> ChunkMeta:
        """Abstract method for mapping chunk metadata.

        Takes the chunk metadata of type `ChunkMeta` and returns a new chunk metadata of the same type."""
        raise NotImplementedError


# ############# Chunk taking ############# #


@define
class NotChunked(Evaluable, Annotatable, DefineMixin):
    """Class that represents an argument that shouldn't be chunked."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""


@define
class ChunkTaker(Evaluable, Annotatable, DefineMixin):
    """Abstract class for taking one or more elements based on the chunk index or range.

    !!! note
        Use `ChunkTaker.apply` instead of `ChunkTaker.take`."""

    single_type: tp.Optional[tp.TypeLike] = define.field(default=None)
    """One or multiple types to consider as a single value."""

    ignore_none: bool = define.field(default=True)
    """Whether to ignore None."""

    mapper: tp.Optional[ChunkMapper] = define.field(default=None)
    """Chunk mapper of type `ChunkMapper`."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def get_size(self, obj: tp.Any, **kwargs) -> int:
        """Get the actual size of the argument."""
        raise NotImplementedError

    def suggest_size(self, obj: tp.Any, **kwargs) -> tp.Optional[int]:
        """Suggest a global size based on the argument's size."""
        if self.mapper is not None:
            return None
        return self.get_size(obj, **kwargs)

    def should_take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> bool:
        """Check whether to take a chunk or leave the argument as it is."""
        if self.ignore_none and obj is None:
            return False
        if self.single_type is not None:
            if checks.is_instance_of(obj, self.single_type):
                return False
        return True

    def apply(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Apply the taker."""
        if self.mapper is not None:
            chunk_meta = self.mapper.apply(chunk_meta, **kwargs)
        if not self.should_take(obj, chunk_meta, **kwargs):
            return obj
        return self.take(obj, chunk_meta, **kwargs)

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        """Abstract method for taking subset of data.

        Takes the argument object, the chunk meta (tuple out of the index, start index,
        and end index of the chunk), and other keyword arguments passed down the stack,
        such as `chunker` and `silence_warnings`."""
        raise NotImplementedError


@define
class ChunkSelector(ChunkTaker, DimRetainer, DefineMixin):
    """Class for selecting one element based on the chunk index."""

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        return LenSizer.get_obj_size(obj, single_type=self.single_type)

    def suggest_size(self, obj: tp.Sequence, **kwargs) -> tp.Optional[int]:
        return None

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        if self.keep_dims:
            return obj[chunk_meta.idx : chunk_meta.idx + 1]
        return obj[chunk_meta.idx]


class ChunkSlicer(ChunkTaker):
    """Class for slicing multiple elements based on the chunk range."""

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        return LenSizer.get_obj_size(obj, single_type=self.single_type)

    def take(self, obj: tp.Sequence, chunk_meta: ChunkMeta, **kwargs) -> tp.Sequence:
        if chunk_meta.indices is not None:
            return obj[chunk_meta.indices]
        return obj[chunk_meta.start : chunk_meta.end]


class CountAdapter(ChunkSlicer):
    """Class for adapting a count based on the chunk range."""

    def get_size(self, obj: int, **kwargs) -> int:
        return CountSizer.get_obj_size(obj, single_type=self.single_type)

    def take(self, obj: int, chunk_meta: ChunkMeta, **kwargs) -> int:
        checks.assert_instance_of(obj, int)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj):
                raise IndexError(f"Positional indexers are out-of-bounds")
            return len(indices)
        if chunk_meta.start >= obj:
            return 0
        return min(obj, chunk_meta.end) - chunk_meta.start


@define
class ShapeSelector(ChunkSelector, AxisSpecifier, DefineMixin):
    """Class for selecting one element from a shape's axis based on the chunk index."""

    def get_size(self, obj: tp.ShapeLike, **kwargs) -> int:
        return ShapeSizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.ShapeLike, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        if checks.is_int(obj):
            obj = (obj,)
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        if chunk_meta.idx >= obj[axis]:
            raise IndexError(f"Index {chunk_meta.idx} is out of bounds for axis {axis} with size {obj[axis]}")
        obj = list(obj)
        if self.keep_dims:
            obj[axis] = 1
        else:
            del obj[axis]
        return tuple(obj)


@define
class ShapeSlicer(ChunkSlicer, AxisSpecifier, DefineMixin):
    """Class for slicing multiple elements from a shape's axis based on the chunk range."""

    def get_size(self, obj: tp.ShapeLike, **kwargs) -> int:
        return ShapeSizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.ShapeLike, chunk_meta: ChunkMeta, **kwargs) -> tp.Shape:
        if checks.is_int(obj):
            obj = (obj,)
        checks.assert_instance_of(obj, tuple)
        if len(obj) == 0:
            return ()
        axis = self.axis
        if axis is None:
            if len(obj) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj):
            raise IndexError(f"Shape is {len(obj)}-dimensional, but {axis} were indexed")
        obj = list(obj)
        if chunk_meta.indices is not None:
            indices = np.asarray(chunk_meta.indices)
            if np.any(indices >= obj[axis]):
                raise IndexError(f"Positional indexers are out-of-bounds")
            obj[axis] = len(indices)
        else:
            if chunk_meta.start >= obj[axis]:
                del obj[axis]
            else:
                obj[axis] = min(obj[axis], chunk_meta.end) - chunk_meta.start
        return tuple(obj)


class ArraySelector(ShapeSelector):
    """Class for selecting one element from an array's axis based on the chunk index."""

    def get_size(self, obj: tp.AnyArray, **kwargs) -> int:
        return ArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.ArrayLike:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj.shape):
            raise IndexError(f"Array is {len(obj.shape)}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if self.keep_dims:
            slc[axis] = slice(chunk_meta.idx, chunk_meta.idx + 1)
        else:
            slc[axis] = chunk_meta.idx
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


class ArraySlicer(ShapeSlicer):
    """Class for slicing multiple elements from an array's axis based on the chunk range."""

    def get_size(self, obj: tp.AnyArray, **kwargs) -> int:
        return ArraySizer.get_obj_size(obj, self.axis, single_type=self.single_type)

    def take(self, obj: tp.AnyArray, chunk_meta: ChunkMeta, **kwargs) -> tp.AnyArray:
        checks.assert_instance_of(obj, (pd.Series, pd.DataFrame, np.ndarray))
        if len(obj.shape) == 0:
            return obj
        axis = self.axis
        if axis is None:
            if len(obj.shape) == 1:
                axis = 0
        checks.assert_not_none(axis, arg_name="axis")
        if axis >= len(obj.shape):
            raise IndexError(f"Array is {len(obj.shape)}-dimensional, but {axis} were indexed")
        slc = [slice(None)] * len(obj.shape)
        if chunk_meta.indices is not None:
            slc[axis] = np.asarray(chunk_meta.indices)
        else:
            slc[axis] = slice(chunk_meta.start, chunk_meta.end)
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.iloc[tuple(slc)]
        return obj[tuple(slc)]


@define
class ContainerTaker(ChunkTaker, DefineMixin):
    """Class for taking from a container with other chunk takers.

    Accepts the specification of the container."""

    cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = define.field(default=None)
    """Specification of the container."""

    def __init__(
        self,
        cont_take_spec: tp.Optional[tp.ContainerTakeSpec] = None,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
    ) -> None:
        ChunkTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=cont_take_spec,
        )

    def get_size(self, obj: tp.Sequence, **kwargs) -> int:
        raise NotImplementedError

    def check_cont_take_spec(self) -> None:
        """Check that `ContainerTaker.cont_take_spec` is not None."""
        if self.cont_take_spec is None:
            raise ValueError("Please provide cont_take_spec")

    def take(self, obj: tp.Any, chunk_meta: ChunkMeta, **kwargs) -> tp.Any:
        raise NotImplementedError


class SequenceTaker(ContainerTaker):
    """Class for taking from a sequence container.

    Calls `Chunker.take_from_arg` on each element."""

    def adapt_cont_take_spec(self, obj: tp.Sequence) -> tp.ContainerTakeSpec:
        """Prepare the specification of the container to the object."""
        cont_take_spec = list(self.cont_take_spec)
        if len(cont_take_spec) >= 2:
            if isinstance(cont_take_spec[-1], type(...)):
                if len(obj) >= len(cont_take_spec):
                    cont_take_spec = cont_take_spec[:-1]
                    cont_take_spec.extend([cont_take_spec[-1]] * (len(obj) - len(cont_take_spec)))
        return cont_take_spec

    def suggest_size(self, obj: tp.Sequence, chunker: tp.Optional["Chunker"] = None, **kwargs) -> tp.Optional[int]:
        if self.mapper is not None:
            return None
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        size_i = None
        size = None
        for i, v in enumerate(obj):
            if i < len(cont_take_spec) and cont_take_spec[i] is not MISSING:
                take_spec = chunker.resolve_take_spec(cont_take_spec[i])
                if isinstance(take_spec, ChunkTaker):
                    try:
                        new_size = take_spec.suggest_size(v)
                        if new_size is not None:
                            if size is None:
                                size_i = i
                                size = new_size
                            elif size != new_size:
                                warnings.warn(
                                    (
                                        f"Arguments at indices {size_i} and {i} have conflicting sizes "
                                        f"{size} and {new_size}. Setting size to None."
                                    ),
                                    stacklevel=2,
                                )
                                return None
                    except NotImplementedError as e:
                        pass
        return size

    def take(
        self,
        obj: tp.Sequence,
        chunk_meta: ChunkMeta,
        chunker: tp.Optional["Chunker"] = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Sequence:
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        new_obj = []
        for i, v in enumerate(obj):
            if i < len(cont_take_spec) and cont_take_spec[i] is not MISSING:
                take_spec = cont_take_spec[i]
            else:
                if not silence_warnings:
                    warnings.warn(
                        (
                            f"Argument at index {i} not found in SequenceTaker.cont_take_spec. "
                            "Setting its specification to None."
                        ),
                        stacklevel=2,
                    )
                take_spec = None
            new_obj.append(
                chunker.take_from_arg(
                    v,
                    take_spec,
                    chunk_meta,
                    chunker=chunker,
                    silence_warnings=silence_warnings,
                    **kwargs,
                )
            )
        if checks.is_namedtuple(obj):
            return type(obj)(*new_obj)
        return type(obj)(new_obj)


class MappingTaker(ContainerTaker):
    """Class for taking from a mapping container.

    Calls `Chunker.take_from_arg` on each element."""

    def adapt_cont_take_spec(self, obj: tp.Mapping) -> tp.ContainerTakeSpec:
        """Prepare the specification of the container to the object."""
        cont_take_spec = dict(self.cont_take_spec)
        ellipsis_take_spec = None
        ellipsis_found = False
        for k in cont_take_spec:
            if isinstance(k, type(...)):
                ellipsis_take_spec = cont_take_spec[k]
                ellipsis_found = True
        if ellipsis_found:
            for k, v in dict(obj).items():
                if k not in cont_take_spec:
                    cont_take_spec[k] = ellipsis_take_spec
        return cont_take_spec

    def suggest_size(self, obj: tp.Mapping, chunker: tp.Optional["Chunker"] = None, **kwargs) -> tp.Optional[int]:
        if self.mapper is not None:
            return None
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        size_k = None
        size = None
        for k, v in dict(obj).items():
            if k in cont_take_spec and cont_take_spec[k] is not MISSING:
                take_spec = chunker.resolve_take_spec(cont_take_spec[k])
                if isinstance(take_spec, ChunkTaker):
                    try:
                        new_size = take_spec.suggest_size(v)
                        if new_size is not None:
                            if size is None:
                                size_k = k
                                size = new_size
                            elif size != new_size:
                                warnings.warn(
                                    (
                                        f"Arguments with keys '{size_k}' and '{k}' have conflicting sizes "
                                        f"{size} and {new_size}. Setting size to None."
                                    ),
                                    stacklevel=2,
                                )
                                return None
                    except NotImplementedError as e:
                        pass
        return size

    def take(
        self,
        obj: tp.Mapping,
        chunk_meta: ChunkMeta,
        chunker: tp.Optional["Chunker"] = None,
        silence_warnings: bool = False,
        **kwargs,
    ) -> tp.Mapping:
        self.check_cont_take_spec()
        cont_take_spec = self.adapt_cont_take_spec(obj)
        if chunker is None:
            chunker = Chunker
        new_obj = {}
        for k, v in dict(obj).items():
            if k in cont_take_spec and cont_take_spec[k] is not MISSING:
                take_spec = cont_take_spec[k]
            else:
                if not silence_warnings:
                    warnings.warn(
                        (
                            f"Argument with key '{k}' not found in MappingTaker.cont_take_spec. "
                            "Setting its specification to None."
                        ),
                        stacklevel=2,
                    )
                take_spec = None
            new_obj[k] = chunker.take_from_arg(
                v,
                take_spec,
                chunk_meta,
                chunker=chunker,
                silence_warnings=silence_warnings,
                **kwargs,
            )
        return type(obj)(new_obj)


class ArgsTaker(SequenceTaker):
    """Class for taking from a variable arguments container."""

    def __init__(
        self,
        *args,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
    ) -> None:
        SequenceTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=args,
        )


class KwargsTaker(MappingTaker):
    """Class for taking from a variable keyword arguments container."""

    def __init__(
        self,
        single_type: tp.Optional[tp.TypeLike] = None,
        ignore_none: bool = True,
        mapper: tp.Optional[ChunkMapper] = None,
        **kwargs,
    ) -> None:
        MappingTaker.__init__(
            self,
            single_type=single_type,
            ignore_none=ignore_none,
            mapper=mapper,
            cont_take_spec=kwargs,
        )


# ############# Chunkables ############# #


class Chunkable(Evaluable, Annotatable):
    """Abstract class representing a value and a chunk taking specification."""

    def get_value(self) -> tp.Any:
        """Get the value."""
        raise NotImplementedError

    def get_take_spec(self) -> tp.TakeSpec:
        """Get the chunk taking specification."""
        raise NotImplementedError


@define
class Chunked(Chunkable, DefineMixin):
    """Class representing a chunkable value.

    Can take a variable number of keyword arguments, which will be used as `Chunked.take_spec_kwargs`."""

    value: tp.Any = define.required_field()
    """Value."""

    take_spec: tp.TakeSpec = define.optional_field()
    """Chunk taking specification."""

    take_spec_kwargs: tp.KwargsLike = define.field(default=None)
    """Keyword arguments passed to the respective `ChunkTaker` subclass.

    If `Chunked.take_spec` is an instance rather than a class, will "evolve" it."""

    select: bool = define.field(default=False)
    """Whether to chunk by selection."""

    eval_id: tp.Optional[tp.MaybeSequence[tp.Hashable]] = define.field(default=None)
    """One or more identifiers at which to evaluate this instance."""

    def __init__(self, *args, **kwargs) -> None:
        attr_names = [a.name for a in self.fields]
        if attr_names.index("take_spec_kwargs") < len(args):
            new_args = list(args)
            take_spec_kwargs = new_args[attr_names.index("take_spec_kwargs")]
            if take_spec_kwargs is None:
                take_spec_kwargs = {}
            else:
                take_spec_kwargs = dict(take_spec_kwargs)
            take_spec_kwargs.update({k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names})
            new_args[attr_names.index("take_spec_kwargs")] = take_spec_kwargs
            args = tuple(new_args)
        else:
            take_spec_kwargs = kwargs.pop("take_spec_kwargs", None)
            if take_spec_kwargs is None:
                take_spec_kwargs = {}
            else:
                take_spec_kwargs = dict(take_spec_kwargs)
            take_spec_kwargs.update({k: kwargs.pop(k) for k in list(kwargs.keys()) if k not in attr_names})
            kwargs["take_spec_kwargs"] = take_spec_kwargs

        DefineMixin.__init__(self, *args, **kwargs)

    def get_value(self) -> tp.Any:
        self.assert_field_not_missing("value")
        return self.value

    @property
    def take_spec_missing(self) -> bool:
        """Check whether `Chunked.take_spec` is missing."""
        return self.take_spec is MISSING

    def resolve_take_spec(self) -> tp.TakeSpec:
        """Resolve `take_spec`."""
        if self.take_spec_missing:
            if self.select:
                return ChunkSelector
            return ChunkSlicer
        return self.take_spec

    def get_take_spec(self) -> tp.TakeSpec:
        take_spec = self.resolve_take_spec()
        take_spec_kwargs = self.take_spec_kwargs
        if take_spec_kwargs is None:
            take_spec_kwargs = {}
        else:
            take_spec_kwargs = dict(take_spec_kwargs)
        if "eval_id" not in take_spec_kwargs:
            take_spec_kwargs["eval_id"] = self.eval_id
        if isinstance(take_spec, type) and issubclass(take_spec, ChunkTaker):
            take_spec = take_spec(**take_spec_kwargs)
        elif isinstance(take_spec, ChunkTaker):
            take_spec = take_spec.replace(**take_spec_kwargs)
        return take_spec


class ChunkedCount(Chunked):
    """Class representing a chunkable count."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            return CountAdapter
        return self.take_spec


class ChunkedShape(Chunked):
    """Class representing a chunkable shape."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.select:
                return ShapeSelector
            return ShapeSlicer
        return self.take_spec


@define
class ChunkedArray(Chunked, DefineMixin):
    """Class representing a chunkable array."""

    flex: bool = define.field(default=False)
    """Whether the array is flexible."""

    def resolve_take_spec(self) -> tp.TakeSpec:
        if self.take_spec_missing:
            if self.flex:
                if self.select:
                    from vectorbtpro.base.chunking import FlexArraySelector

                    return FlexArraySelector
                from vectorbtpro.base.chunking import FlexArraySlicer

                return FlexArraySlicer
            if self.select:
                return ArraySelector
            return ArraySlicer
        return self.take_spec


# ############# Chunker ############# #


class Chunker(Configured):
    """Class responsible for chunking arguments of a function and running the function.

    Does the following:

    1. Generates chunk metadata by passing `n_chunks`, `size`, `min_size`, `chunk_len`,
        and `chunk_meta` to `Chunker.get_chunk_meta_from_args`.
    2. Splits arguments and keyword arguments by passing chunk metadata, `arg_take_spec`,
        and `template_context` to `Chunker.yield_tasks`, which yields one chunk at a time.
    3. Executes all chunks by passing `**execute_kwargs` to `vectorbtpro.utils.execution.execute`.
    4. Optionally, post-processes and merges the results by passing them and
        `**merge_kwargs` to `merge_func`.

    For defaults, see `vectorbtpro._settings.chunking`."""

    _settings_path: tp.SettingsPath = "chunking"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "func",
        "size",
        "min_size",
        "n_chunks",
        "chunk_len",
        "chunk_meta",
        "skip_single_chunk",
        "arg_take_spec",
        "template_context",
        "prepend_chunk_meta",
        "merge_func",
        "merge_kwargs",
        "return_raw_chunks",
        "silence_warnings",
        "forward_kwargs_as",
        "execute_kwargs",
        "disable",
    }

    def __init__(
        self,
        func: tp.Callable,
        size: tp.Optional[int] = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Optional[tp.SizeLike] = None,
        chunk_len: tp.Optional[tp.SizeLike] = None,
        chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
        prepend_chunk_meta: tp.Optional[bool] = None,
        skip_single_chunk: tp.Optional[bool] = None,
        arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
        template_context: tp.KwargsLike = None,
        merge_func: tp.Optional[tp.MergeFuncLike] = None,
        merge_kwargs: tp.KwargsLike = None,
        return_raw_chunks: tp.Optional[bool] = None,
        silence_warnings: tp.Optional[bool] = None,
        forward_kwargs_as: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        disable: tp.Optional[bool] = None,
        **kwargs,
    ) -> None:
        Configured.__init__(
            self,
            func=func,
            size=size,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            prepend_chunk_meta=prepend_chunk_meta,
            skip_single_chunk=skip_single_chunk,
            arg_take_spec=arg_take_spec,
            template_context=template_context,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            return_raw_chunks=return_raw_chunks,
            silence_warnings=silence_warnings,
            forward_kwargs_as=forward_kwargs_as,
            execute_kwargs=execute_kwargs,
            disable=disable,
            **kwargs,
        )

        self._func = func
        self._size = self.resolve_setting(size, "size")
        self._min_size = self.resolve_setting(min_size, "min_size")
        self._n_chunks = self.resolve_setting(n_chunks, "n_chunks")
        self._chunk_len = self.resolve_setting(chunk_len, "chunk_len")
        self._chunk_meta = self.resolve_setting(chunk_meta, "chunk_meta")
        self._prepend_chunk_meta = self.resolve_setting(prepend_chunk_meta, "prepend_chunk_meta")
        self._skip_single_chunk = self.resolve_setting(skip_single_chunk, "skip_single_chunk")
        self._arg_take_spec = self.resolve_setting(arg_take_spec, "arg_take_spec")
        self._template_context = self.resolve_setting(template_context, "template_context", merge=True)
        self._merge_func = self.resolve_setting(merge_func, "merge_func")
        self._merge_kwargs = self.resolve_setting(merge_kwargs, "merge_kwargs", merge=True)
        self._return_raw_chunks = self.resolve_setting(return_raw_chunks, "return_raw_chunks")
        self._silence_warnings = self.resolve_setting(silence_warnings, "silence_warnings")
        self._forward_kwargs_as = self.resolve_setting(forward_kwargs_as, "forward_kwargs_as", merge=True)
        self._execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        self._disable = self.resolve_setting(disable, "disable")

    @property
    def func(self) -> tp.Callable:
        """Function."""
        return self._func

    @property
    def size(self) -> tp.Optional[int]:
        """See `Chunker.get_chunk_meta_from_args`."""
        return self._size

    @property
    def min_size(self) -> tp.Optional[int]:
        """See `Chunker.get_chunk_meta_from_args`."""
        return self._min_size

    @property
    def n_chunks(self) -> tp.Optional[tp.SizeLike]:
        """See `Chunker.get_chunk_meta_from_args`."""
        return self._n_chunks

    @property
    def chunk_len(self) -> tp.Optional[tp.SizeLike]:
        """See `Chunker.get_chunk_meta_from_args`."""
        return self._chunk_len

    @property
    def chunk_meta(self) -> tp.Optional[tp.ChunkMetaLike]:
        """See `Chunker.get_chunk_meta_from_args`."""
        return self._chunk_meta

    @property
    def prepend_chunk_meta(self) -> tp.Optional[bool]:
        """Whether to prepend an instance of `ChunkMeta` to the arguments.

        If None, prepends automatically if the first argument is named 'chunk_meta'."""
        return self._prepend_chunk_meta

    @property
    def skip_single_chunk(self) -> bool:
        """Whether to execute the function directly if there's only one chunk."""
        return self._skip_single_chunk

    @property
    def arg_take_spec(self) -> tp.Optional[tp.ArgTakeSpecLike]:
        """See `yield_tasks`."""
        return self._arg_take_spec

    @property
    def template_context(self) -> tp.Kwargs:
        """Template context.

        Any template in both `execute_kwargs` and `merge_kwargs` will be substituted. You can use
        the keys `ann_args`, `chunk_meta`, `arg_take_spec`, and `tasks` to be replaced by
        the actual objects."""
        return self._template_context

    @property
    def merge_func(self) -> tp.Optional[tp.MergeFuncLike]:
        """Merging function.

        Resolved using `vectorbtpro.base.merging.resolve_merge_func`."""
        return self._merge_func

    @property
    def merge_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to the merging function."""
        return self._merge_kwargs

    @property
    def return_raw_chunks(self) -> bool:
        """Whether to return chunks in a raw format."""
        return self._return_raw_chunks

    @property
    def silence_warnings(self) -> bool:
        """Whether to silence any warnings."""
        return self._silence_warnings

    @property
    def forward_kwargs_as(self) -> tp.Kwargs:
        """Map to rename keyword arguments.

        Can also pass any variable from the scope of `Chunker.run`"""
        return self._forward_kwargs_as

    @property
    def execute_kwargs(self) -> tp.Kwargs:
        """Keyword arguments passed to `vectorbtpro.utils.execution.execute`."""
        return self._execute_kwargs

    @property
    def disable(self) -> bool:
        """Whether to disable chunking."""
        return self._disable

    @classmethod
    def get_chunk_meta_from_args(
        cls,
        ann_args: tp.AnnArgs,
        size: tp.Optional[tp.SizeLike] = None,
        min_size: tp.Optional[int] = None,
        n_chunks: tp.Optional[tp.SizeLike] = None,
        chunk_len: tp.Optional[tp.SizeLike] = None,
        chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
        **kwargs,
    ) -> tp.Iterable[ChunkMeta]:
        """Get chunk metadata from annotated arguments.

        Args:
            ann_args (dict): Arguments annotated with `vectorbtpro.utils.parsing.annotate_args`.
            size (int, Sizer, or callable): See `yield_chunk_meta`.

                Can be an integer, an instance of `Sizer`, or a callable taking
                the annotated arguments and returning a value.
            min_size (int): See `yield_chunk_meta`.
            n_chunks (int, str, Sizer, or callable): See `yield_chunk_meta`.

                Can be an integer, a string, an instance of `Sizer`, or a callable taking
                the annotated arguments and other keyword arguments and returning a value.
            chunk_len (int, str, Sizer, or callable): See `yield_chunk_meta`.

                Can be an integer, a string, an instance of `Sizer`, or a callable taking
                the annotated arguments and returning a value.
            chunk_meta (iterable of ChunkMeta, ChunkMetaGenerator, or callable): Chunk meta.

                Can be an iterable of `ChunkMeta`, an instance of `ChunkMetaGenerator`, or
                a callable taking the annotated arguments and other arguments and returning an iterable.
            **kwargs: Other keyword arguments passed to any callable.
        """
        if chunk_meta is None:
            if size is not None:
                if isinstance(size, Sizer):
                    size = size.apply(ann_args, **kwargs)
                elif callable(size):
                    size = size(ann_args, **kwargs)
                elif not isinstance(size, int):
                    raise TypeError(f"Type {type(size)} for size is not supported")
            if n_chunks is not None:
                if isinstance(n_chunks, Sizer):
                    n_chunks = n_chunks.apply(ann_args, **kwargs)
                elif callable(n_chunks):
                    n_chunks = n_chunks(ann_args, **kwargs)
                elif not isinstance(n_chunks, (int, str)):
                    raise TypeError(f"Type {type(n_chunks)} for n_chunks is not supported")
            if chunk_len is not None:
                if isinstance(chunk_len, Sizer):
                    chunk_len = chunk_len.apply(ann_args, **kwargs)
                elif callable(chunk_len):
                    chunk_len = chunk_len(ann_args, **kwargs)
                elif not isinstance(chunk_len, (int, str)):
                    raise TypeError(f"Type {type(chunk_len)} for chunk_len is not supported")
            return yield_chunk_meta(size=size, min_size=min_size, n_chunks=n_chunks, chunk_len=chunk_len)
        if isinstance(chunk_meta, ChunkMetaGenerator):
            return chunk_meta.get_chunk_meta(ann_args, **kwargs)
        if callable(chunk_meta):
            return chunk_meta(ann_args, **kwargs)
        return chunk_meta

    @classmethod
    def resolve_take_spec(cls, take_spec: tp.TakeSpec) -> tp.TakeSpec:
        """Resolve the chunk taking specification."""
        if isinstance(take_spec, type) and issubclass(take_spec, Chunked):
            take_spec = take_spec()
        if isinstance(take_spec, Chunkable):
            take_spec = take_spec.get_take_spec()
        if isinstance(take_spec, type) and issubclass(take_spec, (NotChunked, ChunkTaker)):
            take_spec = take_spec()
        return take_spec

    @classmethod
    def take_from_arg(
        cls,
        arg: tp.Any,
        take_spec: tp.TakeSpec,
        chunk_meta: ChunkMeta,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Any:
        """Take from the argument given the specification `take_spec`.

        If `take_spec` is None or it's an instance of `NotChunked`, returns the original object.
        Otherwise, must be an instance of `ChunkTaker`.

        `**kwargs` are passed to `ChunkTaker.apply`."""
        if take_spec is None:
            return arg
        take_spec = cls.resolve_take_spec(take_spec)
        if isinstance(take_spec, NotChunked):
            return arg
        if isinstance(take_spec, ChunkTaker):
            if not take_spec.meets_eval_id(eval_id):
                return arg
            return take_spec.apply(arg, chunk_meta, **kwargs)
        raise TypeError(f"Specification of type {type(take_spec)} is not supported")

    @classmethod
    def find_take_spec(
        cls,
        i: int,
        ann_arg_name: str,
        ann_arg: tp.Kwargs,
        arg_take_spec: tp.ArgTakeSpec,
    ) -> tp.TakeSpec:
        """Resolve the specification for an argument."""
        take_spec_found = False
        found_take_spec = None
        for k, v in arg_take_spec.items():
            if isinstance(k, int):
                if k == i:
                    take_spec_found = True
                    found_take_spec = v
                    break
            elif isinstance(k, Regex):
                if k.matches(ann_arg_name):
                    take_spec_found = True
                    found_take_spec = v
                    break
            elif isinstance(v, Regex):
                if v.matches(k):
                    take_spec_found = True
                    found_take_spec = v
                    break
            else:
                if k == ann_arg_name:
                    take_spec_found = True
                    found_take_spec = v
                    break
        if take_spec_found:
            found_take_spec = cls.resolve_take_spec(found_take_spec)
            if ann_arg["kind"] == inspect.Parameter.VAR_POSITIONAL:
                if not isinstance(found_take_spec, ContainerTaker):
                    if checks.is_sequence(found_take_spec):
                        found_take_spec = SequenceTaker(found_take_spec)
                    else:
                        found_take_spec = SequenceTaker([found_take_spec, ...])
            elif ann_arg["kind"] == inspect.Parameter.VAR_KEYWORD:
                if not isinstance(found_take_spec, ContainerTaker):
                    if checks.is_mapping(found_take_spec):
                        found_take_spec = MappingTaker(found_take_spec)
                    else:
                        found_take_spec = MappingTaker({...: found_take_spec})
            return found_take_spec
        return MISSING

    @classmethod
    def take_from_args(
        cls,
        ann_args: tp.AnnArgs,
        arg_take_spec: tp.ArgTakeSpec,
        chunk_meta: ChunkMeta,
        silence_warnings: bool = False,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Tuple[tp.Args, tp.Kwargs]:
        """Take from each in the annotated arguments given the specification using `Chunker.take_from_arg`.

        Additionally, passes to `Chunker.take_from_arg` as keyword arguments `ann_args` and `arg_take_spec`.

        `arg_take_spec` must be a dictionary, with keys being argument positions or names as generated by
        `vectorbtpro.utils.parsing.annotate_args`. For values, see `Chunker.take_from_arg`.

        Returns arguments and keyword arguments that can be directly passed to the function
        using `func(*args, **kwargs)`."""
        new_args = ()
        new_kwargs = dict()
        for i, (k, v) in enumerate(ann_args.items()):
            take_spec = cls.find_take_spec(i, k, v, arg_take_spec)
            if take_spec is MISSING:
                take_spec = None
                if not silence_warnings:
                    warnings.warn(
                        f"Argument '{k}' not found in arg_take_spec. Setting its specification to None.",
                        stacklevel=2,
                    )
            result = cls.take_from_arg(
                v["value"],
                take_spec,
                chunk_meta,
                ann_args=ann_args,
                arg_take_spec=arg_take_spec,
                silence_warnings=silence_warnings,
                eval_id=eval_id,
                **kwargs,
            )
            if v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                for new_arg in result:
                    new_args += (new_arg,)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                for new_kwarg_name, new_kwarg in result.items():
                    new_kwargs[new_kwarg_name] = new_kwarg
            elif v["kind"] == inspect.Parameter.KEYWORD_ONLY:
                new_kwargs[k] = result
            else:
                new_args += (result,)
        return new_args, new_kwargs

    @classmethod
    def yield_tasks(
        cls,
        func: tp.Callable,
        ann_args: tp.AnnArgs,
        chunk_meta: tp.Iterable[ChunkMeta],
        arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Generator[Task, None, None]:
        """Split annotated arguments into chunks using `Chunker.take_from_args` and yield each chunk as a task.

        Args:
            func (callable): Callable.
            ann_args (dict): Arguments annotated with `vectorbtpro.utils.parsing.annotate_args`.
            chunk_meta (iterable of ChunkMeta): Chunk metadata.
            arg_take_spec (mapping, sequence, callable, or CustomTemplate): Chunk taking specification.

                Can be a dictionary (see `Chunker.take_from_args`), or a sequence that will be
                converted into a dictionary. If a callable, will be called instead of `Chunker.take_from_args`,
                thus it must have the same arguments apart from `arg_take_spec`.
            template_context (mapping): Context used to substitute templates in arguments and specification.
            **kwargs: Keyword arguments passed to `Chunker.take_from_args` or to `arg_take_spec`
                if it's a callable.
        """
        if arg_take_spec is None:
            arg_take_spec = {}
        if template_context is None:
            template_context = {}

        for _chunk_meta in chunk_meta:
            _template_context = dict(template_context)
            _template_context["ann_args"] = ann_args
            _template_context["chunk_meta"] = _chunk_meta
            chunk_ann_args = substitute_templates(ann_args, _template_context, eval_id="chunk_ann_args")
            _template_context["chunk_ann_args"] = chunk_ann_args
            chunk_arg_take_spec = substitute_templates(arg_take_spec, _template_context, eval_id="chunk_arg_take_spec")
            _template_context["chunk_arg_take_spec"] = chunk_arg_take_spec

            if callable(chunk_arg_take_spec):
                chunk_args, chunk_kwargs = chunk_arg_take_spec(
                    chunk_ann_args,
                    _chunk_meta,
                    template_context=_template_context,
                    **kwargs,
                )
            else:
                if not checks.is_mapping(chunk_arg_take_spec):
                    chunk_arg_take_spec = dict(zip(range(len(chunk_arg_take_spec)), chunk_arg_take_spec))
                chunk_args, chunk_kwargs = cls.take_from_args(
                    chunk_ann_args,
                    chunk_arg_take_spec,
                    _chunk_meta,
                    template_context=_template_context,
                    **kwargs,
                )
            yield Task(func, *chunk_args, **chunk_kwargs)

    @classmethod
    def parse_sizer_from_func(
        cls,
        func: tp.Callable,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.Optional[Sizer]:
        """Parse the sizer from a function."""
        annotations = flatten_annotations(get_annotations(func))
        sizer = None
        for k, v in annotations.items():
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                if isinstance(annotation, type) and issubclass(annotation, Sizer):
                    annotation = annotation()
                if isinstance(annotation, Sizer) and annotation.meets_eval_id(eval_id):
                    if isinstance(annotation, ArgGetter):
                        if annotation.arg_query is None:
                            annotation = annotation.replace(arg_query=k)
                    if sizer is not None:
                        raise ValueError(f"Two sizers found in annotations: {sizer} and {annotation}")
                    sizer = annotation
        return sizer

    @classmethod
    def parse_spec_from_annotations(
        cls,
        annotations: tp.Annotations,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse the chunk taking specification from annotations."""
        arg_take_spec = {}
        for k, v in annotations.items():
            if not isinstance(v, Union):
                v = Union(v)
            for annotation in v.annotations:
                annotation = cls.resolve_take_spec(annotation)
                if isinstance(annotation, ChunkTaker) and annotation.meets_eval_id(eval_id):
                    if isinstance(annotation, ArgGetter):
                        if annotation.arg_query is None:
                            annotation = annotation.replace(arg_query=k)
                    if k in arg_take_spec:
                        raise ValueError(
                            f"Two specifications found in annotations for the key '{k}': "
                            f"{arg_take_spec[k]} and {annotation}"
                        )
                    arg_take_spec[k] = annotation
        return arg_take_spec

    @classmethod
    def parse_spec_from_func(
        cls,
        func: tp.Callable,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse the chunk taking specification from a function."""
        annotations = get_annotations(func)
        arg_take_spec = cls.parse_spec_from_annotations(annotations, eval_id=eval_id)
        flat_annotations, var_args_map, var_kwargs_map = flatten_annotations(
            annotations,
            only_var_args=True,
            return_var_arg_maps=True,
        )
        if len(flat_annotations) > 0:
            flat_arg_take_spec = cls.parse_spec_from_annotations(flat_annotations, eval_id=eval_id)
            if len(var_args_map) > 0:
                var_args_name = None
                var_args_specs = []
                for k in var_args_map:
                    if k in flat_arg_take_spec:
                        if var_args_map[k] in arg_take_spec:
                            raise ValueError(
                                "Two specifications found in annotations: "
                                f"{arg_take_spec[var_args_map[k]]} ('*{var_args_map[k]}') and "
                                f"{flat_arg_take_spec[k]} ('{k}')"
                            )
                        if var_args_name is None:
                            var_args_name = var_args_map[k]
                        i = int(k.split("_")[-1])
                        if i > len(var_args_specs):
                            var_args_specs.extend([MISSING] * (i - len(var_args_specs)))
                        var_args_specs.append(flat_arg_take_spec[k])
                if len(var_args_specs) > 0:
                    arg_take_spec[var_args_name] = ArgsTaker(*var_args_specs)
            if len(var_kwargs_map) > 0:
                var_kwargs_name = None
                var_kwargs_specs = dict()
                for k in var_kwargs_map:
                    if k in flat_arg_take_spec:
                        if var_kwargs_map[k] in arg_take_spec:
                            raise ValueError(
                                "Two specifications found in annotations: "
                                f"{arg_take_spec[var_kwargs_map[k]]} ('**{var_kwargs_map[k]}') and "
                                f"{flat_arg_take_spec[k]} ('{k}')"
                            )
                        if var_kwargs_name is None:
                            var_kwargs_name = var_kwargs_map[k]
                        var_kwargs_specs[k] = flat_arg_take_spec[k]
                if len(var_kwargs_specs) > 0:
                    arg_take_spec[var_kwargs_name] = KwargsTaker(**var_kwargs_specs)
        return arg_take_spec

    @classmethod
    def parse_spec_from_args(
        cls,
        ann_args: tp.AnnArgs,
        eval_id: tp.Optional[tp.Hashable] = None,
    ) -> tp.ArgTakeSpec:
        """Parse the chunk taking specification from (annotated) arguments."""
        arg_take_spec = {}
        for k, v in ann_args.items():
            if isinstance(v["value"], Chunkable) and v["value"].meets_eval_id(eval_id):
                arg_take_spec[k] = v["value"].get_take_spec()
            elif v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                chunkable_found = False
                for v2 in v["value"]:
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        chunkable_found = True
                        break
                if chunkable_found:
                    take_spec = []
                    for v2 in v["value"]:
                        if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                            take_spec.append(v2.get_take_spec())
                        else:
                            take_spec.append(MISSING)
                    arg_take_spec[k] = ArgsTaker(*take_spec)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                chunkable_found = False
                for v2 in v["value"].values():
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        chunkable_found = True
                        break
                if chunkable_found:
                    take_spec = {}
                    for k2, v2 in v["value"].items():
                        if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                            take_spec[k2] = v2.get_take_spec()
                        else:
                            take_spec[k2] = MISSING
                    arg_take_spec[k] = KwargsTaker(**take_spec)
        return arg_take_spec

    @classmethod
    def fill_arg_take_spec(cls, arg_take_spec: tp.ArgTakeSpec, ann_args: tp.AnnArgs) -> tp.ArgTakeSpec:
        """Fill the chunk taking specification with None to avoid warnings."""
        arg_take_spec = dict(arg_take_spec)
        for k, v in ann_args.items():
            if k not in arg_take_spec:
                arg_take_spec[k] = None
        return arg_take_spec

    @classmethod
    def adapt_ann_args(cls, ann_args: tp.AnnArgs, eval_id: tp.Optional[tp.Hashable] = None) -> tp.AnnArgs:
        """Adapt annotated arguments."""
        new_ann_args = {}
        for k, v in ann_args.items():
            new_ann_args[k] = v = dict(v)
            if isinstance(v["value"], Chunkable) and v["value"].meets_eval_id(eval_id):
                v["value"] = v["value"].get_value()
            elif v["kind"] == inspect.Parameter.VAR_POSITIONAL:
                new_value = []
                for v2 in v["value"]:
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        new_value.append(v2.get_value())
                    else:
                        new_value.append(v2)
                v["value"] = tuple(new_value)
            elif v["kind"] == inspect.Parameter.VAR_KEYWORD:
                new_value = {}
                for k2, v2 in v["value"].items():
                    if isinstance(v2, Chunkable) and v2.meets_eval_id(eval_id):
                        new_value[k2] = v2.get_value()
                    else:
                        new_value[k2] = v2
                v["value"] = new_value
        return new_ann_args

    @classmethod
    def suggest_size(
        cls,
        ann_args: tp.AnnArgs,
        arg_take_spec: tp.ArgTakeSpec,
        eval_id: tp.Optional[tp.Hashable] = None,
        **kwargs,
    ) -> tp.Optional[int]:
        """Suggest a global size given the annotated arguments and the chunk taking specification."""
        size_k = None
        size = None
        for i, (k, v) in enumerate(ann_args.items()):
            take_spec = cls.find_take_spec(i, k, v, arg_take_spec)
            if isinstance(take_spec, ChunkTaker) and take_spec.meets_eval_id(eval_id):
                try:
                    new_size = take_spec.suggest_size(v["value"], **kwargs)
                    if new_size is not None:
                        if size is None:
                            size_k = k
                            size = new_size
                        elif size != new_size:
                            warnings.warn(
                                (
                                    f"Arguments '{size_k}' and '{k}' have conflicting sizes "
                                    f"{size} and {new_size}. Setting size to None."
                                ),
                                stacklevel=2,
                            )
                            return None
                except NotImplementedError as e:
                    pass
        return size

    def run(self, *args, eval_id: tp.Optional[tp.Hashable] = None, **kwargs) -> tp.Any:
        """Chunk arguments and run the function."""
        func = self.func
        size = self.size
        min_size = self.min_size
        n_chunks = self.n_chunks
        chunk_len = self.chunk_len
        chunk_meta = self.chunk_meta
        prepend_chunk_meta = self.prepend_chunk_meta
        skip_single_chunk = self.skip_single_chunk
        arg_take_spec = self.arg_take_spec
        template_context = self.template_context
        merge_func = self.merge_func
        merge_kwargs = self.merge_kwargs
        return_raw_chunks = self.return_raw_chunks
        silence_warnings = self.silence_warnings
        forward_kwargs_as = self.forward_kwargs_as
        execute_kwargs = self.execute_kwargs
        disable = self.disable

        template_context["eval_id"] = eval_id

        if arg_take_spec is None:
            arg_take_spec = {}
        if checks.is_mapping(arg_take_spec):
            main_arg_take_spec = dict(arg_take_spec)
            arg_take_spec = dict(arg_take_spec)
            if "chunk_meta" not in arg_take_spec:
                arg_take_spec["chunk_meta"] = None
        else:
            main_arg_take_spec = None

        if forward_kwargs_as is None:
            forward_kwargs_as = {}
        if len(forward_kwargs_as) > 0:
            new_kwargs = dict()
            for k, v in kwargs.items():
                if k in forward_kwargs_as:
                    new_kwargs[forward_kwargs_as.pop(k)] = v
                else:
                    new_kwargs[k] = v
            kwargs = new_kwargs
        if len(forward_kwargs_as) > 0:
            for k, v in forward_kwargs_as.items():
                kwargs[v] = locals()[k]

        if disable:
            return func(*args, **kwargs)

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == "chunk_meta":
                    prepend_chunk_meta = True
        if prepend_chunk_meta:
            args = (Rep("chunk_meta"), *args)

        parsed_sizer = self.parse_sizer_from_func(func, eval_id=eval_id)
        if parsed_sizer is not None:
            if size is not None:
                raise ValueError(f"Two conflicting sizers: {parsed_sizer} (annotations) and {size} (size)")
            size = parsed_sizer
        parsed_arg_take_spec = self.parse_spec_from_func(func, eval_id=eval_id)
        if len(parsed_arg_take_spec) > 0:
            if not isinstance(arg_take_spec, dict) or parsed_arg_take_spec.keys() & arg_take_spec.keys():
                raise ValueError(
                    f"Two conflicting specifications: {parsed_arg_take_spec} (annotations) "
                    f"and {arg_take_spec} (arg_take_spec)"
                )
            arg_take_spec = {**parsed_arg_take_spec, **arg_take_spec}
        parsed_merge_func = parse_merge_func(func, eval_id=eval_id)
        if parsed_merge_func is not None:
            if merge_func is not None:
                raise ValueError(
                    f"Two conflicting merge functions: {parsed_merge_func} (annotations) and {merge_func} (merge_func)"
                )
            merge_func = parsed_merge_func
        ann_args = annotate_args(func, args, kwargs)
        parsed_arg_take_spec = self.parse_spec_from_args(ann_args, eval_id=eval_id)
        if len(parsed_arg_take_spec) > 0:
            if not isinstance(arg_take_spec, dict) or parsed_arg_take_spec.keys() & arg_take_spec.keys():
                raise ValueError(
                    f"Two conflicting specifications: {parsed_arg_take_spec} (arguments) "
                    f"and {arg_take_spec} (arg_take_spec & annotations)"
                )
            arg_take_spec = {**parsed_arg_take_spec, **arg_take_spec}
        if main_arg_take_spec is not None and len(main_arg_take_spec) == 0 and len(arg_take_spec) > 0:
            arg_take_spec = self.fill_arg_take_spec(arg_take_spec, ann_args)
        ann_args = self.adapt_ann_args(ann_args, eval_id=eval_id)
        args, kwargs = ann_args_to_args(ann_args)
        template_context["chunker"] = self
        template_context["arg_take_spec"] = arg_take_spec
        template_context["ann_args"] = ann_args

        if size is None and isinstance(arg_take_spec, dict):
            size = self.suggest_size(
                ann_args,
                arg_take_spec,
                template_context=template_context,
                silence_warnings=silence_warnings,
                chunker=self,
                eval_id=eval_id,
            )
        template_context["size"] = size
        chunk_meta = list(
            self.get_chunk_meta_from_args(
                ann_args,
                size=size,
                min_size=min_size,
                n_chunks=n_chunks,
                chunk_len=chunk_len,
                chunk_meta=chunk_meta,
                template_context=template_context,
                silence_warnings=silence_warnings,
                chunker=self,
                eval_id=eval_id,
            )
        )
        template_context["chunk_meta"] = chunk_meta
        if len(chunk_meta) < 2 and skip_single_chunk:
            return func(*args, **kwargs)
        tasks = self.yield_tasks(
            func,
            ann_args,
            chunk_meta,
            arg_take_spec=arg_take_spec,
            template_context=template_context,
            silence_warnings=silence_warnings,
            chunker=self,
            eval_id=eval_id,
        )
        if return_raw_chunks:
            return chunk_meta, tasks
        execute_kwargs = substitute_templates(execute_kwargs, template_context, eval_id="execute_kwargs")
        execute_kwargs = merge_dicts(dict(show_progress=False if len(chunk_meta) == 1 else None), execute_kwargs)
        keys = []
        for _chunk_meta in chunk_meta:
            if _chunk_meta.indices is not None:
                key = "{}..{}".format(_chunk_meta.indices[0], _chunk_meta.indices[-1])
            elif _chunk_meta.start is not None and _chunk_meta.end is not None:
                if _chunk_meta.start == _chunk_meta.end - 1:
                    key = _chunk_meta.start
                else:
                    key = "{}..{}".format(_chunk_meta.start, _chunk_meta.end - 1)
            else:
                key = MISSING
            if eval_id is not None:
                keys.append((MISSING, key))
            else:
                keys.append(key)
        if eval_id is not None:
            keys = pd.MultiIndex.from_tuples(keys, names=(f"eval_id={eval_id}", "chunk_indices"))
        else:
            keys = pd.Index(keys, name="chunk_indices")
        results = execute(tasks, size=len(chunk_meta), keys=keys, **execute_kwargs)
        if merge_func is not None:
            template_context["tasks"] = tasks
            if isinstance(merge_func, MergeFunc):
                merge_func = merge_func.replace(
                    merge_kwargs=merge_kwargs,
                    context=template_context,
                )
            else:
                merge_func = MergeFunc(
                    merge_func,
                    merge_kwargs=merge_kwargs,
                    context=template_context,
                )
            return merge_func(results)
        return results


def chunked(
    *args,
    chunker_cls: tp.Optional[tp.Type[Chunker]] = None,
    size: tp.Optional[tp.SizeLike] = None,
    min_size: tp.Optional[int] = None,
    n_chunks: tp.Optional[tp.SizeLike] = None,
    chunk_len: tp.Optional[tp.SizeLike] = None,
    chunk_meta: tp.Optional[tp.ChunkMetaLike] = None,
    prepend_chunk_meta: tp.Optional[bool] = None,
    skip_single_chunk: tp.Optional[bool] = None,
    arg_take_spec: tp.Optional[tp.ArgTakeSpecLike] = None,
    template_context: tp.KwargsLike = None,
    merge_func: tp.Optional[tp.MergeFuncLike] = None,
    merge_kwargs: tp.KwargsLike = None,
    return_raw_chunks: bool = False,
    silence_warnings: tp.Optional[bool] = None,
    forward_kwargs_as: tp.KwargsLike = None,
    execute_kwargs: tp.KwargsLike = None,
    merge_to_execute_kwargs: tp.Optional[bool] = None,
    disable: tp.Optional[bool] = None,
    eval_id: tp.Optional[tp.Hashable] = None,
    **kwargs,
) -> tp.Callable:
    """Decorator that chunks the inputs of a function using `Chunker`.

    Returns a new function with the same signature as the passed one.

    Each option can be modified in the `options` attribute of the wrapper function or
    directly passed as a keyword argument with a leading underscore.

    Keyword arguments `**kwargs` and `execute_kwargs` are merged into `execute_kwargs`
    if `merge_to_execute_kwargs` is True, otherwise, `**kwargs` are passed directly to `Chunker`.

    Chunking can be disabled using `disable` argument. Additionally, the entire wrapping mechanism
    can be disabled by using the global setting `disable_wrapping` (=> returns the wrapped function).

    Usage:
        For testing purposes, let's divide the input array into 2 chunks and compute
        the mean in a sequential manner:

        ```pycon
        >>> from vectorbtpro import *

        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer())
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Same can be done using annotations:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.LenSizer() | vbt.ChunkSlicer()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Sizer can be omitted most of the time:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkSlicer()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        [2.0, 7.0]
        ```

        Another way is by using specialized `Chunker` subclasses that depend on the type of the argument:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkedArray()):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        ```

        Also, instead of specifying the chunk taking specification beforehand, it can be passed
        dynamically by wrapping each value to be chunked with `Chunked` or any of its subclasses:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a):
        ...     return np.mean(a)

        >>> f(vbt.ChunkedArray(np.arange(10)))
        [2.0, 7.0]
        ```

        The `chunked` function is a decorator that takes `f` and creates a function that splits
        passed arguments, runs each chunk using an engine, and optionally, merges the results.
        It has the same signature as the original function:

        ```pycon
        >>> f
        <function __main__.f(a)>
        ```

        We can change any option at any time:

        ```pycon
        >>> # Change the option directly on the function
        >>> f.options.n_chunks = 3

        >>> f(np.arange(10))
        [1.5, 5.0, 8.0]

        >>> # Pass a new option with a leading underscore
        >>> f(np.arange(10), _n_chunks=4)
        [1.0, 4.0, 6.5, 8.5]
        ```

        When we run the wrapped function, it first generates a list of chunk metadata of type `ChunkMeta`.
        Chunk metadata contains the chunk index that can be used to split any input:

        ```pycon
        >>> list(vbt.yield_chunk_meta(n_chunks=2))
        [ChunkMeta(uuid='84d64eed-fbac-41e7-ad61-c917e809b3b8', idx=0, start=None, end=None, indices=None),
         ChunkMeta(uuid='577817c4-fdee-4ceb-ab38-dcd663d9ab11', idx=1, start=None, end=None, indices=None)]
        ```

        Additionally, it may contain the start and end index of the space we want to split.
        The space can be defined by the length of an input array, for example. In our case:

        ```pycon
        >>> list(vbt.yield_chunk_meta(n_chunks=2, size=10))
        [ChunkMeta(uuid='c1593842-dc31-474c-a089-e47200baa2be', idx=0, start=0, end=5, indices=None),
         ChunkMeta(uuid='6d0265e7-1204-497f-bc2c-c7b7800ec57d', idx=1, start=5, end=10, indices=None)]
        ```

        If we know the size of the space in advance, we can pass it as an integer constant.
        Otherwise, we need to tell `chunked` to derive the size from the inputs dynamically
        by passing any subclass of `Sizer`. In the example above, we instruct the wrapped function
        to derive the size from the length of the input array `a`.

        Once all chunks are generated, the wrapped function attempts to split inputs into chunks.
        The specification for this operation can be provided by the `arg_take_spec` argument, which
        in most cases is a dictionary of `ChunkTaker` instances keyed by the input name.
        Here's an example of a complex specification:

        ```pycon
        >>> arg_take_spec = dict(
        ...     a=vbt.ChunkSelector(),
        ...     args=vbt.ArgsTaker(
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ),
        ...     b=vbt.SequenceTaker([
        ...         None,
        ...         vbt.ChunkSelector()
        ...     ]),
        ...     kwargs=vbt.KwargsTaker(
        ...         c=vbt.MappingTaker(dict(
        ...             d=vbt.ChunkSelector(),
        ...             e=None
        ...         ))
        ...     )
        ... )

        >>> @vbt.chunked(
        ...     n_chunks=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=arg_take_spec
        ... )
        ... def f(a, *args, b=None, **kwargs):
        ...     return a + sum(args) + sum(b) + sum(kwargs['c'].values())

        >>> f([1, 2, 3], 10, [1, 2, 3], b=(100, [1, 2, 3]), c=dict(d=[1, 2, 3], e=1000))
        [1114, 1118, 1122]
        ```

        After splitting all inputs into chunks, the wrapped function forwards them to the engine function.
        The engine argument can be either the name of a supported engine, or a callable. Once the engine
        has finished all tasks and returned a list of results, we can merge them back using `merge_func`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     merge_func="concat"
        ... )
        ... def f(a):
        ...     return a

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        The same using annotations:

        ```pycon
        >>> @vbt.chunked(n_chunks=2)
        ... def f(a: vbt.ChunkSlicer()) -> vbt.MergeFunc("concat"):
        ...     return a

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Instead of (or in addition to) specifying `arg_take_spec`, we can define our function with the
        first argument being `chunk_meta` to be able to split the arguments during the execution.
        The `chunked` decorator will automatically recognize and replace it with the actual `ChunkMeta` object:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat"
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        This may be a good idea in multi-threading, but a bad idea in multi-processing.

        The same can be accomplished by using templates (here we tell `chunked` to not replace
        the first argument by setting `prepend_chunk_meta` to False):

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=None),
        ...     merge_func="concat",
        ...     prepend_chunk_meta=False
        ... )
        ... def f(chunk_meta, a):
        ...     return a[chunk_meta.start:chunk_meta.end]

        >>> f(vbt.Rep('chunk_meta'), np.arange(10))
        array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        ```

        Templates in arguments are substituted right before taking a chunk from them.

        Keyword arguments to the engine can be provided using `execute_kwargs`:

        ```pycon
        >>> @vbt.chunked(
        ...     n_chunks=2,
        ...     size=vbt.LenSizer(arg_query='a'),
        ...     arg_take_spec=dict(a=vbt.ChunkSlicer()),
        ...     show_progress=True
        ... )
        ... def f(a):
        ...     return np.mean(a)

        >>> f(np.arange(10))
        100% |█████████████████████████████████| 2/2 [00:00<00:00, 81.11it/s]
        [2.0, 7.0]
        ```
    """

    def decorator(func: tp.Callable) -> tp.Callable:
        nonlocal prepend_chunk_meta

        from vectorbtpro._settings import settings

        chunking_cfg = settings["chunking"]

        if chunking_cfg["disable_wrapping"]:
            return func

        if prepend_chunk_meta is None:
            prepend_chunk_meta = False
            func_arg_names = get_func_arg_names(func)
            if len(func_arg_names) > 0:
                if func_arg_names[0] == "chunk_meta":
                    prepend_chunk_meta = True

        if merge_to_execute_kwargs is None:
            _merge_to_execute_kwargs = chunking_cfg["merge_to_execute_kwargs"]
        else:
            _merge_to_execute_kwargs = merge_to_execute_kwargs
        if _merge_to_execute_kwargs:
            _execute_kwargs = merge_dicts(kwargs, execute_kwargs)
            _chunker_kwargs = {}
        else:
            _execute_kwargs = execute_kwargs
            _chunker_kwargs = kwargs

        @wraps(func)
        def wrapper(*args, **kwargs) -> tp.Any:
            def _resolve_key(key, merge=False):
                if "_" + key in kwargs:
                    if merge:
                        return merge_dicts(wrapper.options[key], kwargs.pop("_" + key))
                    return kwargs.pop("_" + key)
                return wrapper.options[key]

            chunker_cls = wrapper.options["chunker_cls"]
            if chunker_cls is None:
                chunker_cls = chunking_cfg["chunker_cls"]
            if chunker_cls is None:
                chunker_cls = Chunker
            return chunker_cls(
                func=func,
                size=_resolve_key("size"),
                min_size=_resolve_key("min_size"),
                n_chunks=_resolve_key("n_chunks"),
                chunk_len=_resolve_key("chunk_len"),
                skip_single_chunk=_resolve_key("skip_single_chunk"),
                chunk_meta=_resolve_key("chunk_meta"),
                prepend_chunk_meta=prepend_chunk_meta,
                arg_take_spec=_resolve_key("arg_take_spec"),
                template_context=_resolve_key("template_context", merge=True),
                execute_kwargs=_resolve_key("execute_kwargs", merge=True),
                merge_func=_resolve_key("merge_func"),
                merge_kwargs=_resolve_key("merge_kwargs", merge=True),
                return_raw_chunks=_resolve_key("return_raw_chunks"),
                silence_warnings=_resolve_key("silence_warnings"),
                forward_kwargs_as=_resolve_key("forward_kwargs_as", merge=True),
                disable=_resolve_key("disable"),
                **_resolve_key("chunker_kwargs", merge=True),
            ).run(*args, eval_id=_resolve_key("eval_id"), **kwargs)

        wrapper.func = func
        wrapper.name = func.__name__
        wrapper.is_chunked = True
        wrapper.options = FrozenConfig(
            chunker_cls=chunker_cls,
            size=size,
            min_size=min_size,
            n_chunks=n_chunks,
            chunk_len=chunk_len,
            chunk_meta=chunk_meta,
            skip_single_chunk=skip_single_chunk,
            arg_take_spec=arg_take_spec,
            template_context=template_context,
            merge_func=merge_func,
            merge_kwargs=merge_kwargs,
            return_raw_chunks=return_raw_chunks,
            silence_warnings=silence_warnings,
            forward_kwargs_as=forward_kwargs_as,
            execute_kwargs=_execute_kwargs,
            chunker_kwargs=_chunker_kwargs,
            disable=disable,
            eval_id=eval_id,
        )

        if prepend_chunk_meta:
            signature = inspect.signature(wrapper)
            wrapper.__signature__ = signature.replace(parameters=tuple(signature.parameters.values())[1:])

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    raise ValueError("Either function or keyword arguments must be passed")


# ############# Chunking option ############# #


def resolve_chunked_option(option: tp.ChunkedOption = None) -> tp.KwargsLike:
    """Return keyword arguments for `chunked`.

    `option` can be:

    * True: Chunk using default settings
    * None or False: Do not chunk
    * string: Use `option` as the name of an execution engine (see `vectorbtpro.utils.execution.execute`)
    * dict: Use `option` as keyword arguments passed to `chunked`

    For defaults, see `option` in `vectorbtpro._settings.chunking`."""
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    if option is None:
        option = chunking_cfg["option"]

    if isinstance(option, bool):
        if not option:
            return None
        return dict()
    if isinstance(option, dict):
        return option
    elif isinstance(option, str):
        return dict(engine=option)
    raise TypeError(f"Type {type(option)} is invalid for a chunking option")


def specialize_chunked_option(option: tp.ChunkedOption = None, **kwargs) -> tp.KwargsLike:
    """Resolve `option` and merge it with `kwargs` if it's not None so the dict can be passed
    as an option to other functions."""
    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        return merge_dicts(kwargs, chunked_kwargs)
    return chunked_kwargs


def resolve_chunked(func: tp.Callable, option: tp.ChunkedOption = None, **kwargs) -> tp.Callable:
    """Decorate with `chunked` based on an option."""
    from vectorbtpro._settings import settings

    chunking_cfg = settings["chunking"]

    chunked_kwargs = resolve_chunked_option(option)
    if chunked_kwargs is not None:
        if isinstance(chunking_cfg["option"], dict):
            chunked_kwargs = merge_dicts(chunking_cfg["option"], kwargs, chunked_kwargs)
        else:
            chunked_kwargs = merge_dicts(kwargs, chunked_kwargs)
        return chunked(func, **chunked_kwargs)
    return func
