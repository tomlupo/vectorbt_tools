# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Class decorators for data."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks
from vectorbtpro.utils.config import copy_dict

__all__ = []


def attach_symbol_dict_methods(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
    """Class decorator to attach methods for updating symbol dictionaries."""

    checks.assert_subclass_of(cls, "Data")

    DataT = tp.TypeVar("DataT", bound="Data")

    for target_name in cls._key_dict_attrs:

        def select_method(self, key: tp.Key, _target_name=target_name, **kwargs) -> tp.Any:
            if _target_name.endswith("_kwargs"):
                return self.select_key_kwargs(
                    key,
                    getattr(self, _target_name),
                    kwargs_name=_target_name,
                    **kwargs,
                )
            return self.select_key_from_dict(
                key,
                getattr(self, _target_name),
                dct_name=_target_name,
                **kwargs,
            )

        select_method.__name__ = "select_" + target_name
        select_method.__module__ = cls.__module__
        select_method.__qualname__ = f"{cls.__name__}.{select_method.__name__}"
        select_method.__doc__ = f"""Select a feature or symbol from `Data.{target_name}`."""
        setattr(cls, select_method.__name__, select_method)

    for target_name in cls._updatable_attrs:

        def update_method(self: DataT, _target_name=target_name, check_dict_type: bool = True, **kwargs) -> DataT:
            from vectorbtpro.data.base import key_dict

            new_kwargs = copy_dict(getattr(self, _target_name))
            for s in self.get_keys(type(new_kwargs)):
                if s not in new_kwargs:
                    new_kwargs[s] = dict()
            for k, v in kwargs.items():
                if check_dict_type:
                    self.check_dict_type(v, k, dict_type=type(new_kwargs))
                if type(v) is key_dict or isinstance(v, type(new_kwargs)):
                    for s, _v in v.items():
                        new_kwargs[s][k] = _v
                else:
                    for s in new_kwargs:
                        new_kwargs[s][k] = v
            return self.replace(**{_target_name: new_kwargs})

        update_method.__name__ = "update_" + target_name
        update_method.__module__ = cls.__module__
        update_method.__qualname__ = f"{cls.__name__}.{update_method.__name__}"
        update_method.__doc__ = f"""Update `Data.{target_name}`. Returns a new instance."""
        setattr(cls, update_method.__name__, update_method)

    return cls
