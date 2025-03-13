# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Global registry for progress bars."""

import uuid

from vectorbtpro import _typing as tp

if tp.TYPE_CHECKING:
    from vectorbtpro.utils.pbar import ProgressBar as ProgressBarT
else:
    ProgressBarT = "ProgressBar"

__all__ = [
    "PBarRegistry",
    "pbar_reg",
]


class PBarRegistry:
    """Class for registering `vectorbtpro.utils.pbar.ProgressBar` instances."""

    @classmethod
    def generate_bar_id(cls) -> tp.Hashable:
        """Generate a unique bar id."""
        return str(uuid.uuid4())

    def __init__(self):
        self._instances = {}

    @property
    def instances(self) -> tp.Dict[tp.Hashable, ProgressBarT]:
        """Dict of registered instances by their bar id."""
        return self._instances

    def register_instance(self, instance: ProgressBarT) -> None:
        """Register an instance."""
        self.instances[instance.bar_id] = instance

    def deregister_instance(self, instance: ProgressBarT) -> None:
        """Deregister an instance."""
        if instance.bar_id in self.instances:
            del self.instances[instance.bar_id]

    def has_conflict(self, instance: ProgressBarT) -> bool:
        """Return whether there is an (active) instance with the same bar id."""
        if instance.bar_id is None:
            return False
        for inst in self.instances.values():
            if inst is not instance and inst.bar_id == instance.bar_id and inst.active:
                return True
        return False

    def get_last_active_instance(self) -> tp.Optional[ProgressBarT]:
        """Get the last active instance."""
        max_open_time = None
        last_active_instance = None
        for inst in self.instances.values():
            if inst.active:
                if max_open_time is None or inst.open_time > max_open_time:
                    max_open_time = inst.open_time
                    last_active_instance = inst
        return last_active_instance

    def get_first_pending_instance(self) -> tp.Optional[ProgressBarT]:
        """Get the first pending instance."""
        last_active_instance = self.get_last_active_instance()
        if last_active_instance is None:
            return None
        min_open_time = None
        first_pending_instance = None
        for inst in self.instances.values():
            if inst.pending and inst.open_time > last_active_instance.open_time:
                if min_open_time is None or inst.open_time < min_open_time:
                    min_open_time = inst.open_time
                    first_pending_instance = inst
        return first_pending_instance

    def get_pending_instance(self, instance: ProgressBarT) -> tp.Optional[ProgressBarT]:
        """Get the pending instance.

        If the bar id is not None, searches for the same id in the dictionary."""
        if instance.bar_id is not None:
            for inst in self.instances.values():
                if inst is not instance and inst.pending:
                    if inst.bar_id == instance.bar_id:
                        return inst
            return None
        last_active_instance = self.get_last_active_instance()
        if last_active_instance is None:
            return None
        min_open_time = None
        first_pending_instance = None
        for inst in self.instances.values():
            if inst.pending and inst.open_time > last_active_instance.open_time:
                if min_open_time is None or inst.open_time < min_open_time:
                    min_open_time = inst.open_time
                    first_pending_instance = inst
        return first_pending_instance

    def get_parent_instances(self, instance: ProgressBarT) -> tp.List[ProgressBarT]:
        """Get the (active) parent instances of an instance."""
        parent_instances = []
        for inst in self.instances.values():
            if inst is not instance and inst.active:
                if inst.open_time < instance.open_time:
                    parent_instances.append(inst)
        return parent_instances

    def get_parent_instance(self, instance: ProgressBarT) -> tp.Optional[ProgressBarT]:
        """Get the (active) parent instance of an instance."""
        max_open_time = None
        parent_instance = None
        for inst in self.get_parent_instances(instance):
            if max_open_time is None or inst.open_time > max_open_time:
                max_open_time = inst.open_time
                parent_instance = inst
        return parent_instance

    def get_child_instances(self, instance: ProgressBarT) -> tp.List[ProgressBarT]:
        """Get child (active or pending) instances of an instance."""
        child_instances = []
        for inst in self.instances.values():
            if inst is not instance and (inst.active or inst.pending):
                if inst.open_time > instance.open_time:
                    child_instances.append(inst)
        return child_instances

    def clear_instances(self) -> None:
        """Clear instances."""
        self.instances.clear()


pbar_reg = PBarRegistry()
"""Default registry of type `PBarRegistry`."""
