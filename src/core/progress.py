"""Nested tqdm progress bars for experiments."""

from tqdm.auto import tqdm

from core.experiment import ExpProtocol, Visit


def group_protocol(protocol: ExpProtocol) -> list[tuple[str, list[Visit]]]:
    """Group consecutive protocol visits by `group_tag` (epoch / rep / cycle)."""
    groups: list[tuple[str, list[Visit]]] = []
    current_tag: str | None = None
    current_visits: list[Visit] = []
    for visit in protocol:
        if visit.group_tag != current_tag:
            if current_visits:
                groups.append((current_tag, current_visits))
            current_tag = visit.group_tag
            current_visits = [visit]
        else:
            current_visits.append(visit)
    if current_visits:
        groups.append((current_tag, current_visits))
    return groups


class TrainingProgress:
    def __init__(
        self,
        level: int,
        *,
        experiment_name: str,
        group_unit: str,
        n_groups: int,
    ) -> None:
        if level not in range(4):
            raise ValueError(f"show_progress_level must be 0-3, got {level}")
        self.level = level
        self._parent: tqdm | None = None
        self._room: tqdm | None = None
        self._traj: tqdm | None = None
        self._seg: tqdm | None = None
        if level >= 1:
            self._parent = tqdm(
                total=n_groups,
                desc=f"{experiment_name} {group_unit}",
                unit=group_unit,
            )

    def begin_group(self, n_rooms: int) -> None:
        if self.level >= 2:
            self._room = tqdm(total=n_rooms, desc="room", unit="room", leave=False)

    def begin_room(self, n_traj: int) -> None:
        if self.level >= 2:
            self._traj = tqdm(total=n_traj, desc="traj", unit="traj", leave=False)

    def begin_trajectory(self, n_segments: int) -> tqdm | None:
        if self.level >= 3:
            self._seg = tqdm(total=n_segments, desc="seg", unit="seg", leave=False)
            return self._seg
        return None

    def end_trajectory(self) -> None:
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        if self._traj is not None:
            self._traj.update(1)

    def end_room(self) -> None:
        if self._traj is not None:
            self._traj.close()
            self._traj = None
        if self._room is not None:
            self._room.update(1)

    def end_group(self) -> None:
        if self._room is not None:
            self._room.close()
            self._room = None
        if self._parent is not None:
            self._parent.update(1)

    def close(self) -> None:
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        if self._traj is not None:
            self._traj.close()
            self._traj = None
        if self._room is not None:
            self._room.close()
            self._room = None
        if self._parent is not None:
            self._parent.close()
            self._parent = None
