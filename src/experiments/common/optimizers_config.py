"""Parse `optimizers` from room experiment input config JSON."""
import json
from pathlib import Path
from typing import Any

from optimizers.defaults import OPTIMIZER_SHORTHAND_TO_CLASS


def raw_optimizers_value(config_path: Path | str) -> Any:
    """Return the raw top-level `optimizers` value from a config JSON file."""
    path = Path(config_path)
    payload = json.loads(path.read_text())
    return payload.get("optimizers", "adam")


def parse_optimizers(raw: Any, *, path: Path | str) -> list[str]:
    """Parse top-level `optimizers` from an input config JSON value."""
    path = Path(path)
    if raw == "all":
        return sorted(OPTIMIZER_SHORTHAND_TO_CLASS)
    if isinstance(raw, str):
        if raw not in OPTIMIZER_SHORTHAND_TO_CLASS:
            raise ValueError(
                f"Unknown optimizer {raw!r} in {path}; "
                f"expected one of {sorted(OPTIMIZER_SHORTHAND_TO_CLASS)} or 'all'"
            )
        return [raw]
    if isinstance(raw, list):
        if not raw:
            raise ValueError(f"'optimizers' must not be an empty list: {path}")
        optimizers: list[str] = []
        for item in raw:
            if not isinstance(item, str) or item not in OPTIMIZER_SHORTHAND_TO_CLASS:
                raise ValueError(
                    f"Each entry in 'optimizers' must be one of "
                    f"{sorted(OPTIMIZER_SHORTHAND_TO_CLASS)}; got {item!r} in {path}"
                )
            optimizers.append(item)
        return optimizers
    raise ValueError(
        f"'optimizers' must be 'all', a single optimizer name, or a list of names: {path}"
    )


def load_optimizers_from_config_json(config_json: dict[str, Any], *, path: Path | str) -> list[str]:
    """Read and validate `optimizers`; reject deprecated `optimizer` key."""
    path = Path(path)
    if "optimizer" in config_json:
        raise ValueError(
            f"Deprecated 'optimizer' key in {path}; use 'optimizers' "
            f"(e.g. \"all\", \"adam\", or [\"adam\", \"sgd\"])"
        )
    raw = config_json.get("optimizers", "adam")
    return parse_optimizers(raw, path=path)
