"""Config file reader for Knapsack GNN project.

Rewritten from config_reader.py (original Neuro-Knapsack project).

Key changes vs original:
    - Handles empty lines and comment lines (# prefix) — original crashed.
    - Supports float values — original only parsed int and str.
    - Raises clear errors for malformed lines.
    - File is properly closed via context manager.
    - Added load_config() as the main public function.
    - Added get() with default for safe access.
    - Added save_config() to write config back to file.

Config file format:
    # comment lines are ignored
    KEY = value          # string
    KEY = 42             # int
    KEY = 3.14           # float
    KEY = [1, 2, 3]      # list of int/float/str
    KEY = true           # bool (true/false, case-insensitive)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Union


ConfigValue = Union[str, int, float, bool, List[Any]]


def _parse_value(raw: str) -> ConfigValue:
    """Parse a raw string value into the appropriate Python type."""
    raw = raw.strip()

    # List
    if raw.startswith("[") and raw.endswith("]"):
        inner  = raw[1:-1]
        items  = [item.strip() for item in inner.split(",")]
        return [_parse_scalar(item) for item in items if item]

    return _parse_scalar(raw)


def _parse_scalar(raw: str) -> Union[str, int, float, bool]:
    """Parse a single (non-list) value."""
    raw = raw.strip()

    # Bool
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False

    # Int
    try:
        return int(raw)
    except ValueError:
        pass

    # Float
    try:
        return float(raw)
    except ValueError:
        pass

    # String (strip surrounding quotes if present)
    if (raw.startswith('"') and raw.endswith('"')) or \
       (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]

    return raw


def load_config(filename: Union[str, Path]) -> Dict[str, ConfigValue]:
    """Load a key=value config file into a dict.

    Args:
        filename: Path to the config file.

    Returns:
        Dict mapping variable names to parsed values.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError:        If a line cannot be parsed as key=value.
    """
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    config: Dict[str, ConfigValue] = {}

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()

            # Skip blank lines and comments
            if not line or line.startswith("#"):
                continue

            # Remove inline comment
            if " #" in line:
                line = line[:line.index(" #")].rstrip()

            if "=" not in line:
                raise ValueError(
                    f"Config parse error at line {lineno}: "
                    f"expected 'KEY = VALUE', got: {line!r}"
                )

            key, _, raw_value = line.partition("=")
            key       = key.strip()
            raw_value = raw_value.strip()

            if not key:
                raise ValueError(f"Empty key at line {lineno}: {line!r}")

            config[key] = _parse_value(raw_value)

    return config


def get(
    config: Dict[str, ConfigValue],
    key: str,
    default: Any = None,
) -> Any:
    """Safe config lookup with default.

    Args:
        config:  Dict returned by load_config().
        key:     Config key to look up.
        default: Value to return if key is absent.
    """
    return config.get(key, default)


def save_config(
    config: Dict[str, ConfigValue],
    filename: Union[str, Path],
) -> None:
    """Write a config dict back to a key=value file.

    Args:
        config:   Dict to serialise.
        filename: Output path.
    """
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for key, value in config.items():
            if isinstance(value, list):
                formatted = "[" + ", ".join(str(v) for v in value) + "]"
            elif isinstance(value, bool):
                formatted = "true" if value else "false"
            else:
                formatted = str(value)
            f.write(f"{key} = {formatted}\n")


# ---------------------------------------------------------------------------
# Example config file content (for reference)
# ---------------------------------------------------------------------------
#
# configs/Config:
#
#   # Model settings
#   HIDDEN = 64
#   N = [5, 10, 20]
#   TYPE = gnn
#   RANGE = 1000.0
#   MEM_LAYERS = 2
#   DROPOUT = [0.1, 0.1, 0.0]
#   PROBLEM_TYPE = knapsack
#   PROBLEM_TYPE_TEST = knapsack
#   MAX_WAIT = 5
#   EPOCHS = 20
#   BATCH_SIZE = 8
#   LEARNING_RATE = 0.001
#   SEED = 2025