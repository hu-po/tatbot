# Code Style Guide

- Sort imports alphabetically within three groups, separated by a blank line:
  1. Standard library imports (e.g., `os`, `logging`)
  2. External module imports (e.g., `numpy`, `tyro`)
  3. Local module imports (e.g., `from _log import get_logger`)
- Use `dataclasses`  and `jdc.pytree_dataclass` to define and store configuration parameters for scripts.
- Use `_l` and `_r` suffixes for variables and parameters related to the left and right robot arms, respectively (e.g., `joint_pos_l`, `ik_target_r`).
- Utilize the high-level abstractions provided in `pattern.py` and `ik.py` where applicable.
- Strive for self-documenting code through clear variable and function names that have short names and adhere to the clean reusable abstractions.
- Use `log.info()` for important status updates and `log.debug()` for detailed debugging information. Use emojis to enhance log readability.
- When defining float values in configuration `dataclasses`, use up to four decimal places, omitting unnecessary trailing zeros (e.g., use `0.2` not `0.2000`, and `0.04` not `0.040`). For whole numbers, use one decimal place (e.g., `50.0`).
- Do not remove any TODOs or comments
- Put all dependencies in `pyproject.toml`, use pinned versions
- ⚙️ Use emojis tastefully