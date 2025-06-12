# Code Style Guide

- **Imports**: Sort imports alphabetically within three groups, separated by a blank line:
  1. Standard library imports (e.g., `os`, `logging`)
  2. External module imports (e.g., `numpy`, `tyro`)
  3. Local module imports (e.g., `from ik import ik`)

- **Configuration**: Use `dataclasses`  and `jdc.pytree_dataclass` to define and store configuration parameters for scripts.

- **Arm Naming**: Use `_l` and `_r` suffixes for variables and parameters related to the left and right robot arms, respectively (e.g., `joint_pos_l`, `ik_target_r`).

- **Abstractions**: Utilize the high-level abstractions provided in `pattern.py` and `ik.py` where applicable.

- **Code Comments**: Do not add any additional comments to the code. Strive for self-documenting code through clear variable and function names.

- **Logging**: Use `log.info()` for important status updates and `log.debug()` for detailed debugging information. Use emojis to enhance log readability.

- **Float Precision**: When defining float values in configuration `dataclasses`, format them with exactly three decimal places (e.g., `0.123`).

- **TODOs**: Do not remove any TODOs in this file.

- Use emojis tastefully.