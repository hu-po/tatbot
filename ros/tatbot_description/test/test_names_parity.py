"""tatbot_hardware's names.hpp and names.py name the same GPIO interfaces and codes."""
import re
from pathlib import Path

from tatbot_description import names

HPP = Path(__file__).resolve().parents[2] / "tatbot_hardware" / "include" / "tatbot_hardware" / "names.hpp"


def _array(text, name):
    body = re.search(name + r"\s*=\s*\{(.*?)\};", text, re.S).group(1)
    return tuple(re.findall(r'"([^"]+)"', body))


def test_gpio_names_match():
    text = HPP.read_text()
    assert _array(text, "kSafetyState") == names.SAFETY_STATE_INTERFACES
    assert _array(text, "kSafetyCommand") == names.SAFETY_COMMAND_INTERFACES
    for prefix, enum in (("LATCH", "kLatch"), ("GUARD", "kGuard")):
        codes = dict(re.findall(enum + r"(\w+) = (\d+)", text))
        assert codes and {f"{prefix}_{re.sub(r'(?<!^)(?=[A-Z])', '_', k).upper()}": int(v) for k, v in codes.items()} == {
            k: v for k, v in vars(names).items() if k.startswith(prefix + "_") and isinstance(v, int)}
