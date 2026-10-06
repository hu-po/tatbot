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
    codes = dict(re.findall(r"kLatch(\w+) = (\d+)", text))
    assert int(codes["StepRefused"]) == names.LATCH_STEP_REFUSED
    assert int(codes["Estop"]) == names.LATCH_ESTOP
    assert len(codes) == 11
