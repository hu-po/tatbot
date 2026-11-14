"""Server configuration; visitor credentials never enter this mapping."""
from __future__ import annotations

import os
import shlex
from pathlib import Path

from contracts import RequestError

KEY_VARIABLES = {"fal": "FAL_KEY", "gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY"}
PROVIDER_NAMES = ("inkgen", *KEY_VARIABLES)


def _dotenv(path: Path) -> dict[str, str]:
    """Read literal assignments only: no shell evaluation or interpolation."""
    values = {}
    if not path.is_file():
        return values
    for number, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#"):
            continue
        name, separator, raw = line.partition("=")
        name = name.strip()
        if not separator or not (name.startswith("INKGEN_") or name in KEY_VARIABLES.values()):
            continue
        try:
            parts = shlex.split(raw, comments=True)
        except ValueError:
            raise RequestError(f"invalid generator configuration on line {number}") from None
        if len(parts) > 1:
            raise RequestError(f"quote generator configuration values on line {number}")
        values[name] = parts[0] if parts else ""
    return values


def server_environment(app_file: str, env: dict[str, str] | None = None) -> dict[str, str]:
    """Space Secrets or process environment win over a local private .env."""
    environment = dict(os.environ if env is None else env)
    if environment.get("SPACE_ID"):
        return environment
    directory = Path(app_file).resolve().parent
    root = directory.parent.parent
    values = _dotenv(root / ".env") if (root / "scripts/tatbot").is_file() else {}
    values.update(_dotenv(directory / ".env"))
    return {**values, **environment}


def default_provider(env: dict[str, str]) -> str:
    """An operator can choose a local default; hosted visitors start with Inkgen."""
    provider = "inkgen" if env.get("SPACE_ID") else env.get("INKGEN_PROVIDER", "inkgen").strip()
    if provider not in PROVIDER_NAMES:
        raise RequestError("INKGEN_PROVIDER must be inkgen, fal, gemini or openai")
    return provider


def provider_key(provider: str, supplied: str | None, *, public: bool, env: dict[str, str]) -> str:
    """A hosted request can never borrow an operator's paid-provider key."""
    variable = KEY_VARIABLES[provider]
    key = supplied or ("" if public else env.get(variable, ""))
    if not isinstance(key, str):
        raise RequestError("enter an API key for the selected provider")
    key = key.strip()
    if not key:
        raise RequestError(f"enter your {provider} API key" if public else f"enter an API key or set {variable}")
    if len(key) > 512 or any(character.isspace() for character in key):
        raise RequestError("paste only the API key, without an environment-variable assignment")
    return key
