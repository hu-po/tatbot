"""Local configuration and hosted credential isolation, with dummy keys only."""
from pathlib import Path

import pytest
from configuration import default_provider, provider_key, server_environment
from contracts import RequestError


def installation(tmp_path: Path) -> Path:
    app = tmp_path / "web/inkgen/app.py"
    app.parent.mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/tatbot").touch()
    return app


def test_local_env_is_literal_and_process_environment_wins(tmp_path):
    app = installation(tmp_path)
    (tmp_path / ".env").write_text('export INKGEN_PROVIDER=fal\nFAL_KEY="literal-$(touch nope)"\nHF_TOKEN=ignored\n')
    (app.parent / ".env").write_text('FAL_KEY=beside-app\nOPENAI_API_KEY="local-value" # comment\n')
    environment = {"FAL_KEY": "process-value"}
    result = server_environment(str(app), environment)
    assert result["INKGEN_PROVIDER"] == "fal"
    assert result["FAL_KEY"] == "process-value"
    assert result["OPENAI_API_KEY"] == "local-value"
    assert "HF_TOKEN" not in result
    assert environment == {"FAL_KEY": "process-value"}, "configuration must not mutate shared environment"
    assert not (tmp_path / "nope").exists()


def test_space_ignores_dotenv_and_starts_with_inkgen(tmp_path):
    app = installation(tmp_path)
    (app.parent / ".env").write_text("FAL_KEY=file-value\n")
    env = {"SPACE_ID": "example/inkgen", "INKGEN_PROVIDER": "fal", "FAL_KEY": "operator-value"}
    assert server_environment(str(app), env) == env
    assert default_provider(env) == "inkgen"
    with pytest.raises(RequestError, match="enter your fal API key"):
        provider_key("fal", "", public=True, env=env)


def test_only_local_requests_can_use_an_operator_key():
    env = {"FAL_KEY": "operator-value"}
    assert provider_key("fal", "", public=False, env=env) == "operator-value"
    assert provider_key("fal", "visitor-value", public=True, env=env) == "visitor-value"
    assert env == {"FAL_KEY": "operator-value"}


@pytest.mark.parametrize("key", ["bad\nkey", "env key=value", "x" * 513])
def test_invalid_keys_are_never_echoed(key):
    with pytest.raises(RequestError) as error:
        provider_key("fal", key, public=True, env={})
    assert key not in str(error.value)


def test_invalid_configuration_does_not_print_its_value(tmp_path):
    app = installation(tmp_path)
    (app.parent / ".env").write_text('FAL_KEY="dummy-secret-without-closing-quote\n')
    with pytest.raises(RequestError) as error:
        server_environment(str(app), {})
    assert "dummy-secret" not in str(error.value)


def test_provider_default_does_not_follow_key_presence():
    assert default_provider({"FAL_KEY": "operator-value"}) == "inkgen"
    assert default_provider({"INKGEN_PROVIDER": "openai"}) == "openai"
    with pytest.raises(RequestError):
        default_provider({"INKGEN_PROVIDER": "unknown"})
