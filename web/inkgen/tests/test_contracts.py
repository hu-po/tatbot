"""The request contract, with no model, no CUDA and no Gradio in sight.

If any of these ever needs a GPU to pass, the separation the whole batch worker
depends on has been lost.
"""
import subprocess
import sys
from pathlib import Path

import pytest
from contracts import (  # noqa: E402
    DEFAULT_MODEL,
    SEED_MAX,
    GenerationRequest,
    GenerationSettings,
    RequestError,
    canonical_digest,
    normalize_request,
    normalize_seed,
    result_document,
    tattoo_prompt,
)


def test_import_pulls_in_no_model_stack():
    """A schema read or a --help must not import torch, diffusers or gradio."""
    code = ("import sys, contracts, engine; "
            "print(sorted(m for m in ('torch', 'diffusers', 'gradio', 'fastapi') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_engine_identity_needs_no_weights():
    """`identity()` answers before anything is loaded, so health is cheap."""
    from engine import Engine

    engine = Engine(GenerationSettings(model="x/y", model_revision="a" * 40), device="cpu")
    identity = engine.identity()
    assert identity["loaded"] is False
    assert identity["model_revision"] == "a" * 40
    assert identity["settings"]["steps"] == 8


def test_seed_zero_is_a_seed():
    request = normalize_request({"subject": "a swallow", "seed": 0})
    assert request.seed == 0
    assert request.seed_requested is True


def test_absent_seed_is_chosen_once_and_reported():
    request = normalize_request({"subject": "a swallow"})
    assert 0 <= request.seed <= SEED_MAX
    assert request.seed_requested is False
    assert result_document(request, png_sha256="0" * 64, seconds=1, device="cpu")["seed"] == request.seed


@pytest.mark.parametrize("value", [-1, SEED_MAX + 1, "abc", 1.5, True, [], {}])
def test_unusable_seeds_are_refused(value):
    with pytest.raises(RequestError):
        normalize_seed(value)


def test_blank_seed_string_means_choose_one():
    assert 0 <= normalize_seed("") <= SEED_MAX
    assert normalize_request({"subject": "x", "seed": ""}).seed_requested is False


@pytest.mark.parametrize("payload", [
    {"subject": ""},
    {"subject": "x" * 121},
    {"subject": "x", "style": "s" * 161},
    {"subject": "x", "steps": 0},
    {"subject": "x", "width": 64},
    {"subject": "x", "model_revision": "main"},
    {"subject": "x", "seeed": 1},
])
def test_unusable_requests_are_refused(payload):
    with pytest.raises(RequestError):
        normalize_request(payload)


def test_whitespace_normalization_matches_the_prompt_builder():
    request = normalize_request({"subject": "  a   swallow \n", "style": " bold  lines ", "seed": 1})
    assert request.subject == "a swallow"
    assert request.style == "bold lines"
    assert request.prompt == tattoo_prompt("a swallow", "bold lines")


def test_identical_intents_share_one_digest_and_different_settings_do_not():
    a = normalize_request({"subject": "a swallow", "seed": 7})
    b = normalize_request({"subject": " a  swallow", "seed": 7})
    assert a.digest == b.digest
    assert a.digest != normalize_request({"subject": "a swallow", "seed": 8}).digest
    assert a.digest != normalize_request({"subject": "a swallow", "seed": 7, "steps": 9}).digest


def test_digest_binds_the_resolved_revision():
    """Two runs of the same words on different weights are different work."""
    request = normalize_request({"subject": "a swallow", "seed": 0})
    assert request.pinned("b" * 40).digest != request.digest
    assert request.pinned("b" * 40).digest != request.pinned("c" * 40).digest


def test_digest_ignores_how_the_seed_was_obtained():
    """A recorded random seed and the same seed typed by hand are one request."""
    chosen = normalize_request({"subject": "a swallow"})
    typed = normalize_request({"subject": "a swallow", "seed": chosen.seed})
    assert chosen.digest == typed.digest


def test_server_defaults_fill_unstated_settings():
    defaults = GenerationSettings(model="local/model", steps=4, width=512, height=512)
    request = normalize_request({"subject": "x", "seed": 1}, defaults=defaults)
    assert request.settings.model == "local/model"
    assert request.settings.steps == 4
    assert normalize_request({"subject": "x", "seed": 1, "steps": 12}, defaults=defaults).settings.steps == 12


def test_settings_from_env_refuse_a_moving_revision():
    with pytest.raises(RequestError):
        GenerationSettings.from_env({"INKGEN_MODEL_REVISION": "refs/pr/3"})
    assert GenerationSettings.from_env({}).model == DEFAULT_MODEL


def test_result_document_repeats_what_was_used():
    request = GenerationRequest(subject="a swallow", seed=0,
                                settings=GenerationSettings(model="m", model_revision="d" * 40, steps=3))
    document = result_document(request, png_sha256="a" * 64, seconds=1.2345, device="cuda")
    assert document["seed"] == 0
    assert document["model"] == "m"
    assert document["model_revision"] == "d" * 40
    assert document["settings"]["steps"] == 3
    assert document["request_sha256"] == request.digest
    assert document["seconds"] == 1.234 or document["seconds"] == 1.235


def test_canonical_digest_is_key_order_independent():
    assert canonical_digest({"a": 1, "b": 2}) == canonical_digest({"b": 2, "a": 1})


def test_public_visitor_quotas_are_a_property_of_public_serving():
    """A 24-image batch must not die at image seven on a six-per-minute cap.

    The quota exists to protect the owner's shared ZeroGPU allowance from
    strangers. A private worker a batch was pointed at owns its own job limits.
    The policy lives in the stdlib `serving` module, so it is read from there
    the way app.py reads it, without gradio, fastapi or the model.
    """
    from serving import AdmissionPolicy

    for public, expected in ((True, 6), (False, 0)):
        assert AdmissionPolicy.from_env(public=public, env={}).per_ip_per_min == expected
        # An explicit number always wins, on the Hub or off it.
        assert AdmissionPolicy.from_env(public=public, env={"INKGEN_PER_IP_PER_MIN": "3"}).per_ip_per_min == 3
