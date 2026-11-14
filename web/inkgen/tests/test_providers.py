"""Exercise provider wire formats, result bytes and error redaction without paid calls."""
import base64
import io
import json
import urllib.error

import provider_http
import providers
import pytest
from contracts import RequestError
from fake_engine import png_bytes

PNG = png_bytes(16, 16, lambda x, y: x == y)
ENCODED = base64.b64encode(PNG).decode()
KEY = "dummy-visitor-credential"


def encoded(document):
    return json.dumps(document).encode()


@pytest.mark.parametrize("provider", ["gemini", "openai"])
def test_cloud_request_has_no_invented_seed_or_diffusion_settings(provider):
    request = providers.normalize_remote({"provider": provider, "subject": "a moth"})
    assert request.seed is None and not request.seed_requested
    assert "steps" not in request.parameters() and "guidance" not in request.parameters()
    with pytest.raises(RequestError, match="does not support"):
        providers.normalize_remote({"provider": provider, "subject": "a moth", "seed": 0})


def test_flux_pro_uses_its_own_supported_parameter_set():
    request = providers.normalize_remote({"provider": "fal", "model": "fal-ai/flux-2-pro", "subject": "a moth"})
    assert "num_images" not in request.parameters()
    assert request.parameters()["output_format"] == "png"


@pytest.mark.parametrize("extra", [{"model": "unreviewed/model"}, {"api_key": KEY}, {"guidance": 9}])
def test_unknown_models_and_unsupported_settings_are_refused(extra):
    with pytest.raises(RequestError) as error:
        providers.normalize_remote({"provider": "fal", "subject": "a moth", **extra})
    assert KEY not in str(error.value)


def test_openai_wire_format_original_bytes_and_metadata():
    pytest.importorskip("PIL")
    calls = []

    def send(method, url, headers, payload=None, **kwargs):
        calls.append((method, url, headers, payload))
        return encoded({"data": [{"b64_json": ENCODED}], "usage": {"input_tokens": 4, "output_tokens": 8, "extra": KEY}})

    request = providers.normalize_remote({"provider": "openai", "subject": "a moth"})
    result = providers.generate(request, KEY, send=send)
    method, url, headers, payload = calls[0]
    assert method == "POST" and url == "https://api.openai.com/v1/images/generations"
    assert headers == {"Authorization": f"Bearer {KEY}"}
    assert payload["model"] == "gpt-image-2.5-flare" and payload["n"] == 1
    assert payload["quality"] == "medium" and "seed" not in payload
    assert result.png == result.original == PNG
    assert result.metadata["seed"] is None and result.metadata["model_revision"] is None
    assert result.metadata["schema"] == "tatbot.inkgen-result/2"
    assert result.metadata["usage"] == {"input_tokens": 4, "output_tokens": 8}
    assert KEY not in json.dumps(result.metadata)
    assert result.metadata["settings"]["width"] == 16


def test_gemini_interactions_wire_format_and_image_extraction():
    pytest.importorskip("PIL")

    def send(method, url, headers, payload=None, **kwargs):
        assert method == "POST" and url.endswith("/v1beta/interactions")
        assert headers == {"x-goog-api-key": KEY, "Api-Revision": "2026-05-20"}
        assert payload["store"] is False
        assert payload["response_format"]["mime_type"] == "image/jpeg"
        assert payload["input"][0]["type"] == "text"
        return encoded({"steps": [{"type": "model_output", "content": [
            {"type": "text", "text": "Image"}, {"type": "image", "mime_type": "image/png", "data": ENCODED}]}]})

    result = providers.generate(providers.normalize_remote({"provider": "gemini", "subject": "a rose"}), KEY, send=send)
    assert result.png == PNG and result.metadata["model"] == "gemini-3.1-flash-image"


def test_fal_queue_and_download_never_forward_credentials_to_image_host(monkeypatch):
    pytest.importorskip("PIL")
    calls = []
    monkeypatch.setattr(providers.time, "sleep", lambda seconds: None)

    def send(method, url, headers, payload=None, **kwargs):
        calls.append((method, url, headers, payload))
        if method == "POST":
            assert payload["seed"] == 0 and payload["num_images"] == 1
            return encoded({"status_url": "https://queue.fal.run/requests/one/status",
                            "response_url": "https://queue.fal.run/requests/one"})
        if url.endswith("/status"):
            return encoded({"status": "IN_QUEUE" if len(calls) == 2 else "COMPLETED"})
        if "queue.fal.run" in url:
            return encoded({"images": [{"url": "https://v3.fal.media/output.png"}], "seed": 0})
        assert headers == {}
        return PNG

    result = providers.generate(providers.normalize_remote({"provider": "fal", "subject": "a rose", "seed": 0}), KEY, send=send)
    assert len(calls) == 5 and result.metadata["seed"] == 0
    assert result.metadata["seed_requested"] is True
    assert KEY not in json.dumps(result.metadata)


@pytest.mark.parametrize("url", ["http://queue.fal.run/a", "https://attacker.test/a", "https://queue.fal.run:8443/a",
                                  "https://key@queue.fal.run/a", "https://queue.fal.run.attacker.test/a"])
def test_provider_urls_are_restricted(url):
    with pytest.raises(provider_http.ProviderError):
        provider_http.checked_url(url, ("queue.fal.run",))


@pytest.mark.parametrize("status", [401, 402, 429])
def test_upstream_errors_are_sanitized_and_paid_posts_are_not_retried(monkeypatch, status):
    calls = []

    class Opener:
        def open(self, request, timeout):
            calls.append(request)
            raise urllib.error.HTTPError(request.full_url, status, KEY, {}, io.BytesIO(KEY.encode()))

    monkeypatch.setattr(provider_http.urllib.request, "build_opener", lambda *args: Opener())
    with pytest.raises(provider_http.ProviderError) as error:
        provider_http.request("POST", "https://api.openai.com/v1/images/generations", {"Authorization": KEY}, {})
    assert len(calls) == 1 and error.value.status == status
    assert KEY not in str(error.value)
    if status == 402:
        assert "credit balance" in str(error.value) and "payment status" in str(error.value)


def test_text_only_or_malformed_cloud_response_is_an_explicit_failure():
    request = providers.normalize_remote({"provider": "gemini", "subject": "a rose"})
    with pytest.raises(provider_http.ProviderError, match="no image"):
        providers.generate(request, KEY, send=lambda *args: encoded({"steps": []}))
    request = providers.normalize_remote({"provider": "openai", "subject": "a rose"})
    with pytest.raises(provider_http.ProviderError, match="incomplete"):
        providers.generate(request, KEY, send=lambda *args: encoded({"error": KEY}))


def test_original_jpeg_is_retained_when_png_preview_is_created():
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGB", (24, 16), "white").save(buffer, format="JPEG")
    original = buffer.getvalue()
    request = providers.normalize_remote({"provider": "openai", "subject": "a rose"})
    result = providers.generate(request, KEY, send=lambda *args: encoded({"data": [{"b64_json": base64.b64encode(original).decode()}]}))
    assert result.original == original and result.png.startswith(b"\x89PNG")
    assert result.metadata["source_media_type"] == "image/jpeg"
    assert result.metadata["source_sha256"] != result.metadata["png_sha256"]


def test_preview_preserves_transparent_negative_space():
    image = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    image.new("RGBA", (16, 16), (0, 0, 0, 0)).save(buffer, format="WEBP", lossless=True)
    png, _, _ = providers._image(buffer.getvalue())
    with image.open(io.BytesIO(png)) as decoded:
        assert decoded.mode == "RGBA" and decoded.getpixel((0, 0))[3] == 0
