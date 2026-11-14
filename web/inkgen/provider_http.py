"""Bounded provider HTTP requests; credentials and upstream errors stay private."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from urllib.parse import urlsplit

MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class ProviderError(RuntimeError):
    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.status = status


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def checked_url(url: str, hosts: tuple[str, ...], *, subdomains: bool = False) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    allowed = host in hosts or (subdomains and any(host.endswith("." + item) for item in hosts))
    if parsed.scheme != "https" or not allowed or parsed.username or parsed.password or parsed.port:
        raise ProviderError("provider returned an unsupported result URL")
    return url


def request(method: str, url: str, headers: dict[str, str], payload: dict | None = None,
            *, timeout: float = 180) -> bytes:
    body = json.dumps(payload).encode() if payload is not None else None
    outgoing = urllib.request.Request(url, body, {"Content-Type": "application/json", **headers}, method=method)
    try:
        with urllib.request.build_opener(_NoRedirect()).open(outgoing, timeout=timeout) as response:
            data = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise _http_error(exc.code) from None
    except (TimeoutError, OSError, urllib.error.URLError):
        raise ProviderError("provider connection failed; check its dashboard before retrying a paid request", 504) from None
    if len(data) > MAX_RESPONSE_BYTES:
        raise ProviderError("provider response exceeds the image size limit")
    return data


def _http_error(status: int) -> ProviderError:
    if status in (401, 403):
        return ProviderError("provider rejected this key or account access; check its API dashboard", 401)
    if status == 402:
        return ProviderError("provider requires a positive credit balance; check its billing dashboard and payment status", 402)
    if status == 429:
        return ProviderError("provider quota or rate limit reached; check its billing and limits", 429)
    return ProviderError("provider could not complete this request; check its dashboard before retrying")


def document(data: bytes) -> dict:
    try:
        result = json.loads(data)
    except (ValueError, UnicodeError):
        raise ProviderError("provider returned an unreadable response") from None
    if not isinstance(result, dict):
        raise ProviderError("provider returned an unexpected response")
    return result
