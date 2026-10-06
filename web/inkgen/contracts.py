"""What a generation request is, independent of who serves it.

Stdlib only, and deliberately model-free: importing this module must not pull
in torch, diffusers, gradio or CUDA, so `--help`, a schema dump and every
bookkeeping test run on a laptop with no GPU. The Space adapter, the local
engine and the batch worker all normalize through here, which is what makes
"the same request" mean the same thing in all three.

A request is complete: the subject and style a person typed, plus every setting
that changes the bytes the model emits (model id, pinned revision, size, steps,
guidance, seed). A reply repeats the settings that were actually used, so a
caller never has to assume its defaults were the server's.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, replace

PRODUCER = "tatbot-inkgen"
PRODUCER_VERSION = "2"
REQUEST_SCHEMA = "tatbot.inkgen-request/1"
RESULT_SCHEMA = "tatbot.inkgen-result/1"

DEFAULT_MODEL = "Tongyi-MAI/Z-Image-Turbo"
DEFAULT_STEPS = 8
DEFAULT_SIZE = 768
DEFAULT_GUIDANCE = 0.0
DEFAULT_LOOK = "clean black linework suitable for vector tracing"

SUBJECT_MAX = 120
STYLE_MAX = 160
# torch seeds a generator from a 64-bit value; 32 bits is what the browser and
# every recorded sidecar have ever carried, and it keeps a seed readable.
SEED_MAX = 2**32 - 1
SIZE_MIN, SIZE_MAX = 128, 2048
STEPS_MIN, STEPS_MAX = 1, 100
# A resolved revision is a git object id on the Hub; a branch or tag name is a
# moving reference and is never accepted as one.
REVISION_RE = re.compile(r"[0-9a-f]{40}")


class RequestError(ValueError):
    """The caller's request was not usable, with the reason it can act on."""


def tattoo_prompt(subject: str, style: str | None = None) -> str:
    """`style` replaces the default look."""
    s = " ".join(subject.split())
    look = " ".join((style or "").split())[:STYLE_MAX] or DEFAULT_LOOK
    return (f"tattoo flash design of {s}, {look}, "
            "isolated on a plain white background, centered, no text")


def random_seed() -> int:
    """A seed for a caller who named none. Chosen once, then returned explicitly."""
    return int.from_bytes(os.urandom(4), "big") % 1_000_000


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Everything except the words that changes the image the model emits."""

    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    width: int = DEFAULT_SIZE
    height: int = DEFAULT_SIZE
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> GenerationSettings:
        env = os.environ if env is None else env
        size = _int(env.get("INKGEN_SIZE"), DEFAULT_SIZE, "INKGEN_SIZE", SIZE_MIN, SIZE_MAX)
        return cls(
            model=(env.get("INKGEN_MODEL") or DEFAULT_MODEL).strip(),
            model_revision=_revision(env.get("INKGEN_MODEL_REVISION")),
            width=size, height=size,
            steps=_int(env.get("INKGEN_STEPS"), DEFAULT_STEPS, "INKGEN_STEPS", STEPS_MIN, STEPS_MAX),
            guidance=_float(env.get("INKGEN_GUIDANCE"), DEFAULT_GUIDANCE, "INKGEN_GUIDANCE"),
        )

    def pinned(self, revision: str | None) -> GenerationSettings:
        """The same settings with an immutable revision resolved into them."""
        return replace(self, model_revision=_revision(revision))

    def as_json(self) -> dict:
        return asdict(self)


def _int(raw: object, default: int, name: str, low: int, high: int) -> int:
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be an integer, got {raw!r}") from exc
    if not low <= value <= high:
        raise RequestError(f"{name} must be {low}-{high}, got {value}")
    return value


def _float(raw: object, default: float, name: str) -> float:
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{name} must be a number, got {raw!r}") from exc
    if not 0.0 <= value <= 50.0:
        raise RequestError(f"{name} must be 0-50, got {value}")
    return value


def _revision(raw: object) -> str | None:
    if raw is None or raw == "":
        return None
    text = str(raw).strip()
    if not REVISION_RE.fullmatch(text):
        raise RequestError(
            f"a pinned model revision must be a 40-character commit id, got {text!r}; "
            "a branch or tag moves and cannot identify what produced an image")
    return text


def normalize_seed(raw: object) -> int:
    """Seed zero is a seed. `None` and only `None` means 'choose one for me'."""
    if raw is None:
        return random_seed()
    if isinstance(raw, bool) or not isinstance(raw, (int, str, float)):
        raise RequestError(f"seed must be an integer 0-{SEED_MAX}, got {raw!r}")
    if isinstance(raw, str) and not raw.strip():
        return random_seed()
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"seed must be an integer 0-{SEED_MAX}, got {raw!r}") from exc
    if isinstance(raw, float) and value != raw:
        raise RequestError(f"seed must be a whole number, got {raw!r}")
    if not 0 <= value <= SEED_MAX:
        raise RequestError(f"seed must be 0-{SEED_MAX}, got {value}")
    return value


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One normalized request. Two equal requests name the same intended image."""

    subject: str
    seed: int
    style: str | None = None
    settings: GenerationSettings = GenerationSettings()
    seed_requested: bool = True

    @property
    def prompt(self) -> str:
        return tattoo_prompt(self.subject, self.style)

    def pinned(self, revision: str | None) -> GenerationRequest:
        return replace(self, settings=self.settings.pinned(revision))

    def canonical(self) -> dict:
        """The content that identifies this request; nothing incidental in it.

        Not the seed's provenance and not the prompt, which is a pure function
        of subject and style: adding either would make two identical intents
        look like different work.
        """
        return {"schema": REQUEST_SCHEMA, "subject": self.subject, "style": self.style,
                "seed": self.seed, **self.settings.as_json()}

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical())

    def as_json(self) -> dict:
        return {**self.canonical(), "prompt": self.prompt, "seed_requested": self.seed_requested}


def canonical_digest(document: object) -> str:
    """One spelling of a document, so a digest means the same thing everywhere."""
    return hashlib.sha256(canonical_json(document).encode()).hexdigest()


def canonical_json(document: object) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize_request(payload: dict, *, defaults: GenerationSettings | None = None) -> GenerationRequest:
    """A caller's JSON to a normalized request, refusing what cannot be drawn.

    Unstated settings come from `defaults` (the server's own), never from the
    caller's silence being read as agreement: the reply repeats what was used.
    """
    if not isinstance(payload, dict):
        raise RequestError("request must be a JSON object")
    base = defaults or GenerationSettings()
    subject = " ".join(str(payload.get("subject", "")).split())
    if not 1 <= len(subject) <= SUBJECT_MAX:
        raise RequestError(f"subject must be 1-{SUBJECT_MAX} characters")
    style_raw = payload.get("style")
    style = " ".join(str(style_raw).split()) if style_raw is not None else ""
    if len(style) > STYLE_MAX:
        raise RequestError(f"style must be at most {STYLE_MAX} characters")
    unknown = set(payload) - {"subject", "style", "seed", "turnstile", "model", "model_revision",
                              "width", "height", "steps", "guidance"}
    if unknown:
        raise RequestError(f"unknown request field(s): {', '.join(sorted(unknown))}")
    model = str(payload.get("model") or base.model).strip()
    if not model:
        raise RequestError("model must not be empty")
    size = _int(payload.get("width"), base.width, "width", SIZE_MIN, SIZE_MAX), \
        _int(payload.get("height"), base.height, "height", SIZE_MIN, SIZE_MAX)
    settings = GenerationSettings(
        model=model,
        model_revision=_revision(payload.get("model_revision") if payload.get("model_revision")
                                 else base.model_revision),
        width=size[0], height=size[1],
        steps=_int(payload.get("steps"), base.steps, "steps", STEPS_MIN, STEPS_MAX),
        guidance=_float(payload.get("guidance"), base.guidance, "guidance"),
    )
    raw_seed = payload.get("seed")
    return GenerationRequest(subject=subject, seed=normalize_seed(raw_seed), style=style or None,
                             settings=settings, seed_requested=raw_seed is not None
                             and not (isinstance(raw_seed, str) and not raw_seed.strip()))


def result_document(request: GenerationRequest, *, png_sha256: str, seconds: float,
                    device: str) -> dict:
    """The metadata half of a reply: what actually produced these bytes.

    `model` stays a bare id at the top level because every existing client reads
    it there; the resolved revision and the rest live beside it, additively.
    """
    return {"schema": RESULT_SCHEMA, "seed": request.seed, "prompt": request.prompt,
            "model": request.settings.model, "model_revision": request.settings.model_revision,
            "settings": request.settings.as_json(), "request_sha256": request.digest,
            "png_sha256": png_sha256, "seconds": round(float(seconds), 3), "device": device,
            "producer": PRODUCER, "producer_version": PRODUCER_VERSION,
            "seed_requested": request.seed_requested}
