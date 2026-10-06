"""Camera pipelines after the optics: the wrist D405's (blur, exposure, white balance, noise, YUYV) and
the scene camera's H.264 sub stream (soft, sharpened, compressed, 4:2:0 colour).

The real stream runs auto exposure (about 32 ms at indoor light), so motion
smears the image along the camera's own movement and the brightness chases a
target with a lag. It arrives as YUYV 4:2:2, which halves horizontal colour
resolution -- a detail a pristine render would give away.

Everything per-frame is uint8/OpenCV work on precomputed tables so the
sensor costs a few milliseconds, not the render's multiple.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NOISE_BANK = 4


@dataclass(frozen=True)
class CameraLook:
    """Per-episode sensor behaviour."""

    exposure_s: float
    target_mean: float  # auto-exposure set point, 0..255
    ae_rate: float  # fraction of the log-error corrected per frame
    wb: tuple[float, float, float]
    gamma: float
    read_noise: float
    shot_noise: float
    focus_blur: float
    vignette: float
    saturation: float
    contrast: float = 1.0  # about mid-grey, after gamma: the rig's frames clip the white wall and the silicone

    @classmethod
    def sample(cls, rng: np.random.Generator) -> CameraLook:
        return cls(
            exposure_s=float(rng.uniform(0.012, 0.034)),
            target_mean=float(rng.uniform(85, 145)),  # the rig's frames: mean V 89-110, grey 98-114
            ae_rate=float(rng.uniform(0.05, 0.3)),
            wb=tuple(float(g) for g in 1.0 + rng.normal(0, 0.06, 3)),
            gamma=float(rng.uniform(0.85, 1.15)),
            read_noise=float(rng.uniform(0.5, 4.0)),
            shot_noise=float(rng.uniform(0.0, 0.6)),
            focus_blur=float(rng.uniform(0.0, 1.1)),
            vignette=float(rng.uniform(0.0, 0.35)),
            saturation=float(rng.uniform(0.65, 1.05)),
            contrast=float(rng.uniform(1.05, 1.6)),  # the rig's wrist frames: grey std 83, 15 % of pixels over 235
        )


def _colour_matrix(wb: tuple[float, float, float], saturation: float) -> np.ndarray:
    """White balance then saturation about the grey axis, as one 3x3 transform."""
    grey = np.full((3, 3), 1.0 / 3.0)
    return (saturation * np.eye(3) + (1.0 - saturation) * grey) @ np.diag(wb)


class SensorState:
    """Carries auto-exposure gain across frames of one episode."""

    def __init__(self, look: CameraLook, rng: np.random.Generator, shape: tuple[int, int]):
        self.look, self.rng = look, rng
        self.gain, self.settled = 1.0, False  # the real stream starts with its exposure already settled
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        self.vignette = (1.0 - look.vignette * np.clip(r2 / 2.0, 0, 1)).astype(np.float32)[..., None]
        self.colour = _colour_matrix(look.wb, look.saturation).astype(np.float32)
        levels = np.arange(256, dtype=np.float32)
        curve = 0.5 + look.contrast * ((levels / 255.0) ** look.gamma - 0.5)
        self.gamma_lut = np.clip(255.0 * curve, 0, 255).astype(np.uint8)
        self.noise_sigma = np.clip(look.read_noise + look.shot_noise * np.sqrt(levels), 0, 40).astype(np.float32)
        self.noise_bank = rng.standard_normal((NOISE_BANK, h, w, 3), dtype=np.float32)

    def _motion_blur(self, image: np.ndarray, shift_px: np.ndarray) -> np.ndarray:
        length = float(np.linalg.norm(shift_px))
        if length < 0.75:
            return image
        n = int(min(np.ceil(length), 31)) | 1
        kernel = np.zeros((n, n), np.float32)
        c = n // 2
        d = shift_px / length * (n / 2)
        cv2.line(kernel, (int(round(c - d[0])), int(round(c - d[1]))), (int(round(c + d[0])), int(round(c + d[1]))),
                 1.0, 1, lineType=cv2.LINE_AA)
        kernel /= max(float(kernel.sum()), 1e-6)
        return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)

    def _auto_exposure(self, image: np.ndarray) -> None:
        if not self.settled:
            raw = float(cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA).mean())
            self.gain = float(np.clip(self.look.target_mean / max(raw, 1e-3), 0.25, 6.0))
            self.settled = True
            return
        mean = float(cv2.resize(image, (80, 60), interpolation=cv2.INTER_AREA).mean()) * self.gain
        if mean > 1e-3:
            error = np.log(self.look.target_mean / mean)
            self.gain = float(np.clip(self.gain * np.exp(self.look.ae_rate * error), 0.25, 6.0))

    def _noise(self, image: np.ndarray) -> np.ndarray:
        bank = self.noise_bank[int(self.rng.integers(NOISE_BANK))]
        dy, dx = (int(v) for v in self.rng.integers(0, 64, 2))
        noise = np.roll(bank, (dy, dx), axis=(0, 1)) * self.noise_sigma[image]
        return cv2.add(image.astype(np.float32), noise, dtype=cv2.CV_32F)

    def __call__(self, image: np.ndarray, shift_px: np.ndarray, overlay=None) -> np.ndarray:
        """``image`` uint8 RGB on the real grid; ``shift_px`` the image motion during exposure; ``overlay``
        (the cradle's real pixels) goes on after the blur: it is rigid to the camera, so it never smears."""
        look = self.look
        if look.focus_blur > 0.3:
            image = cv2.GaussianBlur(image, (0, 0), look.focus_blur)
        image = self._motion_blur(image, shift_px)
        if overlay is not None:
            image = overlay(image)
        self._auto_exposure(image)
        out = cv2.transform(image.astype(np.float32), self.colour * self.gain) * self.vignette
        out = cv2.LUT(np.clip(out, 0, 255).astype(np.uint8), self.gamma_lut)
        out = np.clip(self._noise(out), 0, 255).astype(np.uint8)
        return yuyv_roundtrip(out)


@dataclass(frozen=True)
class StreamLook:
    """Per-episode behaviour of the scene camera's sub stream: a fixed camera, so no motion smear, but a soft
    lens, the ISP's sharpening, and H.264 at a low bitrate (standing in: JPEG at a random quality)."""

    target_mean: float
    wb: tuple[float, float, float]
    gamma: float
    saturation: float
    contrast: float
    read_noise: float
    focus_blur: float
    sharpen: float
    quality: int
    vignette: float

    @classmethod
    def sample(cls, rng: np.random.Generator) -> StreamLook:
        # The rig's sub-stream frames: mean grey 86, contrast (grey std) 81, HSV saturation 55, crisp.
        return cls(
            target_mean=float(rng.uniform(70, 135)),
            wb=tuple(float(g) for g in 1.0 + rng.normal(0, 0.05, 3)),
            gamma=float(rng.uniform(0.85, 1.15)),
            saturation=float(rng.uniform(0.45, 0.95)),
            contrast=float(rng.uniform(1.0, 1.6)),
            read_noise=float(rng.uniform(0.3, 2.5)),
            focus_blur=float(rng.uniform(0.0, 0.9)),
            sharpen=float(rng.uniform(0.2, 1.2)),
            quality=int(rng.integers(18, 70)),
            vignette=float(rng.uniform(0.0, 0.3)),
        )


class StreamSensor:
    """The scene camera's frames: exposure settled on the first, then held (the room's light does not
    change within an episode, and nothing moves the camera)."""

    def __init__(self, look: StreamLook, rng: np.random.Generator, shape: tuple[int, int]):
        self.look, self.rng = look, rng
        self.gain: float | None = None
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
        self.vignette = (1.0 - look.vignette * np.clip(r2 / 2.0, 0, 1)).astype(np.float32)[..., None]
        self.colour = _colour_matrix(look.wb, look.saturation).astype(np.float32)
        levels = np.arange(256, dtype=np.float32) / 255.0
        curve = np.clip(0.5 + look.contrast * (levels ** look.gamma - 0.5), 0, 1)
        self.lut = (255.0 * curve).astype(np.uint8)
        self.noise_bank = rng.standard_normal((NOISE_BANK, h, w, 3), dtype=np.float32) * look.read_noise

    def __call__(self, image: np.ndarray) -> np.ndarray:
        look = self.look
        if look.focus_blur > 0.3:
            image = cv2.GaussianBlur(image, (0, 0), look.focus_blur)
        if self.gain is None:
            raw = float(cv2.resize(image, (88, 60), interpolation=cv2.INTER_AREA).mean())
            self.gain = float(np.clip(look.target_mean / max(raw, 1e-3), 0.25, 6.0))
        out = cv2.transform(image.astype(np.float32), self.colour * self.gain) * self.vignette
        out = cv2.LUT(np.clip(out, 0, 255).astype(np.uint8), self.lut)
        noise = self.noise_bank[int(self.rng.integers(NOISE_BANK))]
        out = np.clip(out.astype(np.float32) + noise, 0, 255)
        if look.sharpen > 0.05:
            out = cv2.addWeighted(out, 1.0 + look.sharpen, cv2.GaussianBlur(out, (0, 0), 1.2), -look.sharpen, 0)
        out = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        ok, jpeg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, look.quality,
                                              cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420])
        return cv2.cvtColor(cv2.imdecode(jpeg, cv2.IMREAD_COLOR) if ok else out, cv2.COLOR_BGR2RGB)


def yuyv_roundtrip(rgb: np.ndarray) -> np.ndarray:
    """Encode to YUYV 4:2:2 and back, as the D405 colour stream does: chroma shared by pixel pairs."""
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    chroma = yuv[:, 0::2, 1:].astype(np.uint16) + yuv[:, 1::2, 1:]
    yuv[:, 0::2, 1:] = yuv[:, 1::2, 1:] = (chroma // 2).astype(np.uint8)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
