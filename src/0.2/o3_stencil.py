import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from math import sin, cos, pi

# ---------- Stroke parameter dataclasses ----------
@dataclass
class BarStrokeConfig:
    thickness: float = 0.4      # fraction of cell size
    length: float = 0.9         # fraction of cell size
    center: tuple = (0.5, 0.5)  # normalized (x, y)

@dataclass
class ArcStrokeConfig:
    radius: float = 0.4         # fraction of cell size
    start_angle: float = 0      # radians
    end_angle: float = pi       # radians
    thickness: float = 0.15     # fraction of cell size
    center: tuple = (0.5, 0.5)

@dataclass
class SCurveStrokeConfig:
    amplitude: float = 0.25     # fraction of cell size
    cycles: float = 1.0
    thickness: float = 0.15

@dataclass
class DiagonalStrokeConfig:
    angle: float = pi / 4       # slope direction
    thickness: float = 0.15

@dataclass
class WaveStrokeConfig:
    amplitude: float = 0.20
    cycles: float = 3.0
    thickness: float = 0.12

@dataclass
class CircleStrokeConfig:
    radius: float = 0.38
    thickness: float = 0.14
    center: tuple = (0.5, 0.5)

# ---------- Drawing helpers ----------
def get_cell(canvas, row, col, cell_size):
    r0, c0 = row * cell_size, col * cell_size
    return canvas[r0:r0 + cell_size, c0:c0 + cell_size]

def draw_bar(cell, cfg: BarStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    cx, cy = np.array(cfg.center) * N
    half_thick = cfg.thickness * N / 2
    half_len = cfg.length * N / 2
    mask = (np.abs(xx - cx) <= half_thick) & (np.abs(yy - cy) <= half_len)
    cell[mask] = 0

def draw_arc(cell, cfg: ArcStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    cx, cy = np.array(cfg.center) * N
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    angle = (np.arctan2(dy, dx) + 2 * pi) % (2 * pi)
    mask_r = np.abs(r - cfg.radius * N) <= cfg.thickness * N / 2
    mask_a = (angle >= cfg.start_angle) & (angle <= cfg.end_angle)
    cell[mask_r & mask_a] = 0

def draw_scurve(cell, cfg: SCurveStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    x_norm = xx / N
    y_curve = 0.5 + cfg.amplitude * np.sin(2 * pi * cfg.cycles * (x_norm - 0.5))
    dist = np.abs(yy / N - y_curve)
    mask = dist <= cfg.thickness
    cell[mask] = 0

def draw_diagonal(cell, cfg: DiagonalStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    # line through center with angle
    cx = cy = N / 2
    # Rotate coords
    x_rot = (xx - cx) * cos(cfg.angle) + (yy - cy) * sin(cfg.angle)
    mask = np.abs(x_rot) <= cfg.thickness * N
    cell[mask] = 0

def draw_wave(cell, cfg: WaveStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    x_norm = xx / N
    y_curve = 0.5 + cfg.amplitude * np.sin(2 * pi * cfg.cycles * x_norm)
    dist = np.abs(yy / N - y_curve)
    mask = dist <= cfg.thickness
    cell[mask] = 0

def draw_circle(cell, cfg: CircleStrokeConfig):
    N = cell.shape[0]
    yy, xx = np.indices(cell.shape)
    cx, cy = np.array(cfg.center) * N
    r = np.hypot(xx - cx, yy - cy)
    mask = np.abs(r - cfg.radius * N) <= cfg.thickness * N / 2
    cell[mask] = 0

# ---------- Canvas assembly ----------
def generate_strokes(cell_size=128):
    rows, cols = 2, 3
    canvas = np.ones((rows * cell_size, cols * cell_size))
    
    # Bar
    draw_bar(get_cell(canvas, 0, 0, cell_size), BarStrokeConfig())
    # Arc (half-circle)
    draw_arc(get_cell(canvas, 0, 1, cell_size), ArcStrokeConfig(start_angle=0, end_angle=pi))
    # S-curve
    draw_scurve(get_cell(canvas, 0, 2, cell_size), SCurveStrokeConfig())
    # Diagonal
    draw_diagonal(get_cell(canvas, 1, 0, cell_size), DiagonalStrokeConfig())
    # Wave
    draw_wave(get_cell(canvas, 1, 1, cell_size), WaveStrokeConfig())
    # Circle
    draw_circle(get_cell(canvas, 1, 2, cell_size), CircleStrokeConfig())
    
    return canvas

# ---------- Run & show ----------
canvas = generate_strokes()
plt.figure(figsize=(6, 4))
plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
plt.axis('off')
plt.title("Synthetic Brush Strokes")
plt.savefig("o3_stencil.png", bbox_inches='tight', pad_inches=0.1)
plt.show()
