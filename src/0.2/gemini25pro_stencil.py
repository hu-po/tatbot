import math
from dataclasses import dataclass
from PIL import Image, ImageDraw

# --- Configuration Dataclasses for Each Stroke ---
# These classes hold the parameters for each shape, making them easy to configure.

@dataclass
class VerticalLineConfig:
    """Parameters for a vertical line."""
    length: int = 80
    thickness: int = 10

@dataclass
class HorizontalLineConfig:
    """Parameters for a horizontal line."""
    length: int = 80
    thickness: int = 10

@dataclass
class SCurveConfig:
    """Parameters for an 'S' shaped curve."""
    amplitude: int = 25  # How wide the curve swings
    frequency: float = 0.1 # How many bends in the curve
    length: int = 80
    thickness: int = 8

@dataclass
class ArcConfig:
    """Parameters for a 'C' shaped arc."""
    radius: int = 40
    start_angle_deg: int = -70 # Start angle in degrees
    end_angle_deg: int = 190   # End angle in degrees
    thickness: int = 10

@dataclass
class CircleConfig:
    """Parameters for a full circle."""
    radius: int = 40
    thickness: int = 10

@dataclass
class WaveConfig:
    """Parameters for a continuous wavy line."""
    amplitude: int = 15
    frequency: float = 0.15
    length: int = 80
    thickness: int = 6

# --- Drawing Functions ---
# Each function takes a configuration and draws the corresponding shape.

def draw_vertical_line(draw: ImageDraw.Draw, config: VerticalLineConfig, origin: tuple[int, int]):
    """Draws a vertical line on the canvas."""
    x0, y0 = origin
    # Draw a line from top to bottom
    draw.line([(x0, y0 - config.length // 2), (x0, y0 + config.length // 2)], fill="black", width=config.thickness)

def draw_horizontal_line(draw: ImageDraw.Draw, config: HorizontalLineConfig, origin: tuple[int, int]):
    """Draws a horizontal line on the canvas."""
    x0, y0 = origin
    # Draw a line from left to right
    draw.line([(x0 - config.length // 2, y0), (x0 + config.length // 2, y0)], fill="black", width=config.thickness)

def draw_s_curve(draw: ImageDraw.Draw, config: SCurveConfig, origin: tuple[int, int]):
    """Draws an S-curve using a sine wave along the y-axis."""
    x0, y0 = origin
    points = []
    # Iterate vertically and calculate the x-offset using sin
    for y_offset in range(-config.length // 2, config.length // 2):
        x_offset = config.amplitude * math.sin(y_offset * config.frequency)
        points.append((x0 + x_offset, y0 + y_offset))
    
    if points:
        draw.line(points, fill="black", width=config.thickness, joint="curve")

def draw_arc(draw: ImageDraw.Draw, config: ArcConfig, origin: tuple[int, int]):
    """Draws a C-shaped arc."""
    x0, y0 = origin
    # Define the bounding box for the arc
    top_left = (x0 - config.radius, y0 - config.radius)
    bottom_right = (x0 + config.radius, y0 + config.radius)
    bounding_box = [top_left, bottom_right]
    
    draw.arc(bounding_box, start=config.start_angle_deg, end=config.end_angle_deg, fill="black", width=config.thickness)

def draw_circle(draw: ImageDraw.Draw, config: CircleConfig, origin: tuple[int, int]):
    """Draws a circle."""
    x0, y0 = origin
    # Define the bounding box for the ellipse (a circle in this case)
    top_left = (x0 - config.radius, y0 - config.radius)
    bottom_right = (x0 + config.radius, y0 + config.radius)
    bounding_box = [top_left, bottom_right]

    draw.ellipse(bounding_box, fill=None, outline="black", width=config.thickness)

def draw_wave(draw: ImageDraw.Draw, config: WaveConfig, origin: tuple[int, int]):
    """Draws a continuous horizontal wave using a sine function."""
    x0, y0 = origin
    points = []
    # Iterate horizontally and calculate the y-offset using sin
    for x_offset in range(-config.length // 2, config.length // 2):
        y_offset = config.amplitude * math.sin(x_offset * config.frequency)
        points.append((x0 + x_offset, y0 + y_offset))

    if points:
        draw.line(points, fill="black", width=config.thickness, joint="curve")

# --- Main Script ---

def generate_stroke_sheet():
    """Creates a canvas and populates it with the generated brush strokes."""
    # Canvas and Grid settings
    cell_size = 200
    grid_cols = 3
    grid_rows = 2
    canvas_width = cell_size * grid_cols
    canvas_height = cell_size * grid_rows
    background_color = "white"

    # Create a blank white canvas
    image = Image.new("RGB", (canvas_width, canvas_height), background_color)
    draw = ImageDraw.Draw(image)

    # List of shapes to draw, pairing each function with its config
    strokes_to_draw = [
        (draw_vertical_line, VerticalLineConfig()),
        (draw_horizontal_line, HorizontalLineConfig()),
        (draw_s_curve, SCurveConfig()),
        (draw_arc, ArcConfig()),
        (draw_circle, CircleConfig()),
        (draw_wave, WaveConfig()),
    ]

    # Iterate through the grid and draw each shape in a cell
    for i, (draw_func, config) in enumerate(strokes_to_draw):
        if i >= grid_cols * grid_rows:
            break # Stop if we run out of cells

        col = i % grid_cols
        row = i // grid_cols

        # Calculate the center of the current cell
        cell_center_x = col * cell_size + cell_size // 2
        cell_center_y = row * cell_size + cell_size // 2
        
        # Call the appropriate drawing function
        draw_func(draw, config, (cell_center_x, cell_center_y))

    # Save and show the final image
    output_filename = "brush_stroke_patterns.png"
    image.save(output_filename)
    print(f"Image saved as '{output_filename}'")
    
    # To display the image automatically after running (optional)
    try:
        image.show()
    except Exception as e:
        print(f"Could not automatically display the image: {e}")

if __name__ == "__main__":
    generate_stroke_sheet()
