"""
Design analysis module for intelligent parameter detection.

This module analyzes G-code designs to automatically determine optimal
stroke sampling parameters based on design complexity and characteristics.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from tatbot.utils.log import get_logger

log = get_logger("gen.analyze", "🔍")


@dataclass
class DesignStats:
    """Statistics about a design's complexity and characteristics."""
    
    total_strokes: int
    total_length_m: float
    median_length_m: float
    complexity_score: float
    suggested_spacing_m: float
    suggested_max_points: int
    
    # Additional statistics
    min_length_m: float = 0.0
    max_length_m: float = 0.0
    avg_length_m: float = 0.0
    std_length_m: float = 0.0
    total_points: int = 0
    avg_curvature: float = 0.0
    density_score: float = 0.0


def analyze_gcode_paths(paths: List[np.ndarray]) -> DesignStats:
    """
    Analyze a list of paths from G-code to determine design statistics.
    
    Args:
        paths: List of numpy arrays, each containing path points (N, 2) or (N, 3)
        
    Returns:
        DesignStats object with analysis results
    """
    if not paths:
        return DesignStats(
            total_strokes=0,
            total_length_m=0.0,
            median_length_m=0.0,
            complexity_score=0.0,
            suggested_spacing_m=0.002,  # Default 2mm
            suggested_max_points=128
        )
    
    # Calculate path lengths
    lengths: list[float] = []
    total_points = 0
    curvatures = []
    
    for path in paths:
        if len(path) < 2:
            continue
        
        # Calculate arc length
        segments = np.diff(path, axis=0)
        segment_lengths = np.linalg.norm(segments, axis=1)
        path_length = segment_lengths.sum()
        lengths.append(path_length)
        total_points += len(path)
        
        # Calculate curvature (simplified as angle changes)
        if len(path) >= 3:
            # Calculate angles between consecutive segments
            for i in range(len(segments) - 1):
                v1 = segments[i]
                v2 = segments[i + 1]
                # Normalize vectors
                v1_norm = v1 / (np.linalg.norm(v1) + 1e-10)
                v2_norm = v2 / (np.linalg.norm(v2) + 1e-10)
                # Calculate angle
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                curvatures.append(angle)
    
    lengths = np.array(lengths)
    
    # Calculate statistics
    total_length = lengths.sum()
    min_length = lengths.min() if len(lengths) > 0 else 0.0
    max_length = lengths.max() if len(lengths) > 0 else 0.0
    avg_length = lengths.mean() if len(lengths) > 0 else 0.0
    median_length = np.median(lengths) if len(lengths) > 0 else 0.0
    std_length = lengths.std() if len(lengths) > 0 else 0.0
    
    # Calculate average curvature
    avg_curvature = np.mean(curvatures) if curvatures else 0.0
    
    # Calculate complexity score (0-1 scale)
    complexity_score = compute_complexity_score(
        paths, lengths, avg_curvature
    )
    
    # Calculate density score (points per meter)
    density_score = total_points / total_length if total_length > 0 else 0.0
    
    # Suggest spacing based on complexity
    suggested_spacing_m = suggest_spacing(
        median_length,
        complexity_score,
        density_score
    )
    
    # Suggest max points based on longest path
    suggested_max_points = suggest_max_points(
        max_length,
        suggested_spacing_m
    )
    
    return DesignStats(
        total_strokes=len(paths),
        total_length_m=total_length,
        median_length_m=median_length,
        complexity_score=complexity_score,
        suggested_spacing_m=suggested_spacing_m,
        suggested_max_points=suggested_max_points,
        min_length_m=min_length,
        max_length_m=max_length,
        avg_length_m=avg_length,
        std_length_m=std_length,
        total_points=total_points,
        avg_curvature=avg_curvature,
        density_score=density_score
    )


def compute_complexity_score(
    paths: List[np.ndarray],
    lengths: np.ndarray,
    avg_curvature: float
) -> float:
    """
    Compute a complexity score for the design (0-1 scale).
    
    Args:
        paths: List of path arrays
        lengths: Array of path lengths
        avg_curvature: Average curvature across all paths
        
    Returns:
        Complexity score between 0 (simple) and 1 (complex)
    """
    if len(paths) == 0:
        return 0.0
    
    # Factor 1: Number of strokes (normalized)
    stroke_factor = min(1.0, len(paths) / 100.0)  # Cap at 100 strokes
    
    # Factor 2: Length variation (coefficient of variation)
    if len(lengths) > 1 and lengths.mean() > 0:
        cv = lengths.std() / lengths.mean()
        variation_factor = min(1.0, cv)  # Cap at 1.0
    else:
        variation_factor = 0.0
    
    # Factor 3: Curvature (normalized)
    # Higher curvature = more complex
    curvature_factor = min(1.0, avg_curvature / (np.pi / 4))  # Normalize by 45 degrees
    
    # Factor 4: Fine detail detection
    # Check for very short strokes indicating detail work
    if len(lengths) > 0:
        short_strokes = (lengths < np.percentile(lengths, 25)).sum()
        detail_factor = min(1.0, short_strokes / len(lengths))
    else:
        detail_factor = 0.0
    
    # Weighted combination
    complexity = (
        0.2 * stroke_factor +
        0.2 * variation_factor +
        0.4 * curvature_factor +  # Weight curvature more heavily
        0.2 * detail_factor
    )
    
    return min(1.0, complexity)


def suggest_spacing(
    median_length: float,
    complexity_score: float,
    density_score: float,
    min_spacing: float = 0.0005,  # 0.5mm minimum
    max_spacing: float = 0.010    # 10mm maximum
) -> float:
    """
    Suggest optimal point spacing based on design characteristics.
    
    Args:
        median_length: Median stroke length in meters
        complexity_score: Design complexity (0-1)
        density_score: Current point density (points/meter)
        min_spacing: Minimum allowed spacing
        max_spacing: Maximum allowed spacing
        
    Returns:
        Suggested spacing in meters
    """
    # Base spacing on median stroke length
    # Aim for ~32-64 points per median stroke
    target_points_per_stroke = 32 + (1.0 - complexity_score) * 32  # More points for complex
    base_spacing = median_length / target_points_per_stroke if median_length > 0 else 0.002
    
    # Adjust for complexity (finer spacing for complex designs)
    complexity_multiplier = 1.0 - (complexity_score * 0.5)  # 0.5x to 1.0x
    adjusted_spacing = base_spacing * complexity_multiplier
    
    # Consider existing density if available
    if density_score > 0:
        # Try to maintain similar density
        density_spacing = 1.0 / density_score
        # Blend with complexity-based spacing
        adjusted_spacing = 0.7 * adjusted_spacing + 0.3 * density_spacing
    
    # Clamp to valid range
    suggested = np.clip(adjusted_spacing, min_spacing, max_spacing)
    
    # Round to nearest 0.1mm for cleaner values
    suggested = np.round(suggested * 10000) / 10000  # Round to 0.1mm
    
    log.debug(
        f"Suggested spacing: {suggested:.4f}m "
        f"(median_len={median_length:.3f}, complexity={complexity_score:.2f})"
    )
    
    return suggested


def suggest_max_points(
    max_length: float,
    spacing: float,
    min_points: int = 64,
    max_points: int = 512
) -> int:
    """
    Suggest maximum points for padding based on longest stroke.
    
    Args:
        max_length: Maximum stroke length in meters
        spacing: Point spacing in meters
        min_points: Minimum allowed points
        max_points: Maximum allowed points
        
    Returns:
        Suggested maximum points for padding
    """
    if spacing <= 0:
        return min_points
    
    # Calculate points needed for longest stroke
    required_points = int(np.ceil(max_length / spacing)) + 1
    
    # Add 20% buffer for safety
    suggested = int(required_points * 1.2)
    
    # Round up to nearest power of 2 for better GPU performance
    suggested = 2 ** int(np.ceil(np.log2(suggested)))
    
    # Clamp to valid range
    suggested = np.clip(suggested, min_points, max_points)
    
    log.debug(
        f"Suggested max points: {suggested} "
        f"(max_len={max_length:.3f}m, spacing={spacing:.4f}m)"
    )
    
    return suggested


def analyze_design_files(
    gcode_files: List[Path],
    target_points: Optional[int] = None
) -> DesignStats:
    """
    Analyze G-code files to determine optimal parameters.
    
    Args:
        gcode_files: List of G-code file paths
        target_points: Optional target points per stroke
        
    Returns:
        DesignStats with analysis and suggestions
    """
    all_paths = []
    
    for gcode_file in gcode_files:
        paths = parse_gcode_for_analysis(gcode_file)
        all_paths.extend(paths)
    
    stats = analyze_gcode_paths(all_paths)
    
    # Adjust suggestions if target points specified
    if target_points is not None and stats.median_length_m > 0:
        stats.suggested_spacing_m = stats.median_length_m / target_points
        stats.suggested_max_points = target_points * 2  # 2x buffer
    
    return stats


def parse_gcode_for_analysis(gcode_path: Path) -> List[np.ndarray]:
    """
    Parse G-code file to extract paths for analysis.
    
    Args:
        gcode_path: Path to G-code file
        
    Returns:
        List of path arrays in millimeters
    """
    paths = []
    current_path = []
    pen_down = False
    
    with open(gcode_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            tokens = line.split()
            if not tokens:
                continue
            
            cmd = tokens[0].upper()
            
            if cmd == 'G0':  # Pen up
                if pen_down and current_path:
                    # Convert to meters and save path
                    path_array = np.array(current_path, dtype=np.float32) / 1000.0
                    paths.append(path_array)
                current_path: list[list[float]] = []
                pen_down = False
                
                # Parse position
                x = y = None
                for token in tokens[1:]:
                    if token.startswith('X'):
                        x = float(token[1:])
                    elif token.startswith('Y'):
                        y = float(token[1:])
                if x is not None and y is not None:
                    current_path.append([x, y])
                    
            elif cmd == 'G1':  # Pen down
                pen_down = True
                
                # Parse position
                x = y = None
                for token in tokens[1:]:
                    if token.startswith('X'):
                        x = float(token[1:])
                    elif token.startswith('Y'):
                        y = float(token[1:])
                if x is not None and y is not None:
                    current_path.append([x, y])
    
    # Save last path if exists
    if pen_down and current_path:
        path_array = np.array(current_path, dtype=np.float32) / 1000.0
        paths.append(path_array)
    
    return paths


def recommend_config_from_stats(stats: DesignStats) -> dict:
    """
    Generate recommended configuration based on design statistics.
    
    Args:
        stats: Design statistics
        
    Returns:
        Dictionary with recommended configuration values
    """
    config = {
        "stroke_sample_mode": "distance",
        "stroke_point_spacing_m": stats.suggested_spacing_m,
        "stroke_max_points": stats.suggested_max_points,
    }
    
    # Add auto-config if complexity warrants it
    if stats.complexity_score > 0.7:
        config["auto_sample_config"] = {
            "percentile": 90,
            "min_spacing_m": 0.0005,
            "max_spacing_m": 0.005,  # Finer max for complex designs
        }
    elif stats.complexity_score < 0.3:
        config["auto_sample_config"] = {
            "percentile": 75,
            "min_spacing_m": 0.002,
            "max_spacing_m": 0.010,  # Coarser for simple designs
        }
    
    log.info(
        f"Recommended config for design with {stats.total_strokes} strokes, "
        f"complexity={stats.complexity_score:.2f}: spacing={stats.suggested_spacing_m:.4f}m, "
        f"max_points={stats.suggested_max_points}"
    )
    
    return config