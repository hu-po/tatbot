"""
Padding strategies for variable-length strokes.

This module provides functions to pad strokes to uniform lengths for GPU batch processing
while maintaining stroke quality and execution efficiency.
"""

from enum import Enum
from typing import Optional

import numpy as np

from tatbot.data.stroke import Stroke
from tatbot.utils.log import get_logger

log = get_logger("gen.padding", "📦")


class PaddingStrategy(Enum):
    """Available padding strategies for strokes."""
    
    REPEAT_LAST = "repeat_last"      # Repeat last point (hover in place)
    LINEAR_INTERP = "linear_interp"  # Linear interpolation
    ZERO_PAD = "zero_pad"            # Pad with zeros
    HOVER_OFFSET = "hover_offset"    # Apply hover offset to padding


def pad_array_to_length(
    array: np.ndarray,
    target_length: int,
    strategy: PaddingStrategy = PaddingStrategy.REPEAT_LAST
) -> tuple[np.ndarray, np.ndarray]:
    """
    Pad a numpy array to target length using specified strategy.
    
    Args:
        array: Input array to pad (N, D) where N is current length
        target_length: Desired length after padding
        strategy: Padding strategy to use
        
    Returns:
        Tuple of (padded_array, pad_mask) where pad_mask is True for valid points
    """
    current_length = array.shape[0]
    
    # If already at target length or longer, truncate
    if current_length >= target_length:
        return array[:target_length], np.ones(target_length, dtype=bool)
    
    # Handle empty arrays
    if current_length == 0:
        padded = np.zeros((target_length, *array.shape[1:]), dtype=array.dtype)
        mask = np.zeros(target_length, dtype=bool)
        return padded, mask
    
    # Create mask (True for valid points, False for padding)
    mask = np.zeros(target_length, dtype=bool)
    mask[:current_length] = True
    
    # Apply padding strategy
    pad_count = target_length - current_length
    
    if strategy == PaddingStrategy.REPEAT_LAST:
        # Repeat the last point
        last_point = array[-1:]
        padding = np.repeat(last_point, pad_count, axis=0)
        padded = np.concatenate([array, padding], axis=0)
        
    elif strategy == PaddingStrategy.LINEAR_INTERP:
        # Linear interpolation between last point and first point (for loops)
        if current_length >= 2:
            # Interpolate from last to first for smooth loop closure
            first_point = array[0]
            last_point = array[-1]
            alpha = np.linspace(0, 1, pad_count + 1)[1:]  # Exclude 0
            padding = last_point + alpha[:, None] * (first_point - last_point)
            padded = np.concatenate([array, padding], axis=0)
        else:
            # Fall back to repeat if only one point
            padded = np.repeat(array, target_length, axis=0)
            
    elif strategy == PaddingStrategy.ZERO_PAD:
        # Pad with zeros
        padding = np.zeros((pad_count, *array.shape[1:]), dtype=array.dtype)
        padded = np.concatenate([array, padding], axis=0)
        
    elif strategy == PaddingStrategy.HOVER_OFFSET:
        # This strategy is handled in apply_padding for stroke-specific logic
        # For now, fall back to repeat_last
        last_point = array[-1:]
        padding = np.repeat(last_point, pad_count, axis=0)
        padded = np.concatenate([array, padding], axis=0)
        
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")
    
    return padded.astype(array.dtype), mask


def apply_padding(
    stroke: Stroke,
    target_length: int,
    strategy: PaddingStrategy = PaddingStrategy.REPEAT_LAST,
    hover_offset: Optional[np.ndarray] = None
) -> Stroke:
    """
    Apply padding to a stroke to reach target length.
    
    Args:
        stroke: Input stroke to pad
        target_length: Desired length after padding
        strategy: Padding strategy to use
        hover_offset: Optional hover offset for HOVER_OFFSET strategy
        
    Returns:
        Padded stroke with metadata updated
    """
    if stroke.is_rest:
        # Rest strokes are already at the correct length
        return stroke
    
    # Track original length before padding
    actual_points = 0
    
    # Handle meter_coords
    if stroke.meter_coords is not None:
        actual_points = stroke.meter_coords.shape[0]
        
        if strategy == PaddingStrategy.HOVER_OFFSET and hover_offset is not None:
            # Special handling for hover offset
            padded_coords, mask = pad_array_to_length(
                stroke.meter_coords, target_length, PaddingStrategy.REPEAT_LAST
            )
            # Apply hover offset to padded region
            if actual_points < target_length:
                padded_coords[actual_points:] += hover_offset
        else:
            padded_coords, mask = pad_array_to_length(
                stroke.meter_coords, target_length, strategy
            )
        
        stroke.meter_coords = padded_coords
        stroke.pad_mask = mask
        stroke.actual_points = actual_points
    
    # Handle pixel_coords
    if stroke.pixel_coords is not None:
        padded_pixels, _ = pad_array_to_length(
            stroke.pixel_coords, target_length, strategy
        )
        stroke.pixel_coords = padded_pixels
    
    # Handle ee_pos
    if stroke.ee_pos is not None:
        if strategy == PaddingStrategy.HOVER_OFFSET and hover_offset is not None:
            padded_ee, _ = pad_array_to_length(
                stroke.ee_pos, target_length, PaddingStrategy.REPEAT_LAST
            )
            # Apply hover offset to padded region
            current_len = stroke.ee_pos.shape[0]
            if current_len < target_length:
                padded_ee[current_len:] += hover_offset
        else:
            padded_ee, _ = pad_array_to_length(
                stroke.ee_pos, target_length, strategy
            )
        stroke.ee_pos = padded_ee
    
    # Handle ee_rot
    if stroke.ee_rot is not None:
        # Rotation should typically use REPEAT_LAST to maintain orientation
        padded_rot, _ = pad_array_to_length(
            stroke.ee_rot, target_length, PaddingStrategy.REPEAT_LAST
        )
        stroke.ee_rot = padded_rot
    
    # Handle normals
    if stroke.normals is not None:
        # Normals should use REPEAT_LAST to maintain surface orientation
        padded_normals, _ = pad_array_to_length(
            stroke.normals, target_length, PaddingStrategy.REPEAT_LAST
        )
        stroke.normals = padded_normals
    
    log.debug(
        f"Padded stroke from {actual_points} to {target_length} points "
        f"using {strategy.value} strategy"
    )
    
    return stroke


def pad_stroke_list(
    strokes: list[tuple[Stroke, Stroke]],
    target_length: Optional[int] = None,
    strategy: PaddingStrategy = PaddingStrategy.REPEAT_LAST,
    hover_offset: Optional[np.ndarray] = None
) -> list[tuple[Stroke, Stroke]]:
    """
    Pad all strokes in a list to uniform length.
    
    Args:
        strokes: List of stroke pairs (left, right)
        target_length: Target length (if None, uses max length)
        strategy: Padding strategy to use
        hover_offset: Optional hover offset for HOVER_OFFSET strategy
        
    Returns:
        List of padded stroke pairs
    """
    # Determine target length if not specified
    if target_length is None:
        max_len = 0
        for stroke_l, stroke_r in strokes:
            if stroke_l.meter_coords is not None:
                max_len = max(max_len, stroke_l.meter_coords.shape[0])
            if stroke_r.meter_coords is not None:
                max_len = max(max_len, stroke_r.meter_coords.shape[0])
        target_length = max_len
    
    if target_length <= 0:
        raise ValueError("Cannot pad to length <= 0")
    
    # Pad all strokes
    padded_strokes = []
    for stroke_l, stroke_r in strokes:
        padded_l = apply_padding(stroke_l, target_length, strategy, hover_offset)
        padded_r = apply_padding(stroke_r, target_length, strategy, hover_offset)
        padded_strokes.append((padded_l, padded_r))
    
    log.info(f"Padded {len(strokes)} stroke pairs to length {target_length}")
    
    return padded_strokes


def create_pad_mask(actual_length: int, padded_length: int) -> np.ndarray:
    """
    Create a boolean mask identifying valid vs padded points.
    
    Args:
        actual_length: Number of valid points
        padded_length: Total length after padding
        
    Returns:
        Boolean array where True indicates valid points
    """
    mask = np.zeros(padded_length, dtype=bool)
    mask[:actual_length] = True
    return mask


def compute_padding_stats(strokes: list[tuple[Stroke, Stroke]]) -> dict:
    """
    Compute statistics about padding in a stroke list.
    
    Args:
        strokes: List of stroke pairs
        
    Returns:
        Dictionary with padding statistics
    """
    actual_lengths = []
    padded_lengths = []
    
    for stroke_l, stroke_r in strokes:
        if stroke_l.actual_points is not None:
            actual_lengths.append(stroke_l.actual_points)
            if stroke_l.meter_coords is not None:
                padded_lengths.append(stroke_l.meter_coords.shape[0])
        
        if stroke_r.actual_points is not None:
            actual_lengths.append(stroke_r.actual_points)
            if stroke_r.meter_coords is not None:
                padded_lengths.append(stroke_r.meter_coords.shape[0])
    
    if not actual_lengths:
        return {
            "total_strokes": len(strokes) * 2,
            "padded_strokes": 0,
            "avg_actual_length": 0,
            "avg_padded_length": 0,
            "padding_ratio": 0.0,
            "memory_overhead": 0.0
        }
    
    actual_lengths = np.array(actual_lengths)
    padded_lengths = np.array(padded_lengths)
    
    total_actual = actual_lengths.sum()
    total_padded = padded_lengths.sum()
    
    return {
        "total_strokes": len(strokes) * 2,
        "padded_strokes": len(actual_lengths),
        "avg_actual_length": actual_lengths.mean(),
        "avg_padded_length": padded_lengths.mean(),
        "min_actual_length": actual_lengths.min(),
        "max_actual_length": actual_lengths.max(),
        "padding_ratio": 1.0 - (total_actual / total_padded) if total_padded > 0 else 0.0,
        "memory_overhead": (total_padded - total_actual) / total_actual if total_actual > 0 else 0.0
    }