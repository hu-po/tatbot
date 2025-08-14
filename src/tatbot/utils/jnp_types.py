from typing import Any, Optional

import jax
import jaxlib.xla_extension as xla_ext
import numpy as np


def ensure_numpy_array(obj: Any) -> Optional[np.ndarray]:
    """Convert JAX arrays to NumPy arrays, leave other types unchanged."""
    if obj is None:
        return None
    
    if isinstance(obj, (jax.Array, xla_ext.ArrayImpl)):
        return np.asarray(obj)
    
    if isinstance(obj, np.ndarray):
        return obj
    
    return np.asarray(obj)