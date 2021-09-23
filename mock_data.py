"""Mock data generation, for testing of `dataset` and `col_split` modules.
"""

import numpy as np
from dataset import ColSpec
from typing import List

def generate_mock_data(cols: ColSpec, length: int = 200):
    """Generate a np.ndarray with random integer point data
    
    Args:
        cols: List of names for columns. Is just used for finding out how many
              columns to include in the array.
    """
    rng = np.random.default_rng()

    data = rng.random(size=(length, len(cols)))

    if isinstance(cols, dict):
        for i, (col, groupings) in enumerate(cols.items()):
            if "stratify" in groupings:
                data[:,i] = rng.integers(10, size=(length))

    return data
