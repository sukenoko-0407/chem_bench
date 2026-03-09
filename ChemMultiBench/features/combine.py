from __future__ import annotations

import numpy as np


def concat_features(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate(arrays, axis=1)

