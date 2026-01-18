import numpy as np  # type: ignore
from typing import Any


class Utils:
    @staticmethod
    def windows_vs_non_windows(y_raw: float) -> Any:
        return np.where(y_raw <= 4, 1, -1)