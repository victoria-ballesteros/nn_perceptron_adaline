from abc import ABC, abstractmethod
import numpy as np  # type: ignore


class GetTrainingSetABC(ABC):
    @abstractmethod
    def load(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError