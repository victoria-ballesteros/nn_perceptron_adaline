import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ports.external.get_training_set_abc import GetTrainingSetABC

class GetTrainingSet(GetTrainingSetABC):
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(self.filepath)
        X = df.iloc[:, 1:10].values.astype(float)
        y = df.iloc[:, 10].values
        return X, y
