import numpy as np  # type: ignore

class Train:
    def __init__(self, model, X_raw: np.ndarray, y_raw: np.ndarray) -> None:
        self.model = model
        self.X_raw = X_raw
        self.y_raw = y_raw

    def prepare_data(
        self, columns: list[int] | None = None, target_func=None
    ) -> tuple[np.ndarray, np.ndarray]:
        X = self.X_raw if columns is None else self.X_raw[:, columns]
        y = self.y_raw if target_func is None else target_func(self.y_raw)

        X = (X - X.mean(axis=0)) / X.std(axis=0)

        return X, y

    def run(self, X, y) -> float:
        self.model.train(X, y)
        y_pred = self.model.predict(X)
        accuracy = (y_pred == y).mean()
        print(f"{self.model.__class__.__name__} Accuracy: {accuracy:.4f}")
        return y_pred
