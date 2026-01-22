import numpy as np  # type: ignore
from sklearn.metrics import ( # type: ignore
    accuracy_score,
    confusion_matrix,
)

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

    def run(self, X, y) -> dict:
        self.model.train(X, y)
        y_pred = self.model.predict(X)
        cm = confusion_matrix(y, y_pred)

        tp = cm[1, 1] if cm.shape == (2, 2) else 0
        fp = cm[0, 1] if cm.shape == (2, 2) else 0
        fn = cm[1, 0] if cm.shape == (2, 2) else 0

        accuracy = accuracy_score(y, y_pred)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        return {
            "y_true": y,
            "y_pred": y_pred,
            "accuracy": accuracy,
            "error_rate": 1 - accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": cm,
            "training_errors": self.model.errors,
            "weights": self.model.w,
            "model_name": self.model.__class__.__name__,
        }

