import numpy as np  # type: ignore
from abc import abstractmethod

from ports.internal.linear_model_abc import LinearModelABC


class LinearModel(LinearModelABC):
    def __init__(self, learning_rate: float = 0.01, epochs: int = 50) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None

    def initialize_weights(self, n_features) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0

    def train(self, X, y) -> None:
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)

        for _ in range(self.epochs):
            for i in range(n_samples):
                z = np.dot(self.w, X[i]) + self.b
                y_hat = self.activation(z)
                self.learn(X[i], y[i], y_hat)

    def predict(self, X) -> int:
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, -1)

    @abstractmethod
    def activation(self, z) -> float:
        raise NotImplementedError

    @abstractmethod
    def learn(self, x, y, y_hat) -> None:
        raise NotImplementedError