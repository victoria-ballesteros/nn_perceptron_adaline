import numpy as np  # type: ignore
from abc import abstractmethod

from ports.internal.linear_model_abc import LinearModelABC


class LinearModel(LinearModelABC):
    def __init__(self, learning_rate=0.01, epochs=50) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.errors = []
    
    def initialize_weights(self, n_features) -> None:
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.errors = []
    
    def train(self, X, y) -> None:
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        
        for _ in range(self.epochs):
            epoch_error = 0
            
            for i in range(n_samples):
                z = np.dot(self.w, X[i]) + self.b
                y_hat = self.activation(z)
                error = self.calculate_error(y[i], y_hat)
                self.update_weights(X[i], y[i], y_hat)
                epoch_error += abs(error)
            
            avg_error = epoch_error / n_samples
            self.errors.append(avg_error)
    
    def predict(self, X) -> np.ndarray:
        z = np.dot(X, self.w) + self.b
        return np.where(z >= 0, 1, -1)
    
    @abstractmethod
    def activation(self, z) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def calculate_error(self, y, y_hat) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def update_weights(self, x, y, y_hat) -> None:
        raise NotImplementedError