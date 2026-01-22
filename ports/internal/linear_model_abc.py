from abc import ABC, abstractmethod

class LinearModelABC(ABC):
    @abstractmethod
    def initialize_weights(self, n_features) -> None:
        raise NotImplementedError

    @abstractmethod
    def train(self, X, y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def activation(self, z) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def calculate_error(self, y, y_hat) -> float:
        raise NotImplementedError
    
    @abstractmethod
    def update_weights(self, x, y, y_hat) -> None:
        raise NotImplementedError
