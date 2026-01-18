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
    def learn(self, x, y, y_hat) -> None:
        raise NotImplementedError
