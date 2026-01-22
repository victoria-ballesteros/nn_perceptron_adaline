from adapters.linear_model import LinearModel


class Perceptron(LinearModel):
    def activation(self, z) -> float:
        return 1 if z >= 0 else -1
    
    def calculate_error(self, y, y_hat) -> float:
        return y - y_hat

    def update_weights(self, x, y, y_hat) -> None:
        if y != y_hat:
            self.w += self.learning_rate * y * x
            self.b += self.learning_rate * y
