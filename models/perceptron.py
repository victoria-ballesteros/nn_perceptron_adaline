from adapters.linear_model import LinearModel


class Perceptron(LinearModel):
    def activation(self, z) -> float:
        return 1 if z >= 0 else -1

    def learn(self, x, y, y_hat) -> None:
        if y != y_hat:
            self.w += self.learning_rate * y * x
            self.b += self.learning_rate * y
