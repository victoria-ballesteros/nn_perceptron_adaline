from adapters.linear_model import LinearModel


class Adaline(LinearModel):
    def activation(self, z) -> float:
        return z

    def learn(self, x, y, y_hat) -> None:
        error = y - y_hat
        self.w += self.learning_rate * error * x
        self.b += self.learning_rate * error
