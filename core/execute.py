from adapters.visualizer import Visualizer
from models.perceptron import Perceptron
from models.adaline import Adaline
from adapters.get_training_set import GetTrainingSet
from adapters.train import Train
from adapters.utils import Utils


class Execute:
    @staticmethod
    def execute() -> None:
        dataset = GetTrainingSet("data/glass.csv")
        X_raw, y_raw = dataset.load()

        models = [
            ("Perceptron", Perceptron(learning_rate=0.01, epochs=500)),
            ("ADALINE", Adaline(learning_rate=0.0005, epochs=500)),
        ]

        for _, model in models:
            trainer = Train(model, X_raw, y_raw)
            X_train, y_train = trainer.prepare_data(
                columns=None, target_func=Utils.windows_vs_non_windows
            )
            trainer.run(X_train, y_train)

    @staticmethod
    def visualize() -> None:
        dataset = GetTrainingSet("data/glass.csv")
        X_raw, y_raw = dataset.load()
        columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

        viz = Visualizer(X_raw, y_raw, columns)
        viz.run()
