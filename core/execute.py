from adapters.model_result_visualizer import ModelResultsVisualizer
from adapters.training_data_visualizer import TrainingDataVisualizer
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
    def visualize_training_data() -> None:
        dataset = GetTrainingSet("data/glass.csv")
        X_raw, y_raw = dataset.load()
        columns = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]

        viz = TrainingDataVisualizer(X_raw, y_raw, columns)
        viz.run()

    @staticmethod
    def visualize_results() -> None:
        dataset = GetTrainingSet("data/glass.csv")
        X_raw, y_raw = dataset.load()

        visualizer = ModelResultsVisualizer()

        models = [
            ("Perceptrón", Perceptron(learning_rate=0.01, epochs=500)),
            ("Adaline", Adaline(learning_rate=0.0005, epochs=500)),
        ]

        for name, model in models:
            trainer = Train(model, X_raw, y_raw)
            X_train, y_train = trainer.prepare_data(
                columns=None, target_func=Utils.windows_vs_non_windows
            )
            results = trainer.run(X_train, y_train)
            visualizer.add_results(name, results)
            print(f"Error rate: {name}: {results['error_rate']:.4f}")

        visualizer.plot_convergence_curves()
        visualizer.plot_confusion_matrices()
        visualizer.plot_metrics_comparison()

        feature_names = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
        visualizer.plot_weights_comparison(feature_names)

        print(f"\nTodas las gráficas guardadas en: {visualizer.output_dir}")
