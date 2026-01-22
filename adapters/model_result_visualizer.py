import matplotlib # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import os

class ModelResultsVisualizer:
    def __init__(self, output_dir="data/plots") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = {}

    def add_results(self, model_name, results_dict) -> None:
        self.results[model_name] = results_dict

    def plot_convergence_curves(self, filename="convergence.png") -> None:
        plt.figure(figsize=(10, 6))

        for name, results in self.results.items():
            if "training_errors" in results and results["training_errors"]:
                errors = results["training_errors"]
                plt.plot(
                    range(1, len(errors) + 1),
                    errors,
                    label=name,
                    linewidth=2,
                    alpha=0.8,
                )

        plt.xlabel("Época")
        plt.ylabel("Error")
        plt.title("Curvas de Convergencia")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_confusion_matrices(self, filename="confusion_matrices.png"):
        n_models = len(self.results)
        _, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for ax, (name, results) in zip(axes, self.results.items()):
            cm = results['confusion_matrix']
            
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues) # type: ignore
            ax.figure.colorbar(im, ax=ax)
            
            threshold = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > threshold else "black",
                        fontsize=12, fontweight='bold')
            
            ax.set_title(f'{name}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Predicción', fontsize=12)
            ax.set_ylabel('Real', fontsize=12)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['-1', '1'], fontsize=11)
            ax.set_yticklabels(['-1', '1'], fontsize=11)
            
            ax.set_xticks(np.arange(cm.shape[1]+1)-.5, minor=True)
            ax.set_yticks(np.arange(cm.shape[0]+1)-.5, minor=True)
            ax.grid(which="minor", color="gray", linestyle='-', linewidth=1)
            ax.tick_params(which="minor", bottom=False, left=False)
        
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, filename), 
            dpi=300, 
            facecolor='white',
            bbox_inches='tight'
        )
        plt.close()

    def plot_metrics_comparison(self, filename="metrics_comparison.png") -> None:
        metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
        models = list(self.results.keys())

        data = {metric: [] for metric in metrics_names}
        for model in models:
            results = self.results[model]
            data["Accuracy"].append(results["accuracy"])
            data["Precision"].append(results["precision"])
            data["Recall"].append(results["recall"])
            data["F1-Score"].append(results["f1"])

        x = np.arange(len(models))
        width = 0.2
        multiplier = 0

        _, ax = plt.subplots(figsize=(10, 6))

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        for i, (metric, values) in enumerate(data.items()):
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=metric, color=colors[i])
            ax.bar_label(bars, padding=3, fmt="%.3f")
            multiplier += 1

        ax.set_ylabel("Valor")
        ax.set_title("Comparación de Métricas")
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(models)
        ax.legend(loc="upper left", ncols=2)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def plot_weights_comparison(self, feature_names, filename="weights_comparison.png") -> None:
        n_models = len(self.results)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))

        if n_models == 1:
            axes = [axes]

        for ax, (name, results) in zip(axes, self.results.items()):
            weights = results["weights"]
            colors = ["green" if w > 0 else "red" for w in weights]
            ax.barh(feature_names, weights, color=colors)
            ax.set_title(f"Pesos - {name}")
            ax.set_xlabel("Valor")
            ax.axvline(x=0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()
