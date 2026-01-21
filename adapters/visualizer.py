import matplotlib # type: ignore

matplotlib.use("Agg")
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from sklearn.decomposition import PCA # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
import os
import warnings

warnings.filterwarnings("ignore")


class Visualizer:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        columns: list,
        output_dir: str = "data/plots",
    ):
        self.df = pd.DataFrame(X, columns=columns)
        self.df["Type"] = y
        self.df["Binary"] = np.where(y == 1, "Windows", "No-Windows")
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)

    def scatter(self, x_attr: str, y_attr: str, filename: str = "scatter.png") -> None:
        plt.figure(figsize=(8, 6))
        for label in self.df["Binary"].unique():
            subset = self.df[self.df["Binary"] == label]
            plt.scatter(subset[x_attr], subset[y_attr], label=label, alpha=0.7)
        plt.title(f"{x_attr} vs {y_attr}")
        plt.xlabel(x_attr)
        plt.ylabel(y_attr)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def heatmap(self, filename: str = "heatmap.png") -> None:
        plt.figure(figsize=(10, 8))
        corr_matrix = self.df.iloc[:, :-2].corr() # type: ignore
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
        plt.title("Mapa de calor de correlación")
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def pairplot(self, selected_attrs: list, filename: str = "pairplot.png") -> None:
        plot = sns.pairplot(
            self.df, vars=selected_attrs, hue="Binary", corner=True, palette="Set1"
        )
        plot.savefig(os.path.join(self.output_dir, filename))
        plt.close()

    def class_distribution(self, filename: str = "class_distribution.png") -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        class_counts = self.df["Type"].value_counts().sort_index()
        axes[0].bar(
            class_counts.index.astype(str), class_counts.values, color="skyblue"
        )
        axes[0].set_title("Distribución de Clases Original")
        axes[0].set_xlabel("Clase")
        axes[0].set_ylabel("Número de muestras")
        for i, v in enumerate(class_counts.values):
            axes[0].text(i, v + 0.5, str(v), ha="center")

        binary_counts = self.df["Binary"].value_counts()
        axes[1].bar(
            binary_counts.index,
            binary_counts.values,
            color=["lightgreen", "lightcoral"],
        )
        axes[1].set_title("Distribución Binaria")
        axes[1].set_xlabel("Categoría")
        axes[1].set_ylabel("Número de muestras")
        for i, v in enumerate(binary_counts.values):
            axes[1].text(i, v + 0.5, str(v), ha="center")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def pca_scatter(self, filename: str = "pca_scatter.png") -> None:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)

        pca_df = pd.DataFrame(
            {
                "PC1": X_pca[:, 0],
                "PC2": X_pca[:, 1],
                "Class": self.df["Type"],
                "Binary": self.df["Binary"],
            }
        )

        explained_var = pca.explained_variance_ratio_ * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        scatter1 = axes[0].scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df["Class"],
            cmap="tab10",
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
            s=50,
        )
        axes[0].set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
        axes[0].set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
        axes[0].set_title("PCA - Multiclase")
        plt.colorbar(scatter1, ax=axes[0], label="Clase")

        colors = {"Windows": "green", "No-Windows": "red"}
        for label, color in colors.items():
            mask = pca_df["Binary"] == label
            axes[1].scatter(
                pca_df.loc[mask, "PC1"],
                pca_df.loc[mask, "PC2"],
                c=color,
                label=label,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.5,
                s=50,
            )

        axes[1].set_xlabel(f"PC1 ({explained_var[0]:.1f}%)")
        axes[1].set_ylabel(f"PC2 ({explained_var[1]:.1f}%)")
        axes[1].set_title("PCA - Binaria")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def feature_distribution(self, filename: str = "feature_distribution.png") -> None:
        key_features = ["Mg", "Al", "Si", "Ca", "Na", "K"]
        available_features = [f for f in key_features if f in self.df.columns]

        n_features = len(available_features)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, feature in enumerate(available_features):
            if i < len(axes):
                box = axes[i].boxplot(
                    [
                        self.df[self.df["Binary"] == cat][feature].values
                        for cat in self.df["Binary"].unique()
                    ],
                    labels=self.df["Binary"].unique(),
                    patch_artist=True,
                )

                colors = ["lightgreen", "lightcoral"]
                for patch, color in zip(box["boxes"], colors):
                    patch.set_facecolor(color)

                axes[i].set_title(f"Distribución de {feature}")
                axes[i].set_ylabel("Valor")
                axes[i].grid(True, alpha=0.3)

        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        plt.suptitle("Distribución de Características por Clase", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300)
        plt.close()

    def print_summary(self):
        print("=== RESUMEN DEL DATASET ===")
        print(f"Total muestras: {len(self.df)}")
        print(f"Características: {len(self.df.columns) - 2}")
        print("\nDistribución de clases original:")
        print(self.df["Type"].value_counts().sort_index())
        print("\nDistribución binaria:")
        print(self.df["Binary"].value_counts())

        print("\n=== CORRELACIONES ALTAS (>0.7) ===")
        corr_matrix = self.df.iloc[:, :-2].corr() # type: ignore
        high_corr = corr_matrix[(corr_matrix.abs() > 0.7) & (corr_matrix != 1.0)]
        for col in high_corr.columns:
            high_values = high_corr[col].dropna()
            if len(high_values) > 0:
                print(f"{col}: {high_values.to_dict()}")

        print("\n=== ESTADÍSTICAS POR CLASE BINARIA ===")
        for group in self.df["Binary"].unique():
            print(f"\n{group}:")
            group_data = self.df[self.df["Binary"] == group]
            print(f"  RI: {group_data['RI'].mean():.3f} ± {group_data['RI'].std():.3f}")
            print(f"  Na: {group_data['Na'].mean():.3f} ± {group_data['Na'].std():.3f}")
            print(f"  Mg: {group_data['Mg'].mean():.3f} ± {group_data['Mg'].std():.3f}")

    def run(self) -> None:
        self.print_summary()
        self.scatter("Mg", "Al", "scatter_Mg_Al.png")
        self.heatmap("heatmap_correlation.png")
        self.pairplot(["Mg", "Al", "Si", "Na"], "pairplot_selected.png")
        self.class_distribution()
        self.pca_scatter()
        self.feature_distribution()
