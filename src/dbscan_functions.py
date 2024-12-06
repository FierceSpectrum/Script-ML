import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


def configure_dbscan(eps, min_samples):
    return DBSCAN(eps=eps, min_samples=min_samples)


def fit_dbscan(dbscan, data):
    dbscan.fit(data)
    return dbscan.labels_


def evaluate_model(labels, data, label="Entrenamiento"):
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    num_noise = sum(labels == -1)

    print(f"Número de clusters ({label}): {num_clusters}")
    print(f"Puntos clasificados como ruido ({label}): {num_noise}")

    if len(set(labels)) > 1:
        silhouette = silhouette_score(data, labels)
        print(f"Coeficiente de Silueta ({label}): {silhouette:.4f}")
    else:
        print(
            f"No se puede calcular el coeficiente de silueta ({label}): solo hay un cluster o ruido."
        )


def reduce_dimensions_with_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return pca, reduced_data


def get_top_features(pca, num_features=2, feature_names=None):
    components = pca.components_
    feature_names = (
        feature_names
        if feature_names is not None
        else [f"Variable {i+1}" for i in range(components.shape[1])]
    )
    top_features = []
    for i in range(components.shape[0]):
        indices = np.argsort(np.abs(components[i]))[-num_features:][::-1]
        top_features.append([feature_names[j] for j in indices])
    return top_features


def summarize_labels(labels, label="Nuevos Datos"):
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"Etiquetas en {label}:", dict(zip(unique_labels, counts)))


def plot_clusters(data, labels, title, xlabel, ylabel):
    unique_labels = set(labels)

    palette = sns.color_palette("husl", len(unique_labels) - 1)  # Paleta para clusters
    colors = [
        palette[i] if k != -1 else (0.5, 0.5, 0.5) for i, k in enumerate(unique_labels)
    ]  # Gris para ruido

    # Crear gráfica
    plt.figure(figsize=(10, 8))
    for k, color in zip(unique_labels, colors):
        class_member_mask = labels == k
        xy = data[class_member_mask]
        plt.scatter(
            xy[:, 0],
            xy[:, 1],
            c=[color],
            label=f"Cluster {k}" if k != -1 else "Ruido",
            s=100,
            alpha=0.7,
            edgecolor="k",
        )

    # colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    # for k, col in zip(unique_labels, colors):
    #     if k == -1:  # Ruido
    #         col = [0, 0, 1, 1]

    #     class_member_mask = labels == k
    #     xy = data[class_member_mask]
    #     plt.plot(
    #         xy[:, 0],
    #         xy[:, 1],
    #         "o",
    #         markerfacecolor=tuple(col),
    #         markeredgecolor="k",
    #         markersize=10,
    #     )

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.legend(
        title="Grupos", loc="upper right", fontsize=10, title_fontsize=12, frameon=True
    )
    plt.grid(visible=True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    image_path = "images/" + title.replace(" ", "_")
    plt.savefig(image_path)
    # plt.show()
    print(f"Imagen {title} creada")


def visualize_clusters(data_pca, labels, top_features, title):
    xlabel = ", ".join(top_features[0])
    ylabel = ", ".join(top_features[1])
    plot_clusters(data_pca, labels, title=title, xlabel=xlabel, ylabel=ylabel)
