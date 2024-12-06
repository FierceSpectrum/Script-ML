import os
import matplotlib.pyplot as plt
import numpy as np

from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import plotly.graph_objects as go

from data_functions import validate_env_variables

load_dotenv()


def create_image_path(name_path):
    validate_env_variables('directory_image')
    image_path = os.path.normpath(os.path.join(
        os.getenv('directory_image'), name_path))
    return image_path


def calculate_silhouette_score(df, labels):
    silhouette = silhouette_score(df, labels)
    return silhouette


def train_kmeans(data, n_clusters=None):
    """
    Entrena un modelo K-Means Clustering.

    :param data: DataFrame con los datos normalizados para entrenamiento.
    :param n_clusters: Número de clusters a configurar.
    :return: Modelo entrenado, inertia, silhouette score.
    """

    validate_env_variables('random_state', 'n_clusters')
    random_state = int(os.getenv('random_state'))

    if not n_clusters:
        n_clusters = int(os.getenv('n_clusters'))

    # Inicializar el modelo
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Entrenar el modelo
    kmeans.fit(data)

    # Evaluar el modelo
    inertia = kmeans.inertia_  # Metrica de calidad del modelo
    labels = kmeans.labels_
    silhouette = calculate_silhouette_score(
        data, labels)  # Puntuacion de Silhouette

    return kmeans, inertia, labels, silhouette


def predict_kmeans(df, kmeans):

    validation_labels = kmeans.predict(df)

    return validation_labels


def evaluate_kmeans(inertia, silhouette):
    """
    Evalúa el modelo K-Means mostrando las métricas relevantes.

    :param inertia: Inertia del modelo entrenado.
    :param silhouette: Puntuación de Silhouette del modelo.
    """
    print("Evaluacion del modelo K-Means:")
    print(f"- Inertia: {inertia}")
    print(f"- Silhouette Score: {silhouette}")

    # Agregar logica para justar cluster si es necesatio
    if silhouette < 0.5:
        print("Considera aumentar o reducir el numero de clusters.")


def plot_elbow_method(data, max_clusters=10, name_img="elbow_method.png"):
    """
    Grafica el método del codo para encontrar el número óptimo de clusters.
    :param data: DataFrame con los datos normalizados.
    :param max_clusters: Número máximo de clusters a probar.
    """
    validate_env_variables('random_state')
    random_state = int(os.getenv('random_state'))

    inertias = []

    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_clusters + 1), inertias, marker='o', linestyle='--')
    plt.title("Método del Codo")
    plt.xlabel("Número de Clusters")
    plt.ylabel("Inercia")
    plt.xticks(range(1, max_clusters + 1))
    plt.grid(True)
    plt.savefig(create_image_path(name_img))
    plt.show()


def plot_original_data(df, name_img="original_data_train.png"):
    graf = plt.figure(figsize=(10, 8))
    ax = graf.add_subplot(111, projection='3d')
    scatter = ax.scatter(df[:, 0], df[:, 1], df[:, 2],
                         c="blue", s=50, label="Datos originales")

    ax.set_title("Datos Originales")
    ax.set_xlabel("total_day_minutes")
    ax.set_ylabel("total_evening_minutes")
    ax.set_zlabel("total_night_minutes")

    plt.legend()
    plt.grid(True)
    plt.savefig(create_image_path(name_img))
    plt.show()


def visualize_clusters_3d_matplotlib(data, labels, kmeans, name_img="cluster_3d_train.png"):
    """
    Visualiza clusters en 3D con matplotlib.

    :param data: DataFrame o matriz con al menos 3 columnas (total_day_minutes, total_evening_minutes, total_night_minutes).
    :param labels: Etiquetas de los clusters asignadas por K-Means.
    :param kmeans: Modelo K-Means entrenado para acceder a los centroides.
    """
    if data.shape[1] < 3:
        raise ValueError(
            "El conjunto de datos debe tener al menos 3 columnas para graficar en 3D.")

    # Configuración del gráfico
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Obtener los centroides
    centroids = kmeans.cluster_centers_

    # Agregar puntos de cada cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

    for cluster_id, color in zip(unique_labels, colors):
        # Filtrar puntos del cluster actual
        cluster_data = data[labels == cluster_id]
        ax.scatter(
            cluster_data.iloc[:, 0],
            cluster_data.iloc[:, 1],
            cluster_data.iloc[:, 2],
            c=[color],
            label=f"Cluster {cluster_id}",
            s=50,
            alpha=0.15
        )

    # Agregar centroides
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c='red',
        marker='X',
        s=100,
        label="Centroides"
    )

    # Configurar etiquetas y título
    ax.set_title("Clusters visualizados en 3D")
    ax.set_xlabel("total_day_minutes")
    ax.set_ylabel("total_evening_minutes")
    ax.set_zlabel("total_night_minutes")

    # Mostrar leyenda
    ax.legend()
    plt.grid(True)
    plt.savefig(create_image_path(name_img))
    plt.show()


def visualize_clusters_3d(data, labels, kmeans):
    """
    Visualiza los clusters en 3D con interactividad.

    :param data: DataFrame o matriz con al menos 3 columnas (total_day_minutes, total_evening_minutes, total_night_minutes).
    :param labels: Etiquetas de los clusters asignadas por K-Means.
    :param kmeans: Modelo K-Means entrenado para acceder a los centroides.
    """
    if data.shape[1] < 3:
        raise ValueError(
            "El conjunto de datos debe tener al menos 3 columnas para graficar en 3D.")

    # Crear figura
    fig = go.Figure()

    # Obtener los centroides
    centroids = kmeans.cluster_centers_

    # Agregar puntos de cada cluster
    unique_labels = np.unique(labels)
    for cluster_id in unique_labels:
        # Filtrar puntos del cluster actual
        cluster_data = data[labels == cluster_id]
        fig.add_trace(
            go.Scatter3d(
                x=cluster_data.iloc[:, 0],
                y=cluster_data.iloc[:, 1],
                z=cluster_data.iloc[:, 2],
                mode='markers',
                marker=dict(size=5, opacity=0.7),
                name=f"Cluster {cluster_id}",
                hoverinfo="x+y+z+text",
            )
        )

    # Agregar centroides
    fig.add_trace(
        go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode='markers+text',
            marker=dict(size=10, color='red', symbol='cross'),
            name="Centroides",
            text=[f"Centroide {i}" for i in range(len(centroids))],
            hoverinfo="text"
        )
    )

    # Configurar el diseño
    fig.update_layout(
        title="Clusters visualizados en 3D (interactivo)",
        scene=dict(
            xaxis_title="total_day_minutes",
            yaxis_title="total_evening_minutes",
            zaxis_title="total_night_minutes"
        ),
        legend=dict(
            x=0.1,
            y=0.9,
            title="Leyenda"
        )
    )

    # Mostrar la figura
    fig.show()

    # graf = plt.figure(figsize=(12, 8))
    # ax = graf.add_subplot(111, projection='3d')

    # # Graficar puntos de datos
    # # scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2],
    # #                      c=labels, cmap='viridis', s=50, alpha=0.7, label="Clientes")

    # # Graficar centroides
    # ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2],
    #            s=200, c='red', marker='X', label="Centoides")

    # # Etiquetas y titulo
    # ax.set_title("Clusters visualizados en 3D")
    # ax.set_xlabel("total_day_minutes")
    # ax.set_ylabel("total_evening_minutes")
    # ax.set_zlabel("total_night_minutes")

    # # Agregar colorbar
    # # cbar = graf.colorbar(ax=ax, pad=0.1)
    # # cbar.set_label("Cluster ID", fontsize=12)

    # # Configurar leyenda y grid
    # ax.legend(fontsize=12)
    # ax.grid(True)

    # graf.colorbar(scatter, ax=ax, label="Cluster ID")

    # Guardar y mostrar la imagen
    # plt.savefig(create_image_path('clusters_3d.png'))
    # plt.show()


def assign_clusters(df, labels):
    df["Cluster"] = labels
    return df


def calculate_inertia_percentage(data, kmeans_inertia):
    """
    Calcula el porcentaje de explicación basado en la inertia total.

    :param data: Matriz de datos original (numpy array).
    :param kmeans_inertia: Inertia del modelo K-Means entrenado.
    :return: Porcentaje de explicación.
    """
    # Calcular el centroide global
    global_centroid = np.mean(data, axis=0)

    # Calcular la inertia total (distancia al centroide global)
    total_inertia = np.sum(np.sum((data - global_centroid) ** 2, axis=1))

    # Calcular el porcentaje de explicación
    explanation_percentage = (1 - (kmeans_inertia / total_inertia)) * 100
    return explanation_percentage


def mean_distance_to_centroids(data, labels, kmeans):
    distances = []
    for cluster_id in range(kmeans.n_clusters):
        cluster_points = data[labels == cluster_id]
        centroid = kmeans.cluster_centers_[cluster_id]
        distances.append(np.mean(np.linalg.norm(
            cluster_points - centroid, axis=1)))
    return distances
