import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_functions as df
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering, AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn.metrics import silhouette_score
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from data_connection import load_data
from logging_config import get_logger

logger = get_logger()
load_dotenv()


def extract_data_path(name_file):
    """
    Extrae la ruta completa de un archivo de datos usando variables de entorno.
    """
    df.validate_env_variables('directory_data')
    data_path = os.path.normpath(os.path.join(os.getenv(
        'directory_project'), os.path.join(os.getenv('directory_data'), name_file)))
    return data_path


def create_directory(rute_model):
    """
    Crea el directorio necesario para guardar los resultados del modelo.
    """
    directory_proyect = os.getcwd()
    for dr in ['images', 'data']:
        directory = os.path.normpath(os.path.join(
            directory_proyect, f"{dr}/{rute_model}"))
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directorio creado: {directory}")
    return directory


def process_model(data, labels, model_name, rute_model, centroids, data_type):
    """
    Procesa un conjunto de datos, visualiza los resultados y guarda las métricas.
    """
    print(f"\nProcesando datos para {data_type}...\n")

    # Calcular el coeficiente de silhouette
    # labels deben ser las etiquetas, no el modelo
    silhouette = silhouette_score(data, labels)
    print(f"Silhouette score para {model_name}: {silhouette:.2f}")

    # Asignar etiquetas y guardar resultados
    df_with_clusters = df.assign_clusters(data.copy(), labels)
    df.save_data(df_with_clusters, df.create_data_path(
        f'{rute_model}/{data_type}_data.csv'))
    
    df.plot_original_data(data, f"original_data_{data_type}.png")

    # Visualizar los clusters en 3D
    df.visualize_clusters_3d_matplotlib(data, labels, centroids, f"{
                                         rute_model}/cluster_{data_type}.png", title=f"Distribución de clusters - {model_name}")

    return silhouette


def get_labels_and_centroids_fit(model_name, model, data, n_clusters):
    """
    Ajusta el modelo de clustering y obtiene las etiquetas y los centroides.
    """
    if model_name == "Gaussian Mixture Model":
        labels = model.fit_predict(data)
        centroids = model.means_
    elif model_name == "K-Means Clustering":
        model.fit(data)
        labels = model.labels_
        centroids = model.cluster_centers_
    elif model_name == "MiniBatchKMeans":
        labels = model.fit_predict(data)
        centroids = model.cluster_centers_
    else:
        labels = model.fit_predict(data)
        centroids = np.array([data[labels == i].mean(axis=0)
                             for i in range(n_clusters)])

    return labels, centroids


def get_labels_and_centroids_predict(model_name, model, data, n_clusters):
    """
    Ajusta el modelo de clustering y obtiene las etiquetas y los centroides.
    """
    if model_name == "Gaussian Mixture Model":
        labels = model.predict(data)
        centroids = model.means_
    elif model_name in ["K-Means Clustering", "MiniBatchKMeans"]:
        labels = model.predict(data)
        centroids = model.cluster_centers_
    else:
        labels = model.fit_predict(data)
        centroids = np.array([data[labels == i].mean(axis=0)
                             for i in range(n_clusters)])

    return labels, centroids


def process_clustering_model(data, model, model_name, rute_model, n_clusters=None):
    """
    Función para ajustar un modelo de clustering y calcular métricas, guardar resultados y visualizar.
    """

    # Ajustar el modelo
    print(f"\nAjustando el modelo {model_name}...\n")

    if not n_clusters:
        df.validate_env_variables('n_clusters')
        n_clusters = int(os.getenv('n_clusters'))

    # Ajustar el modelo y obtener etiquetas y centroides
    labels, centroids = get_labels_and_centroids_fit(
        model_name, model, data, n_clusters)

    # Procesar los datos de entrenamiento, validación y prueba
    process_model(data, labels, model_name, rute_model,
                  centroids, data_type="train")

    # Procesar datos de validación
    validation_data_path = extract_data_path('validation_data.csv')
    validation_data = load_data(validation_data_path)
    labels_v, centroids = get_labels_and_centroids_predict(
        model_name, model, validation_data, n_clusters)
    process_model(validation_data, labels_v, model_name,
                  rute_model, centroids, data_type="validation")

    # Procesar datos de prueba
    test_data_path = extract_data_path('test_data.csv')
    test_data = load_data(test_data_path)
    labels_t, centroids = get_labels_and_centroids_predict(
        model_name, model, test_data, n_clusters)
    process_model(test_data, labels_t, model_name,
                  rute_model, centroids, data_type="test")

    return labels, labels_v, labels_t


def model_clustering():
    """
    Entrena, evalúa y visualiza varios modelos de clustering (GMM, Spectral Clustering, Agglomerative y MiniBatchKMeans).
    """
    # Ruta del archivo de datos y carga
    data_path = extract_data_path('train_data.csv')
    train_data = load_data(data_path)

    # Obtener parámetros de entorno
    df.validate_env_variables('random_state', 'n_clusters')
    random_state = int(os.getenv('random_state'))
    n_clusters = int(os.getenv('n_clusters'))

    # Modelos de clustering
    models = {
        "K-Means Clustering": KMeans(n_clusters=n_clusters, random_state=random_state),
        "Gaussian Mixture Model": GaussianMixture(n_components=n_clusters, random_state=random_state),
        "Spectral Clustering": SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=random_state),
        "Agglomerative Clustering": AgglomerativeClustering(n_clusters=n_clusters),
        "MiniBatchKMeans": MiniBatchKMeans(n_clusters=n_clusters, random_state=random_state, batch_size=100)
    }

    model_rute = {
        "K-Means Clustering": "kmeans",
        "Gaussian Mixture Model": "gaussian",
        "Spectral Clustering": "spectral",
        "Agglomerative Clustering": "agglomerative",
        "MiniBatchKMeans": "minikmeans"
    }

    # Procesar cada modelo
    models_results = {}  # Initialize this variable to store results
    for model_name, model in models.items():
        labels, labels_t, labels_v = process_clustering_model(
            train_data, model, model_name, model_rute[model_name])
        models_results[model_name] = labels

    # Visualizar el método del codo (solo para datos de entrenamiento)
    df.plot_elbow_method(train_data.values, max_clusters=10,
                          name_img="elbow_method.png")

    print("\nModelado de Clustering completado.\n")

    # Evaluación de métricas y comparación
    metrics = []
    for model_name, labels in models_results.items():
        silhouette_avg = silhouette_score(train_data, labels)
        cluster_counts = pd.Series(labels).value_counts().sort_index()

        metrics.append({
            "Model": model_name,
            "Silhouette Score": silhouette_avg,
            "Cluster Counts": cluster_counts.tolist()
        })

    # Mostrar métricas en tabla
    metrics_df = pd.DataFrame(metrics)
    print("\n### Comparación de resultados ###")
    print(metrics_df)


def main():
    model_clustering()


if __name__ == "__main__":
    main()
