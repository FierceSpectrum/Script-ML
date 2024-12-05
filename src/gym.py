import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_connection import load_data
from ml_functions import (
    train_kmeans, evaluate_kmeans,
    visualize_clusters, plot_elbow_method,
    plot_original_data, visualize_clusters_2d,
    assign_clusters, visualize_clusters_tsne)


from data_functions import validate_env_variables, save_data, create_data_path

load_dotenv()


def extract_data_path(name_file):
    validate_env_variables('directory_data')

    data_path = os.path.normpath(os.path.join(os.getenv(
        'directory_project'), os.path.join(os.getenv('directory_data'), name_file)))
    return data_path


def model_K_Means():

    validate_env_variables('n_clusters')
    n_clusters = int(os.getenv('n_clusters'))

    # Ruta del archivo de datos
    data_path = extract_data_path('train_data.csv')
    
    # Cargar los datos
    train_data = load_data(data_path)

    # Entrenar modelo K-Means
    kmeans_model, inertia, labels, silhouette, = train_kmeans(
        train_data, n_clusters=n_clusters)

    # Evaluar modelo
    evaluate_kmeans(inertia, silhouette)

    # Visualizar datos originales (normalizados)
    plot_original_data(train_data.values)

    # Visualizar clusters
    visualize_clusters(kmeans_model, train_data.values)

    # Visualizar clusters en 2D
    visualize_clusters_2d(train_data, labels, kmeans_model)

    # Asignar clusters a los datos originales
    df_with_clusters = assign_clusters(train_data.copy(), labels)

    visualize_clusters_tsne(train_data, labels, kmeans_model)

    save_data(df_with_clusters, create_data_path('df_with_cluster.csv'))

    # # Visualizar el método del codo
    plot_elbow_method(train_data.values, max_clusters=10)

    return


def model_DBSCAN():
    return


def main():
    model_K_Means()
    return


if __name__ == "__main__":
    main()
