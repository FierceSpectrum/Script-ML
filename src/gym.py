import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_connection import load_data
from ml_functions import (
    train_kmeans, evaluate_kmeans,
    visualize_clusters, plot_elbow_method)

load_dotenv()


def create_image_path(name_path):
    image_path = os.path.normpath(os.path.join(
        os.getenv('directory_image'), name_path))
    return image_path


def extract_data_path(name_file):
    data_path = os.path.normpath(os.path.join(os.getenv(
        'directory_project'), os.path.join(os.getenv('directory_data'), name_file)))
    return data_path


def model_K_Means():
    # Ruta del archivo de datos
    data_path = extract_data_path('train_data.csv')

    # Cargar los datos
    train_data = load_data(data_path)

    # Entrenar modelo K-Means
    kmeans_model, inertia, silhouette, X_train = train_kmeans(
        train_data, n_clusters=10)

    # Evaluar modelo
    evaluate_kmeans(inertia, silhouette)

    # Visualizar clusters
    visualize_clusters(kmeans_model, X_train)

    # Visualizar el método del codo
    plot_elbow_method(train_data, max_clusters=10)

    return


def model_DBSCAN():
    return


def main():
    model_K_Means()
    return


if __name__ == "__main__":
    main()
