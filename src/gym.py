import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_connection import load_data
import ml_functions as ml


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
    kmeans_model, inertia, labels, silhouette, = ml.train_kmeans(
        train_data, n_clusters=n_clusters)

    explanation_percentage = ml.calculate_inertia_percentage(
        train_data.values, inertia)

    # Evaluar modelo
    ml.evaluate_kmeans(inertia, silhouette)
    ml.evaluate_kmeans(explanation_percentage, silhouette)

    # Visualizar datos originales (normalizados)
    # ml.plot_original_data(train_data.values, "kmeans_original_data_test.png")

    # Visualizar clusters en 2D
    # ml.visualize_clusters_3d(train_data, labels, kmeans_model)
    # ml.visualize_clusters_3d_matplotlib(train_data, labels, kmeans_model, "kmeans_cluster_train.png")

    # Asignar clusters a los datos originales
    df_with_clusters = ml.assign_clusters(train_data.copy(), labels)
    save_data(df_with_clusters, create_data_path('kmeans_train_data.csv'))

    # # Visualizar el método del codo
    # ml.plot_elbow_method(train_data.values, max_clusters=10, "kmeans_elbow_method.png")

    # Validar el modelo con datos

    # Ruta del archivo de datos
    data_path = extract_data_path('validation_data.csv')

    # Cargar los datos
    validation_data = load_data(data_path)
    validation_labels = ml.predict_kmeans(validation_data.values, kmeans_model)

    # Asignar clusters a los datos originales
    df_with_clusters = ml.assign_clusters(validation_data.copy(), validation_labels)
    save_data(df_with_clusters, create_data_path('kmeans_validation_data.csv'))

    # ml.plot_original_data(validation_data.values, "kmeans_original_data_validation.png")

    # ml.visualize_clusters_3d(validation_data, validation_labels, kmeans_model)
    # ml.visualize_clusters_3d_matplotlib(validation_data, validation_labels, kmeans_model,  "kmeans_cluster_validation.png")

    silhouette = ml.calculate_silhouette_score(validation_data, validation_labels)
    print("silhouette")
    print(silhouette)

    # Testear el modelo con datos

    # Ruta del archivo de datos
    data_path = extract_data_path('test_data.csv')

    # Cargar los datos
    test_data = load_data(data_path)
    test_labels = ml.predict_kmeans(test_data.values, kmeans_model)

    # Asignar clusters a los datos originales
    df_with_clusters = ml.assign_clusters(test_data.copy(), test_labels)
    save_data(df_with_clusters, create_data_path('kmeans_test_data.csv'))

    silhouette = ml.calculate_silhouette_score(test_data, test_labels)
    print("silhouette")
    print(silhouette)

    return


def model_DBSCAN():
    return


def main():
    model_K_Means()
    return


if __name__ == "__main__":
    main()
