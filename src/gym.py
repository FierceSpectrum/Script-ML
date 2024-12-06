import os
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from data_connection import load_data
import ml_functions as ml
import seaborn as sns
import dbscan_functions as dbf


from data_functions import validate_env_variables, save_data, create_data_path

load_dotenv()


def extract_data_path(name_file):
    """
    Extrae la ruta guardada en las variables de entorno y añade el nombre de un archivo a leer en la carpeta "data"

    Parámetros:
        name_file (str): Nombre del archivo a leer en la carpeta "data"

    Retorna:
        Ruta completa hacia un archivo
    """

    validate_env_variables("directory_data")

    data_path = os.path.normpath(
        os.path.join(
            os.getenv("directory_project"),
            os.path.join(os.getenv("directory_data"), name_file),
        )
    )
    return data_path


def model_K_Means():

    # Valida la existencia de la variable de entorno antes de asignar su valor.
    validate_env_variables("n_clusters")
    n_clusters = int(os.getenv("n_clusters"))

    # Ruta del archivo de datos
    data_path = extract_data_path("train_data.csv")

    # Cargar los datos
    train_data = load_data(data_path)

    # Entrenar modelo K-Means
    (
        kmeans_model,
        inertia,
        labels,
        silhouette,
    ) = ml.train_kmeans(train_data, n_clusters=n_clusters)

    explanation_percentage = ml.calculate_inertia_percentage(train_data.values, inertia)

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
    save_data(df_with_clusters, create_data_path("kmeans_train_data.csv"))

    # # Visualizar el método del codo
    # ml.plot_elbow_method(train_data.values, max_clusters=10, "kmeans_elbow_method.png")

    # Validar el modelo con datos

    # Ruta del archivo de datos
    data_path = extract_data_path("validation_data.csv")

    # Cargar los datos
    validation_data = load_data(data_path)
    validation_labels = ml.predict_kmeans(validation_data.values, kmeans_model)

    # Asignar clusters a los datos originales
    df_with_clusters = ml.assign_clusters(validation_data.copy(), validation_labels)
    save_data(df_with_clusters, create_data_path("kmeans_validation_data.csv"))

    # ml.plot_original_data(validation_data.values, "kmeans_original_data_validation.png")

    # ml.visualize_clusters_3d(validation_data, validation_labels, kmeans_model)
    # ml.visualize_clusters_3d_matplotlib(validation_data, validation_labels, kmeans_model,  "kmeans_cluster_validation.png")

    silhouette = ml.calculate_silhouette_score(validation_data, validation_labels)
    print("silhouette")
    print(silhouette)

    # Testear el modelo con datos

    # Ruta del archivo de datos
    data_path = extract_data_path("test_data.csv")

    # Cargar los datos
    test_data = load_data(data_path)
    test_labels = ml.predict_kmeans(test_data.values, kmeans_model)

    # Asignar clusters a los datos originales
    df_with_clusters = ml.assign_clusters(test_data.copy(), test_labels)
    save_data(df_with_clusters, create_data_path("kmeans_test_data.csv"))

    # Se aplica modelo para medir la calidad de los cluster
    silhouette = ml.calculate_silhouette_score(test_data, test_labels)
    print("silhouette")
    print(silhouette)

    return


def model_DBSCAN():
    # Archivos de entrada
    train_file = extract_data_path("train_data.csv")
    validation_file = extract_data_path("validation_data.csv")
    test_file = extract_data_path("test_data.csv")

    # Cargar datos
    train_data = load_data(train_file)
    validation_data = load_data(validation_file)
    test_data = load_data(test_file)

    # Configuración de DBSCAN
    # ? Para una división más detallada de los datos, el modelo con eps=0.11 y min_samples=4 (2 clusters) parece ser más adecuado
    # ? Ya que muestra una segmentación clara sin perder demasiados puntos como ruido.
    # dbscan = dbf.configure_dbscan(eps=0.11, min_samples=4)

    # ? Para una agrupación más amplia, el modelo con eps=0.12 y min_samples=4 (1 único cluster) puede ser más útil
    # ? Especialmente si los datos son más homogéneos y puedes trabajar con una agrupación general.
    dbscan = dbf.configure_dbscan(eps=0.12, min_samples=4)

    # Ajustar y evaluar en datos de entrenamiento
    train_labels = dbf.fit_dbscan(dbscan, train_data)
    dbf.evaluate_model(train_labels, train_data, label="Entrenamiento")

    # Reducción de dimensiones para visualización
    pca, train_data_pca = dbf.reduce_dimensions_with_pca(train_data)
    feature_names = train_data.columns if hasattr(train_data, "columns") else None
    top_features = dbf.get_top_features(pca, feature_names=feature_names)
    dbf.visualize_clusters(
        train_data_pca, train_labels, top_features, "Clusters en Datos de Entrenamiento"
    )

    # Evaluación en datos de validación
    validation_labels = dbf.fit_dbscan(dbscan, validation_data)
    dbf.evaluate_model(validation_labels, validation_data, label="Validación")
    validation_data_pca = pca.transform(validation_data)
    dbf.visualize_clusters(
        validation_data_pca,
        validation_labels,
        top_features,
        "Clusters en Datos de Validación",
    )

    # Predicción y evaluación en nuevos datos
    new_labels = dbf.fit_dbscan(dbscan, test_data)
    new_data_pca = pca.transform(test_data)
    dbf.visualize_clusters(
        new_data_pca, new_labels, top_features, "Clusters en Nuevos Datos"
    )
    dbf.summarize_labels(new_labels, label="Nuevos Datos")


def main():
    model_K_Means()
    model_DBSCAN()
    return


if __name__ == "__main__":
    main()
