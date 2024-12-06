import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from logging_config import get_logger
from dotenv import load_dotenv
from data_connection import load_data
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

logger = get_logger()

# Carga de variables de entorno desde el archivo .env
load_dotenv()


def create_data_path(name_path):
    """
    Valida y crea la ruta hacia un archivo en la carpeta de "data".

    Parámetros:
        name_path (str): nombre del archivo a abrir o leer.

    Retorna:
        Ruta completa hacia el archivo dentro del proyecto.
    """

    validate_env_variables('directory_project', 'directory_data')

    directory_project = os.getenv('directory_project')

    output_directory = os.path.normpath(
        os.path.join(
            directory_project,
            os.getenv('directory_data')
        )
    )

    data_path = os.path.join(output_directory, name_path)
    return data_path


def save_data(data, file_path, **kwargs):
    """
    Guarda un DataFrame en un archivo CSV y confirma la acción.
    """
    try:
        # Crear el directorio si no existe
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directorio creado: {directory}")

        data.to_csv(file_path, index=False, **kwargs)
        logger.info(f"Archivo guardado: {file_path}")
        logger.info(f"Tamaño del conjunto: {data.shape}")
    except Exception as e:
        print(f"Error al guardar el archivo en {file_path}: {e}")


def validate_env_variables(*variables):
    """
    Valida que todas las variables de entorno necesarias estén definidas.
    """
    for var in variables:
        if not os.getenv(var):
            raise EnvironmentError(
                f"Falta definir la variable de entorno: {var}")


def filter_columns(data):
    """"
    Filtra las columnas de un DataFrame para incluir solo las necesarias según el archivo .env
    y elimina la primera columna.

    Args:
        data (pd.DataFrame): DataFrame original a filtrar

    Returns:
        pd.DataFrame: DataFrame filtrado
    """

    try:
        # Validar que las variables de entorno necesarias estén definidas
        validate_env_variables("required_columns")

        columns_str = os.getenv("required_columns")

        required_columns = [col.strip() for col in columns_str.split(",")]

        missing_columns = [
            col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(
                f"Las siguientes columnas necesarias no están en los datos: {missing_columns}")

        # Eliminar la primera columna y filtrar por las columnas necesarias
        return data[required_columns]

    except Exception as e:
        raise Exception(f"Error al filtrar columnas: {e}")


def split_data(data, train_size=0.65, val_size=0.30, test_size=0.05, random_state=None):
    """
    Dividir datos en conjuntos de entrenamiento, validación y prueba.

    Args:
        data (pd.DataFrame): Conjunto de datos original
        train_size (float): Proporción para entrenamiento
        val_size (float): Proporción para validación
        test_size (float): Proporción para prueba
        random_state (int): Semilla para reproducibilidad

    Returns:
        Tuple con conjuntos de train, validation y test
    """

    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Las proporciones de división deben sumar 1.0")

    if not random_state:
        # Validar que las variables de entorno necesarias estén definidas
        validate_env_variables('random_state')

        random_state = int(os.getenv('random_state'))

    # División inicial: 65% para entrenamiento y 35% para validación + prueba
    train_data, temp_data = train_test_split(
        data,
        test_size=(val_size + test_size),
        random_state=random_state)

    # División del conjunto temporal: 30% para validación y 5% para prueba
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=(test_size / (val_size + test_size)),
        random_state=random_state)

    return train_data, validation_data, test_data


def create_image_path(name_path):
    """
    Crea la ruta completa donde se guardada las imagenes creadas mediante la libreria matplotlib.

    Parámetros:
        name_path (str): Nombre de la imagen a guardar.

    Retorna:
        Ruta hacia la carpeta "images" con el nombre de la imagen.
    """

    validate_env_variables('directory_image')
    image_path = os.path.normpath(os.path.join(
        os.getenv('directory_image'), name_path))

    return image_path


def plot_original_data(df, name_img="original_data_train.png"):
    """
    Grafico 3D original, el cual mantiene la estructura original de como se verian los datos.

    Parámetros:
        df: DataFrame de los datos agrupados por el modelo K-Means.
        name_img: Nombre de la imagen que recibida el grafico al ser guardado.

    Retorna:
        Ejecuta una funcion ".show", donde muestra en una ventana un grafico 3D con los clusters agrupados.
    """

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


def assign_clusters(df, labels):
    """
    Asigna las etiquetas de los cluster al conjunto de datos.

    Parámetros:
        df: DataFrame de los datos evaluados.
        labels: Etiquetas de los datos evaluados.

    Retorna:
        Conjunto de datos con las etiquedas añadidas en la columna "Cluster".
    """

    df["Cluster"] = labels
    return df


def visualize_clusters_3d(data, labels, kmeans):

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


def visualize_clusters_3d_matplotlib(data, labels, centroids, name_img="cluster_3d_train.png", title="Clusters visualizados en 3D"):
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
    ax.set_title(title)
    ax.set_xlabel("total_day_minutes")
    ax.set_ylabel("total_evening_minutes")
    ax.set_zlabel("total_night_minutes")

    # Mostrar leyenda
    ax.legend()
    plt.grid(True)
    plt.savefig(create_image_path(name_img))
    plt.show()


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


def main():
    """
    Función principal para cargar, procesar y dividir datos.
    Maneja la carga desde archivo principal o respaldo, filtrado de columnas 
    y división de datos en conjuntos de entrenamiento, validación y prueba.
    """

    try:

        """
        Función principal para cargar, procesar y dividir datos.
        Maneja la carga desde archivo principal o respaldo, filtrado de columnas 
        y división de datos en conjuntos de entrenamiento, validación y prueba.
        """

        # Validar variables de entorno
        validate_env_variables(
            'file_data',
            'file_data2',
            'file_data3',
            'directory_data',
            'directory_project'
        )

        # Cargar las rutas de los datos
        directory_project = os.getenv('directory_project')

        output_directory = os.path.normpath(
            os.path.join(
                directory_project,
                os.getenv('directory_data')
            )
        )

        file_path = os.path.normpath(
            os.path.join(
                directory_project,
                os.getenv('file_data')
            )
        )

        backup_url = os.getenv('file_data2')

        # Crear directorio de salida si no existe
        os.makedirs(output_directory, exist_ok=True)
        logger.info(f"Directorio de salida: {output_directory}")

        # Verificar y cargar datos
        if not os.path.exists(file_path):
            logger.warning(f"Archivo no encontrado en: {file_path}")
            logger.info("Descargando datos desde enlace de respaldo...")
            save_data(load_data(backup_url), file_path)

        # Cargar datos
        datos = load_data(file_path)

        # Filtrar columnas
        datos = filter_columns(datos)

        # Dividir datos
        train_data, validation_data, test_data = split_data(datos)

        train_path = create_data_path('train_data.csv')
        val_path = create_data_path('validation_data.csv')
        test_path = create_data_path('test_data.csv')

        # Guardar cada conjunto en archivos separados
        save_data(train_data, train_path)  # 65% de los datos
        save_data(validation_data, val_path)  # 30% de los datos
        save_data(test_data, test_path)  # 5% de los datos

        # Cargar datos
        datos = load_data(os.getenv('file_data3'))

        # Filtrar columnas
        datos = filter_columns(datos)

        # Dividir datos
        train_data2, validation_data2, test_data2 = split_data(datos)

        train_path2 = create_data_path('train_data2.csv')
        val_path2 = create_data_path('validation_data2.csv')
        test_path2 = create_data_path('test_data2.csv')

        # Guardar cada conjunto en archivos separados
        save_data(train_data2, train_path2)  # 65% de los datos
        save_data(validation_data2, val_path2)  # 30% de los datos
        save_data(test_data2, test_path2)  # 5% de los datos

        # Confirmación final
        logger.info("Proceso de división de datos completado exitosamente")

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
