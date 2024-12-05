import os
import pandas as pd
import numpy as np
from logging_config import get_logger
from dotenv import load_dotenv
from data_connection import load_data
from sklearn.model_selection import train_test_split

logger = get_logger()

# Carga de variables de entorno desde el archivo .env
load_dotenv()


def create_data_path(name_path):
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

        # Definir rutas de salida
        output_files = {
            'train': os.path.join(output_directory, 'train_data.csv'),
            'validation': os.path.join(output_directory, 'validation_data.csv'),
            'test': os.path.join(output_directory, 'test_data.csv')
        }

        train_path = create_data_path('train_data.csv')
        val_path = create_data_path('validation_data.csv')
        test_path = create_data_path('test_data.csv')

        # Guardar cada conjunto en archivos separados
        save_data(train_data, train_path)  # 65% de los datos
        save_data(validation_data, val_path)  # 30% de los datos
        save_data(test_data, test_path)  # 5% de los datos

        # Confirmación final
        logger.info("Proceso de división de datos completado exitosamente")

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
