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


def save_data(data, file_path):
    """
    Guarda un DataFrame en un archivo CSV y confirma la acción.
    """
    try:
        data.to_csv(file_path, index=False)
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
    """
    Filtra las columnas de un DataFrame para incluir solo las necesarias según el archivo .env.
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

    if not test_size:
        # Validar que las variables de entorno necesarias estén definidas
        validate_env_variables('random_state')

        test_size = os.getenv('random_state')

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
    try:

        # Validar que las variables de entorno necesarias estén definidas
        validate_env_variables('file_data', 'file_data2', 'directory_data')

        # Cargar las rutas de los datos
        file_path = os.getenv('file_data')
        backup_url = os.getenv('file_data2')
        output_directory = os.getenv('directory_data')

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
        train_path = os.path.join(output_directory, 'train_data.csv')
        val_path = os.path.join(output_directory, 'validation_data.csv')
        test_path = os.path.join(output_directory, 'test_data.csv')

        # Guardar cada conjunto en archivos separados
        save_data(train_data, train_path)  # 65% de los datos
        save_data(validation_data, val_path)  # 30% de los datos
        save_data(test_data, test_path)  # 5% de los datos

        # Confirmación final
        logger.info("Proceso de división de datos completado exitosamente")

        # print("\nDatos divididos y guardados exitosamente:")
        # print(f"Entrenamiento: {train_data.shape}")
        # print(f"Validación: {validation_data.shape}")
        # print(f"Prueba: {test_data.shape}")

    except Exception as e:
        logger.error(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
