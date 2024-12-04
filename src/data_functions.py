import os
from dotenv import load_dotenv
from data_connection import load_data
from sklearn.model_selection import train_test_split

# Carga de variables de entorno desde el archivo .env
load_dotenv()


def create_file(data, file_path):
    """
    Guarda un DataFrame en un archivo CSV y confirma la acción.
    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Archivo creado exitosamente en: {file_path}")
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


def main():
    try:

        # Validar que las variables de entorno necesarias estén definidas
        validate_env_variables('file_data', 'file_data2', 'directory_data')

        # Cargar el archivo de datos
        file_path  = os.getenv('file_data')
        backup_url = os.getenv('file_data2')
        directory_data = os.getenv('directory_data')

        # Verificar si el archivo de datos existe
        if not os.path.exists(file_path):
            print(f"El archivo de datos no existe en: {file_path}")
            print("Se procederá a descargar los datos desde el enlace de respaldo...")
            create_file(load_data(backup_url), file_path)
            print(f"Datos descargado y guardado en: {file_path}")

        # Verificar nuevamente si el archivo ahora existe
        if not os.path.exists(file_path ):
            raise FileNotFoundError(
                f"El archivo de datos no existe en: {file_path }")

        # Cargar los datos desde el archivo
        datos = load_data(file_path )

        # División inicial: 65% para entrenamiento y 35% para validación + prueba
        train_data, temp_data = train_test_split(
            datos, test_size=0.35, random_state=42)

        # División del conjunto temporal: 30% para validación y 5% para prueba
        validation_data, test_data = train_test_split(
            temp_data, test_size=(5/35), random_state=42)

        # Validar si el directorio existe o crearlo
        if not os.path.exists(directory_data):
            os.makedirs(directory_data)
            print(f"Directorio creado: {directory_data}")

        # Guardar cada conjunto en archivos separados
        create_file(train_data, os.path.join(
            directory_data, "train_data.csv"))  # 65% de los datos

        create_file(validation_data, os.path.join(
            directory_data, "validation_data.csv"))  # 30% de los datos

        create_file(test_data, os.path.join(
            directory_data, "test_data.csv"))  # 5% de los datos


        print("\nDatos divididos y guardados exitosamente:")
        print(f"Entrenamiento: {train_data.shape}")
        print(f"Validación: {validation_data.shape}")
        print(f"Prueba: {test_data.shape}")

    except Exception as e:
        print(f"Error durante la ejecución: {e}")


if __name__ == "__main__":
    main()
