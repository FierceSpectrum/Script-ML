import pandas as pd
from logging_config import get_logger
logger = get_logger()

def load_data(file_url):
    """
    Carga el conjunto de datos desde la URL pública de GitHub.
    
    Parámetros:
        url (str): URL directa del archivo CSV en el repositorio de GitHub.
        
    Retorna:
        DataFrame de pandas con los datos cargados.
    """
    try:
        data = pd.read_csv(file_url)
        # data = pd.read_csv(url, error_bad_lines=False, warn_bad_lines=True)
        # data = pd.read_csv(url, on_bad_lines='skip')
        logger.info(f"Datos cargados exitosamente desde: {file_url}")
        return data
    except Exception as e:
        raise FileNotFoundError(f"Erro al cargar los datos: {e}")