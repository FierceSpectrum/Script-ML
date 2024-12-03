import pandas as pd

def load_data(url):
    """
    Carga el conjunto de datos desde la URL pública de GitHub.
    
    Parámetros:
        url (str): URL directa del archivo CSV en el repositorio de GitHub.
        
    Retorna:
        DataFrame de pandas con los datos cargados.
    """
    try:
        data = pd.read_csv(url)
        # data = pd.read_csv(url, error_bad_lines=False, warn_bad_lines=True)
        # data = pd.read_csv(url, on_bad_lines='skip')
        print("Datos cargados exitosamente.")
        return data
    except Exception as e:
        print(f"Erro al cargar los datos: {e}")
        return None