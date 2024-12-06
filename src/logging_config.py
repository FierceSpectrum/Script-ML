import logging
import inspect


def get_logger():

    """
    Asigna las etiquetas de los cluster al conjunto de datos.

    Parámetros:
        

    Retorna:
        El objeto "logger" inicializado
    """

    caller_frame = inspect.stack()[1]
    caller_file = caller_frame.filename.split("\\")[-1]

    logger = logging.getLogger(caller_file)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] : %(message)s'
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# def quien_me_llama():
#     stack = inspect.stack()
#     caller_frame = stack[1]  # El que llama está en el índice 1
#     caller_file = caller_frame.filename  # Archivo desde el que se llamó
#     # Nombre de la función que hizo la llamada
#     caller_function = caller_frame.function
#     return f"Llamado desde el archivo: {caller_file}, función: {caller_function}"

# # Otro archivo o función que llame a esta


# def otra_funcion():
#     print(quien_me_llama())


# # Configurar logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s: %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )


# logger = logging.getLogger(__name__)
