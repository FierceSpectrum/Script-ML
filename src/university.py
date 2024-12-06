import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import data_functions as df
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, auc, mean_squared_error, r2_score
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from data_connection import load_data
from logging_config import get_logger

logger = get_logger()
load_dotenv()

def extract_data_path(name_file):
    """
    Extrae la ruta completa de un archivo de datos usando variables de entorno.
    """
    df.validate_env_variables('directory_data')
    data_path = os.path.normpath(os.path.join(os.getenv(
        'directory_project'), os.path.join(os.getenv('directory_data'), name_file)))
    return data_path


def create_directory(rute_model):
    """
    Crea el directorio necesario para guardar los resultados del modelo.
    """
    directory_proyect = os.getcwd()
    for dr in ['images']:
        directory = os.path.normpath(os.path.join(
            directory_proyect, f"{dr}/{rute_model}"))
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Directorio creado: {directory}")
    return directory

def prepar_data():
    # Ruta del archivo de datos y carga
    clear_path = extract_data_path('kmeans/clear_data.csv')
    clear_data = load_data(clear_path)
    df.divie_data(clear_data)

def random_forest_model():

    """
    Entrena, evalúa y visualiza varios modelos de clustering (GMM, Spectral Clustering, Agglomerative y MiniBatchKMeans).
    """

    # Obtener parámetros de entorno
    df.validate_env_variables('random_state')
    random_state = int(os.getenv('random_state'))

    create_directory('forest')

    # Ruta del archivo de datos y carga
    train_path = extract_data_path('train_data.csv')
    train_data = load_data(train_path)

    x_train = train_data[['total_day_minutes', 'total_evening_minutes', 'total_night_minutes']]
    y_train = train_data['Cluster']

    valid_path = extract_data_path('validation_data.csv')
    validation_data = load_data(valid_path)

    x_validation = validation_data[['total_day_minutes', 'total_evening_minutes', 'total_night_minutes']]
    y_validation = validation_data['Cluster']

    #Demostraccion de los datos guardados
    print("\nDemostracion de datos agrupados por K-Means dentro del CSV\n")
    print(train_data.head())
    

    print("\nInformacion de los datos\n")
    print(train_data.info())

    # Asignamos el modelo a una variable

    forest_model = RandomForestClassifier(random_state=random_state)
    forest_model.fit(x_train,y_train)

    # Aplicamos predicciones al modelo entrenado
    forest_predictions = forest_model.predict(x_validation)
    forest_predictions_prob = forest_model.predict_proba(x_validation)

    # Aplicamos metricas de evaluación al modelo
    accuracy = accuracy_score(y_validation, forest_predictions)
    roc_auc = roc_auc_score(y_validation, forest_predictions_prob, multi_class="ovr")
    class_rep = classification_report(y_validation, forest_predictions)

    print("\n Evaluación del Modelo: Bosque Aleatorio \n")
    print(f"\nPrecisión del modelo (Accuracy): {accuracy:.4f}\n")
    print(f"AUC del modelo: {roc_auc:.4f}")
    print("\nInforme de Clasificación:")
    print(class_rep)

    # Curva ROC (One-vs-Rest)
    plt.figure(figsize=(10, 6))

    # Iterar sobre cada clase
    for i in range(forest_predictions_prob.shape[1]):
        # Calcular la curva ROC para cada clase
        fpr, tpr, _ = roc_curve(y_validation, forest_predictions_prob[:, i], pos_label=i)
        roc_auc = auc(fpr, tpr)
    
        # Graficar la curva ROC para cada clase
        plt.plot(fpr, tpr, label=f'Clase {i} (AUC = {roc_auc:.2f})')

    # Configuración de la gráfica
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal de referencia
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curvas ROC para Cada Clase - Bosque Aleatorio')
    plt.legend(loc="lower right")
    plt.grid(True)

    # Guardar la imagen
    plt.savefig(df.create_image_path("forest/roc_curve.png"))

    # Mostrar la curva ROC
    plt.show()

    # Crear y visualizar la matriz de confusión
    conf_matrix = confusion_matrix(y_validation, forest_predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=forest_model.classes_, yticklabels=forest_model.classes_)
    plt.title("Matriz de Confusión")
    plt.xlabel("Clase Predicha")
    plt.ylabel("Clase Real")
    # Guardar la imagen
    plt.savefig(df.create_image_path("forest/confusion_matrix.png"))

    # Mostrar la curva ROC
    plt.show()


def linear_regression_model():
    # Obtener parámetros de entorno
    df.validate_env_variables('random_state')
    random_state = int(os.getenv('random_state'))

    create_directory('regression')

    # Ruta del archivo de datos y carga
    train_path = extract_data_path('train_data.csv')
    train_data = load_data(train_path)

    x_train = train_data[['total_day_minutes', 'total_evening_minutes', 'total_night_minutes']]
    y_train = train_data['Cluster']

    valid_path = extract_data_path('validation_data.csv')
    validation_data = load_data(valid_path)

    x_validation = validation_data[['total_day_minutes', 'total_evening_minutes', 'total_night_minutes']]
    y_validation = validation_data['Cluster']

    # Crear el modelo de regresión lineal
    linear_model = LinearRegression()

    # Entrenar el modelo con los datos de entrenamiento
    linear_model.fit(x_train, y_train)

    # Aplicamos prediciones al modelo
    linear_prediction = linear_model.predict(x_validation)

    mse = mean_squared_error(y_validation, linear_prediction)
    r2 = r2_score(y_validation, linear_prediction)

    print(f"Error cuadrático medio (MSE): {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # Graficar las predicciones vs los valores reales (si la variable objetivo es continua)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_validation, linear_prediction, color='blue', label='Predicciones')
    plt.plot([y_validation.min(), y_validation.max()], [y_validation.min(), y_validation.max()], color='red', linestyle='--', label='Línea perfecta')
    plt.title('Regresión Lineal: Predicciones vs Real')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.legend()
    plt.grid(True)
    plt.savefig(df.create_image_path("regression/mse_graph.png"))
    plt.show()

def main():
    prepar_data()


    random_forest_model()
    linear_regression_model()

if __name__ == "__main__":
    main()