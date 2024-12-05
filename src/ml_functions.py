import os
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_functions import validate_env_variables

load_dotenv()


def create_image_path(name_path):
    validate_env_variables('directory_image')
    image_path = os.path.normpath(os.path.join(
        os.getenv('directory_image'), name_path))
    return image_path


def train_kmeans(data, n_clusters=None):
    """
    Entrena un modelo K-Means Clustering.

    :param data: DataFrame con los datos normalizados para entrenamiento.
    :param n_clusters: Número de clusters a configurar.
    :return: Modelo entrenado, inertia, silhouette score.
    """

    validate_env_variables('random_state', 'n_clusters')
    random_state = int(os.getenv('random_state'))

    if not n_clusters:
        n_clusters = int(os.getenv('n_clusters'))

    # Inicializar el modelo
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)

    # Entrenar el modelo
    kmeans.fit(data)

    # Evaluar el modelo
    inertia = kmeans.inertia_  # Metrica de calidad del modelo
    labels = kmeans.labels_
    silhouette = silhouette_score(data, labels)  # Puntuacion de Silhouette

    return kmeans, inertia, labels, silhouette


def evaluate_kmeans(inertia, silhouette):
    """
    Evalúa el modelo K-Means mostrando las métricas relevantes.

    :param inertia: Inertia del modelo entrenado.
    :param silhouette: Puntuación de Silhouette del modelo.
    """
    print("Evaluacion del modelo K-Means:")
    print(f"- Inertia: {inertia}")
    print(f"- Silhouette Score: {silhouette}")

    # Agregar logica para justar cluster si es necesatio
    if silhouette < 0.5:
        print("Considera aumentar o reducir el numero de clusters.")


def visualize_clusters(kmeans_model, data):

    # colores = ["red" , "blue", "orange" , "black", "purple", "pink" , "brown"]
    # for cluster in range(kmeans_model.n_clusters):
    # plt.scatter(clientes[clientes["cluster"] == cluster]["saldo"],
    #                 clientes[clientes["cluster"] == cluster]["transacciones"],
    #                 marker = "O", s=180, color = colores[cluster], alpha=0.5)

    # plt.scatter(kmeans_model.cluster_center_[cluster][0],
    #              kmeans_model.cluster_center_[cluster][1],
    #                 marker = "P", s=280, color = colores[cluster])

    # plt.title("C1ientes", fontsize=20)
    # plt.xlabel("Saldo en cuenta de ahorros (pesos)" , fontsize=15)
    # plt.ylabel( "Veces que usó tarjeta de crédito", fontsize=15)
    # plt.text(1.15, 0.2, "K = %i" % kmeans_model.n_clusters, fontsize=25)
    # plt.text(1.15, 0.2, "Inercia = %0.2f" % kmeans_model.inertia_, fontsize=25)
    # plt.xlim(-0.1, 1.1)
    # plt.ylim(-0.1, 1.1)
    # plt.show( )

    # Graficar clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1],
                c=kmeans_model.labels_, cmap='viridis', s=50, alpha=0.5, label="Puntos Clusterizados")
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[
                :, 1], s=200, c='red', marker='X', label="Centroides")

    plt.title(f"Clusters con K={kmeans_model.n_clusters}")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.legend()
    plt.savefig(create_image_path('kmeans_checked.png'))
    plt.show()


def plot_elbow_method(data, max_clusters=10):
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
    plt.savefig(create_image_path('elbow_method.png'))
    plt.show()


# Función para visualizar los datos originales
def plot_original_data(df):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[:, 0], df[:, 1], c="blue", s=50, label="Datos originales")
    plt.title("Datos Originales")
    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.legend()
    plt.savefig(create_image_path('original_data.png'))
    plt.show()


# 5. Visualización de clusters en 2D (opcional, con PCA)


def visualize_clusters_2d(data, labels, kmeans):
    '''
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    

    # Normalizar las dimensiones reducidas (PCA)
    scaler = StandardScaler()
    # Normaliza las 2 componentes principales
    reduced_data = scaler.fit_transform(reduced_data)

    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[
                :, 1], s=200, c='red', marker='X', label="Centroides")
    '''

    # Graficar los resultados
    graf = plt.figure(figsize=(10, 8))
    ax = graf.add_subplot(111,projection='3d')
    scatter = ax.scatter(data.iloc[:, 0], data.iloc[:, 3], data.iloc[:, 6],
                c=labels, cmap='viridis', s=50, alpha=0.7, label="Clientes")
    
    ax.set_title("Clusters visualizados en 3D")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_zlabel("Componente Principal 3")

    graf.colorbar(scatter, ax=ax, label="Cluster ID")
    plt.legend()
    plt.grid(True)
    plt.savefig(create_image_path('clusters_3d.png'))
    plt.show()

# 6. Asignar etiquetas a los datos


def assign_clusters(df, labels):
    df["Cluster"] = labels
    print(df.head())
    return df

from sklearn.manifold import TSNE

def visualize_clusters_tsne(data, labels, kmeans):
    # Reducir dimensiones con t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)

    # Graficar los resultados
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')  # Centroides
    plt.title("Clusters visualizados con t-SNE")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label="Cluster ID")
    plt.grid(True)
    plt.show()
