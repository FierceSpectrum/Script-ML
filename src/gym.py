import os
import pandas as pd
from dotenv import load_dotenv
from data_connection import load_data
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


load_dotenv()

def create_image_path(name_path):
    image_path = os.path.normpath(os.path.join(os.getenv('directory_image'), name_path))
    return image_path

def model_K_Means():
    
    # Cargar el conjunto de datos en una variable
    directory_project = os.getenv('directory_project')

    directory_data = os.path.normpath(
            os.path.join(
                directory_project,
                os.getenv('directory_data')
                )
            )

    train_path = os.path.normpath(os.path.join(directory_data, 'train_data.csv'))

    validation_path = os.path.normpath(os.path.join(directory_data, 'validation_data.csv'))
    
    
    x_train = load_data(train_path)
    x_validation = load_data(validation_path)

    if x_train is None:
        print("No se pudieron cargar los datos de entrenamiento")
        return
    
    if x_validation is None:
        print("No se pudieron cargar los datos de validacion")
        return
    

    # Saca la cantida de agrupaciones ideal para el modelo
    wcss = []
    for i in range(2, 24):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=20, random_state=0)
        kmeans.fit(x_train)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(2, 24), wcss, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WSCC')
    plt.savefig(create_image_path('elbow_method.png'))
    plt.show()
    

    # <------- NO TESTEADO 

    # Arreglando la cantidad de clusters que usaria K-Means
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=20, random_state=0)
    kmeans.fit(x_train)

    cluster_labels = kmeans.labels_

    # Datos del modelo entrenado
    plt.scatter(x_train[:,0], x_train[:,1], x_train[:,3], x_train[:,4], x_train[:,6], x_train[:,7], x_train[:,9], x_train[:,10], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,3], kmeans.cluster_centers_[:,4], kmeans.cluster_centers_[:,6], kmeans.cluster_centers_[:,7], kmeans.cluster_centers_[:,9], kmeans.cluster_centers_[:,10], c='red', marker='X', s=200, label='Center')
    plt.title('K-Means with 4 Clusters')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(create_image_path('kmeans_trained.png'))
    plt.show()

    # Datos del modelo validado
    clusters_validation = kmeans.predict(x_validation)

    plt.scatter(x_train[:,0], x_train[:,1], x_train[:,3], x_train[:,4], x_train[:,6], x_train[:,7], x_train[:,9], x_train[:,10], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,3], kmeans.cluster_centers_[:,4], kmeans.cluster_centers_[:,6], kmeans.cluster_centers_[:,7], kmeans.cluster_centers_[:,9], kmeans.cluster_centers_[:,10], c='red', marker='X', s=200, label='Center')
    plt.title('K-Means with 4 Clusters')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig(create_image_path('kmeans_checked.png'))
    plt.show()

    #Silhoutte Score para validad la calidad de los clusters

    silhouette_avg = silhouette_score(x_validation, clusters_validation)
    print(f"Silhouette Score en validación: {silhouette_avg}")

    client_type = ["day","evening", "night", "international"]

    # Muestra la clasificacion de cada uno de los datos
    list(zip(cluster_labels, client_type.values))

def model_DBSCAN():
    return

def main():
    model_K_Means()
    return




if __name__ == "__main__":
    main()