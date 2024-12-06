import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# 1. Cargar los conjuntos de datos
train_file = "data/train_data.csv"
validation_file = "data/validation_data.csv"

train_data = pd.read_csv(train_file)
validation_data = pd.read_csv(validation_file)

# 2. Preprocesamiento (Estandarización)
scaler = StandardScaler()
train_data_scaled = train_data
validation_data_scaled = validation_data

# 3. Configuración y ajuste inicial de DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan.fit(train_data_scaled)
train_labels = dbscan.labels_

# 4. Evaluación inicial del modelo
num_clusters = len(set(train_labels)) - (1 if -1 in train_labels else 0)
num_noise = sum(train_labels == -1)

print(f"Número de clusters: {num_clusters}")
print(f"Puntos clasificados como ruido: {num_noise}")

if len(set(train_labels)) > 1:
    silhouette = silhouette_score(train_data_scaled, train_labels)
    print(f"Coeficiente de Silueta (Entrenamiento): {silhouette:.4f}")
else:
    print(
        "No se puede calcular el coeficiente de silueta: solo hay un cluster o ruido."
    )


# 5. Optimización de parámetros (eps y min_samples)
def tune_dbscan(data, eps_values, min_samples_values):
    best_score = -1
    best_params = None
    for eps in eps_values:
        for min_samples in min_samples_values:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(data)
            if len(set(labels)) > 1:
                score = silhouette_score(data, labels)
                if score > best_score:
                    best_score = score
                    best_params = (eps, min_samples)
    return best_params, best_score


eps_values = np.arange(0.1, 1.0, 0.1)
min_samples_values = range(2, 10)
best_params, best_score = tune_dbscan(train_data_scaled, eps_values, min_samples_values)

print(
    f"Mejores parámetros: eps={best_params[0]}, min_samples={best_params[1]} (Score: {best_score:.4f})"
)

# Reentrenar con los mejores parámetros
dbscan = DBSCAN(eps=best_params[0], min_samples=best_params[1])
dbscan.fit(train_data_scaled)
train_labels = dbscan.labels_

# 6. Interpretación del modelo (PCA para visualización)
pca = PCA(n_components=2)
train_data_pca = pca.fit_transform(train_data_scaled)


# Extrae las variables más importantes de cada componente principal
def get_top_features(pca, num_features=2, feature_names=None):
    components = pca.components_
    feature_names = (
        feature_names
        if feature_names is not None
        else [f"Variable {i+1}" for i in range(components.shape[1])]
    )
    top_features = []
    for i in range(components.shape[0]):  # Itera por cada componente principal
        indices = np.argsort(np.abs(components[i]))[-num_features:][
            ::-1
        ]  # Ordena por importancia
        top_features.append([feature_names[j] for j in indices])
    return top_features


# Supongamos que train_data_scaled es un DataFrame
feature_names = (
    train_data_scaled.columns if hasattr(train_data_scaled, "columns") else None
)
top_features = get_top_features(pca, feature_names=feature_names)

# Construye el título dinámicamente
pca1_features = ", ".join(top_features[0])
pca2_features = ", ".join(top_features[1])
title = f"Clusters (PCA1: {pca1_features}, PCA2: {pca2_features})"


def plot_clusters(data, labels, title="Clusters"):
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:  # Ruido
            col = [0, 0, 1, 1]

        class_member_mask = labels == k
        xy = data[class_member_mask]
        plt.plot(
            xy[:, 0],
            xy[:, 1],
            "o",
            markerfacecolor=tuple(col),
            markeredgecolor="k",
            markersize=10,
        )

    plt.title(title)
    plt.xlabel(pca1_features)
    plt.ylabel(pca2_features)
    image_path = "images/" + title.replace(" ", "_")
    plt.savefig(image_path)
    # plt.show()
    print(f"Imagen {title} creada")


plot_clusters(train_data_pca, train_labels, title="Clusters en Datos de Entrenamiento")

# 7. Evaluación en datos de validación
validation_labels = dbscan.fit_predict(validation_data_scaled)

if len(set(validation_labels)) > 1:
    silhouette_validation = silhouette_score(validation_data_scaled, validation_labels)
    print(f"Coeficiente de Silueta (Validación): {silhouette_validation:.4f}")
else:
    print(
        "No se puede calcular el coeficiente de silueta en validación: solo hay un cluster o ruido."
    )

validation_data_pca = pca.transform(validation_data_scaled)
plot_clusters(
    validation_data_pca, validation_labels, title="Clusters en Datos de Validación"
)

# 8. Predicción en nuevos datos
new_data_scaled = pd.read_csv("data/test_data.csv")

new_labels = dbscan.fit_predict(new_data_scaled)
new_data_pca = pca.transform(new_data_scaled)
plot_clusters(new_data_pca, new_labels, title="Clusters en Nuevos Datos")

# Resumen de etiquetas en los nuevos datos
unique_labels, counts = np.unique(new_labels, return_counts=True)
print("Etiquetas en nuevos datos:", dict(zip(unique_labels, counts)))
