import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def get_pca_model(data: pd.DataFrame) -> PCA:

    pca = PCA(n_components=3)
    pca.fit(data)
    return pca


def reduce_dimension(df: pd.DataFrame, pca: PCA) -> pd.DataFrame:
    return pd.DataFrame(pca.transform(df), columns=["x", "y", "z"])


def predict_clusters(pca_df: pd.DataFrame):
    """Get model with the parameter `n_clusters`"""
    model = KMeans(n_clusters=4)
    return model.fit_predict(pca_df)


def get_silhouette_score(pca_df: pd.DataFrame, labels: pd.DataFrame) -> float:
    sil_score = silhouette_score(pca_df, labels)
    return sil_score


def insert_clusters_to_df(df: pd.DataFrame, clusters: np.ndarray) -> pd.DataFrame:
    return df.assign(clusters=clusters)


def plot_clusters(data: pd.DataFrame, preds: np.ndarray, centroids) -> None:

    plt.figure(figsize=(8, 5))
    ax = plt.subplot(111, projection="3d")
    ax.scatter(data["x"], data["y"], data["z"], s=40,
               c=data["c"], marker="o", cmap="Accent", alpha=0.6)
    for i in range(3):
        ax.scatter(centroids[i, 0], centroids[i, 1],
                   centroids[i, 2], s=80, color='k')
    ax.set_title("The Plot Of The Clusters")
    ax.set_xlabel('col1')
    ax.set_ylabel('col2')
    ax.set_zlabel('col3')
    plt.show()
