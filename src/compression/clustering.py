from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances_argmin


def apply_kmeans(weights_np: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = kmeans.fit_predict(weights_np.reshape(-1, 1))
    centroids = kmeans.cluster_centers_.flatten()
    return centroids, labels


def apply_gmm(weights_np: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(weights_np.reshape(-1, 1))
    labels = gmm.predict(weights_np.reshape(-1, 1))
    centroids = gmm.means_.flatten()
    return centroids, labels


def apply_dbscan(
    weights_np: np.ndarray,
    eps: float,
    rng: Optional[np.random.Generator] = None,
    sample_cap: int = 20000,
    chunk_size: int = 100000,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    sample_size = min(len(weights_np), sample_cap)
    sample_indices = rng.choice(len(weights_np), sample_size, replace=False)
    sample_data = weights_np[sample_indices].reshape(-1, 1)

    db = DBSCAN(eps=eps, min_samples=5)
    db.fit(sample_data)

    unique_labels = set(db.labels_)
    if -1 in unique_labels:
        unique_labels.remove(-1)

    if len(unique_labels) == 0:
        return np.array([np.mean(weights_np)]), np.zeros(len(weights_np), dtype=int)

    centroids = []
    for label in unique_labels:
        cluster_points = sample_data[db.labels_ == label]
        centroids.append(np.mean(cluster_points))
    centroids = np.array(centroids)

    labels = []
    for i in range(0, len(weights_np), chunk_size):
        chunk = weights_np[i : i + chunk_size].reshape(-1, 1)
        chunk_labels = pairwise_distances_argmin(chunk, centroids.reshape(-1, 1))
        labels.append(chunk_labels)
    labels = np.concatenate(labels)

    return centroids, labels
