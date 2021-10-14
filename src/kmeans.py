import numpy as np
from numpy.random import default_rng
from typing import Tuple

def pick_initial_centroids(k: int, points: np.ndarray, seed: int = None) -> np.ndarray:
    rng = default_rng(seed)
    return rng.choice(points, size=k)

def compute_distances(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    return np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))

def get_closest_centroids(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    distances = compute_distances(points, centroids)
    return np.argmin(distances, axis=0)

def cluster_points(points: np.ndarray, centroids: np.ndarray, closest: np.ndarray) -> np.ndarray:
    return np.array([points[closest==i] for i in range(len(centroids))])

def update_centroids(clusters: np.ndarray) -> np.ndarray:
    return np.array([cluster.mean(axis=0) for cluster in clusters])

def k_means_cluster_step(points: np.ndarray, centroids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    closest = get_closest_centroids(points, centroids)
    clusters = cluster_points(points, centroids, closest)
    centroids = update_centroids(clusters)
    return clusters, centroids

def k_means_clustering(points: np.ndarray, k: int, seed: int = None) -> np.ndarray:
    centroids = pick_initial_centroids(k, points, seed)
    converged = False
    while not converged:
        clusters, new_centroids = k_means_cluster_step(points, centroids)
        converged = np.array_equal(new_centroids, centroids)
        centroids = new_centroids
    return clusters