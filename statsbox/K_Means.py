import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# euclidean distance


def dist(a, b):
    return np.sqrt(sum(np.square(a-b)))


def init_centroids(k, X):

    number_of_samples = X.shape[0]
    sample_points_ids = random.sample(range(0, number_of_samples), k)

    centroids = [tuple(X[id]) for id in sample_points_ids]
    unique_centroids = list(set(centroids))

    number_of_unique_centroids = len(unique_centroids)

    while number_of_unique_centroids < k:
        new_sample_points_ids = random.sample(
            range(0, number_of_samples), k - number_of_unique_centroids)
        new_centroids = [tuple(X[id]) for id in new_sample_points_ids]
        unique_centroids = list(set(unique_centroids + new_centroids))

        number_of_unique_centroids = len(unique_centroids)

    return np.array(unique_centroids)


def assign_cluster(k, X, cg):
    cluster = [-1]*len(X)
    for i in range(len(X)):
        dist_arr = []
        for j in range(k):
            dist_arr.append(dist(X[i], cg[j]))
        idx = np.argmin(dist_arr)
        cluster[i] = idx
    return np.asarray(cluster)


def compute_centroids(k, X, cluster):
    cg_arr = []
    for i in range(k):
        arr = []
        for j in range(len(X)):
            if cluster[j] == i:
                arr.append(X[j])
        cg_arr.append(np.mean(arr, axis=0))
    return np.asarray(cg_arr)


def measure_change(cg_prev, cg_new):
    res = 0
    for a, b in zip(cg_prev, cg_new):
        res += dist(a, b)
    return res


def cluster(k, X):
    cg_prev = init_centroids(k, X)
    cluster = [0]*len(X)
    cg_change = 100
    while cg_change > .001:
        cluster = assign_cluster(k, X, cg_prev)

        cg_new = compute_centroids(k, X, cluster)
        cg_change = measure_change(cg_new, cg_prev)
        cg_prev = cg_new
    return cluster, cg_prev


def predict(centroids, X_test):
    distarray = []
    for i in range(len(centroids)):
        distarray.append(dist(X_test, centroids[i]))
    return np.argmin(distarray)


def calculate_WSS(points, kmax):
    sse = []
    for k in range(1, kmax+1):
        pred_clusters, centroids = cluster(k, points)
        #centroids = kmeans.cluster_centers_
        #pred_clusters = kmeans.predict(points)
        curr_sse = 0
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + \
                (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse


def plot_elbow(X, k):
    plt.plot(calculate_WSS(X, k))


def PCA(X, n_components):
    cov_mat = np.cov(X, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]
    # you can select any number of components.
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    X_reduced = np.dot(eigenvector_subset.transpose(),
                       X.transpose()).transpose()
    return X_reduced


def plot_clusters(clusters, X, dims=2):

    dim1 = []
    dim2 = []

    X_reduced = PCA(X, 2)

    for x in X_reduced:
        dim1.append(x[0])
        dim2.append(x[1])

    for i in range(len(X_reduced)):
        colors = ['y', 'c', 'm', 'r', 'b', 'g', ]
        idx = clusters[i]
        color = colors[idx]
        plt.plot(dim1[i], dim2[i], marker="o",
                 markeredgecolor=color, markerfacecolor=color)
