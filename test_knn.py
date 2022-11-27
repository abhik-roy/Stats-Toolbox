from sklearn.cluster import KMeans
import K_Means
import random
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

df = pd.read_csv("../Credit Card Customer Data.csv")

X = df.iloc[:, 1:].values

cluster, centroids = K_Means.cluster(4, X)

print(centroids)

kmeans = KMeans(n_clusters=4, random_state=0).fit(X)

print('      ')
print('       -           -              -   ')
print('     ')

print(kmeans.cluster_centers_)
