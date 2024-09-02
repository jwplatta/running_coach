import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator

class OptimizeClusters:
    def fit(self, *args):
        return self

    def transform(self, X_pca, *args):
        k_values = range(1, 20)
        inertia = []

        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X_pca)
            inertia.append(kmeans.inertia_)

        kneedle = KneeLocator(
            k_values,
            inertia,
            curve='convex',
            direction='decreasing'
        )

        opt_k = kneedle.elbow
        print(f'Optimal number of clusters: {opt_k}')

        kmeans = KMeans(n_clusters=opt_k)
        y_kmeans = kmeans.fit_predict(X_pca)
        data = np.append(
            X_pca,
            y_kmeans[:, np.newaxis],
            axis=1
        )
        return data, kmeans.cluster_centers_, opt_k

