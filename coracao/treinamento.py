import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

from dados import persistir_modelo, PASTA_PLOTS

class TreinadorCluster:

    def __init__(self, semente=42, k_max=15, n_init=10):
        self.semente = semente
        self.k_max = k_max
        self.n_init = n_init

    def numero_otimo_clusters(self, X):
        K = range(1, self.k_max + 1)
        distorcoes = []
        for i in K:
            km = KMeans(n_clusters=i, random_state=self.semente, n_init=self.n_init).fit(X)
            distorcoes.append(
                sum(np.min(cdist(X, km.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

        x0, y0, xn, yn = K[0], distorcoes[0], K[-1], distorcoes[-1]
        dists = []
        for k, y in zip(K, distorcoes):
            num = abs((yn - y0) * k - (xn - x0) * y + xn * y0 - yn * x0)
            den = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
            dists.append(num / den)
        k_otimo = list(K)[int(np.argmax(dists))]

        os.makedirs(PASTA_PLOTS, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(list(K), distorcoes, marker='o')
        ax.axvline(k_otimo, color='#7b1e3a', linestyle='--', label=f'k otimo = {k_otimo}')
        ax.set(xlabel='Numero de grupos (k)', ylabel='Distorcao media',
               title='Metodo do cotovelo')
        ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(PASTA_PLOTS, 'Coracao_cotovelo.png')); plt.close()

        print(f'Numero otimo de grupos (cotovelo) = {k_otimo}')
        return k_otimo

    def treinar(self, X, k):
        modelo = KMeans(n_clusters=k, random_state=self.semente, n_init=self.n_init).fit(X)
        sil = silhouette_score(X, modelo.labels_)
        persistir_modelo(modelo, 'modelo_kmeans')
        print(f'KMeans treinado com k={k} grupos | silhueta={sil:.3f}')
        return modelo
