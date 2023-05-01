import numpy as np
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix


class KMeansOnDigits():
    def __init__(self, n_clusters, random_state):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def load_dataset(self):
        self.digits = load_digits()

    def predict(self):
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.clusters = kmeans.fit_predict(X=self.digits.data, y=self.digits.target)

    def get_labels(self):
        result = np.zeros_like(self.clusters.shape)
        for i in range(10):
            mask = (self.clusters == i)
            subarray = self.digits.target[mask]
            mode = np.bincount(subarray).argmax()
            result[mask] = mode
        self.labels = result

    def calc_accuracy(self):
        self.accuracy = round(accuracy_score(self.labels, self.digits.target), 2)

    def confusion_matrix(self):
        self.mat = confusion_matrix(self.digits.target, self.labels)
