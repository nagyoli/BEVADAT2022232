import pandas as pd
from typing import Tuple
from sklearn.metrics import confusion_matrix

class KNNClassifier:
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame
    y_preds: pd.DataFrame

    def __init__(self, k: int, test_split_ratio: float):
        self.k = k
        self.test_split_ratio = test_split_ratio

    @staticmethod
    def load_csv(csv_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        seed: int = 42
        dataset = pd.read_csv(csv_path, delimiter=",", header=None, na_values="\"\"", dtype=float)
        dataset.sample(frac=1, random_state=seed)
        dataset = dataset[dataset >= 0.0].dropna()
        dataset = dataset[dataset <= 13.0].dropna()
        dataset = dataset.reset_index()
        x, y = dataset.iloc[:, :4], dataset.iloc[:, -1]
        return x, y

    def train_test_split(self, features: pd.DataFrame, labels: pd.DataFrame) -> None:
        test_size = int(len(features) * self.test_split_ratio)
        train_size = len(features) - test_size
        assert len(features) == test_size + train_size, "Size mismatch!"

        x_train, y_train = features[:train_size, :], labels[:train_size]
        x_test, y_test = features[train_size:train_size + test_size, :], labels[train_size:train_size + test_size]
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def euclidean(self, points: pd.DataFrame, element_of_x: pd.Series) -> pd.DataFrame:
        return ((points - element_of_x) ** 2).sum(axis=1) ** 0.5

    def predict(self) -> None:
        labels_pred = []
        for index, row in self.x_test.iterrows():
            distances = self.euclidean(self.x_train, row)
            distances = pd.DataFrame(sorted(zip(distances, self.y_train)))
            label_pred = distances.iloc[:self.k, 1].mode()
            labels_pred.append(label_pred)
        self.y_preds = pd.DataFrame(labels_pred).iloc[:, 0]

    def accuracy(self) -> float:
        true_positive = (self.y_test == self.y_preds).sum()
        return true_positive / len(self.y_test) * 100

    def plot_confusion_matrix(self) -> pd.DataFrame:
        return confusion_matrix(self.y_test, self.y_preds)

    @property
    def k_neighbors(self):
        return self.k


    def best_k(self) -> Tuple[int, float]:
        best_k = 0
        best_acc = 0.0
        original_k = self.k
        for i in range(20):
            self.k = i+1
            self.predict()
            current_acc = self.accuracy()
            if best_acc < current_acc:
                best_k = self.k
                best_acc = current_acc

        self.k = original_k
        return best_k, round(best_acc, 2)