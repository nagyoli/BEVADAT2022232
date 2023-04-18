import numpy as np
from typing import List
from matplotlib import pyplot as plt


class LinearRegression():

    def __init__(self, epochs: int = 10000, lr: float = 1e-2):
        self.epochs = epochs
        self.learning_rate = lr
        self.alphahat: float = 0
        self.betahat: float = 0

    def fit(self, x: np.array, y: np.array):
        number_of_observations: int = len(x)
        losses: List[float] = []
        for i in range(self.epochs):
            y_pred = self.betahat * x + self.alphahat
            residual = y - y_pred
            loss = np.sum(residual ** 2)
            losses.append(loss)
            delta_loss_delta_beta = (-2 / number_of_observations) * sum(x * residual)
            delta_loss_delta_alpha = (-2 / number_of_observations) * sum(residual)

            self.alphahat = self.alphahat - self.learning_rate * delta_loss_delta_alpha
            self.betahat = self.betahat - self.learning_rate * delta_loss_delta_beta

            if i % 100 == 0:
                print(np.mean(y - y_pred))

    def predict(self, X_test) -> List:
        pred = []
        for X_row in X_test:
            y_pred = self.betahat * X_row + self.alphahat
            pred.append(y_pred)

        return pred

    def evaluate(self, X: np.array, y: np.array):
        y_pred = self.predict(X)
        y_pred = np.array(y_pred)
        print("Mean Absolute Error:", np.mean(np.abs(y_pred - y)))

        # Calculate the Mean Squared Error
        print("Mean Squared Error:", np.mean((y_pred - y) ** 2))

    def plot_prediction(self, X, y):

        y_pred = self.predict(X)
        y_pred = np.array(y_pred)

        plt.scatter(X, y)
        plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # predicted
        plt.show()