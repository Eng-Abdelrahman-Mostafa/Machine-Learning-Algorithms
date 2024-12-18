import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        # Step 2: Initialize coefficients (weights) and intercept
        self.coef_ = np.zeros(X.shape[1])  # For multiple features
        self.intercept_ = 0

        # Step 3: Gradient Descent Algorithm
        n = len(X)
        for _ in range(self.epochs):
            # Predictions using current values of coef_ and intercept_
            y_pred = self.predict(X)

            # Calculate the gradients (derivatives of the MSE cost function)
            d_coef = (-2 / n) * np.dot(X.T, (y - y_pred))  # Gradient w.r.t coefficients
            d_intercept = (-2 / n) * np.sum(y - y_pred)  # Gradient w.r.t intercept

            # Update the coefficients using gradient descent
            self.coef_ -= self.learning_rate * d_coef
            self.intercept_ -= self.learning_rate * d_intercept

    def predict(self, X):
        # Step 4: Make predictions using the learned coefficients
        return np.dot(X, self.coef_) + self.intercept_

    def get_params(self):
        return self.coef_, self.intercept_