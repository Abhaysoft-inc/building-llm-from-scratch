import numpy as np
import matplotlib.pyplot as plt

# Define a SimpleLinearRegression class to model the relationship between a single input feature and a target variable using a linear equation

class SimpleLinearRegression:
   def __init__(self):
    self.coefficients_ = None
    self.intercept_ = None
    self.r2score_ = None

   def fit(self, x, y):
        n = len(x)
        X_b = np.c_[np.ones((n,1)), x]

        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coefficients_[0]
        y_pred = X_b.dot(self.coefficients_)

        self.r2score_ = 1 - (np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2))
        self.y_pred_ = y_pred
    
   def predict(self, X):
        X_b = np.c_[np.ones((len(X),1)), X]
        return X_b.dot(self.coefficients_)
    

# model fitting and visualisation

# starting with 1 feature

import pandas as pd

df = pd.read_csv("Housing.csv")

X = df[[
    "area",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking"
]].values
y = df["price"].values

indices = np.random.permutation(len(X))

split = int(0.8 * len(X))

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X[train_idx]
y_train = y[train_idx]

X_test = X[test_idx]
y_test = y[test_idx]

slr = SimpleLinearRegression()
slr.fit(X_train, y_train)

y_pred = slr.predict(X_test)

# evaluate

mse = np.mean((y_test - y_pred) ** 2)

r2 = 1 - (
    np.sum((y_test - y_pred) ** 2)
    / np.sum((y_test - np.mean(y_test)) ** 2)
)

print("MSE:", mse)
print("R²:", r2)


plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red"
)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual")
plt.show()
