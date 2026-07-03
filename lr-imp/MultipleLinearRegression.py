import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MultipleLinearRegression:
    def __init__(self):
        self.coefficients_ = None 
        self.intercept_ = None     
        self.r2score_ = None    
        
    def fit(self, X, y):
        n = X.shape[0]
        X_b = np.c_[np.ones((n,1)), X]  
        self.coefficients_ = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        self.intercept_ = self.coefficients_[0]
        y_pred = X_b.dot(self.coefficients_)
        self.r2score_ = 1 - (np.sum((y - y_pred)**2)/np.sum((y - np.mean(y))**2))
        self.y_pred_ = y_pred
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0],1)), X]  
        return X_b.dot(self.coefficients_)
    
    
df = pd.read_csv("Housing.csv")

# encode binary yes/no columns as 1/0
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
               "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# one-hot encode furnishingstatus (drop_first to avoid the dummy-variable trap,
# since fit() already adds its own intercept column)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

feature_cols = [c for c in df.columns if c != "price"]

X_multi = df[feature_cols].values.astype(float)
y_multi = df["price"].values.astype(float)

np.random.seed(0)
indices = np.random.permutation(len(X_multi))
split = int(0.8 * len(X_multi))
train_idx, test_idx = indices[:split], indices[split:]

X_train, y_train = X_multi[train_idx], y_multi[train_idx]
X_test, y_test = X_multi[test_idx], y_multi[test_idx]

mlr = MultipleLinearRegression()
mlr.fit(X_train, y_train)
print("Multiple LR Coefficients:")
for name, coef in zip(["intercept"] + feature_cols, mlr.coefficients_):
    print(f"  {name}: {coef:.2f}")
print(f"Train R² Score: {mlr.r2score_:.2f}")

y_pred_test = mlr.predict(X_test)
test_r2 = 1 - (np.sum((y_test - y_pred_test)**2) / np.sum((y_test - np.mean(y_test))**2))
test_mse = np.mean((y_test - y_pred_test)**2)
print(f"Test R² Score: {test_r2:.2f}")
print(f"Test MSE: {test_mse:.2f}")

# 5-fold cross-validation for a more reliable estimate than a single split
k = 5
fold_size = len(X_multi) // k
cv_scores = []
for fold in range(k):
    val_idx = indices[fold*fold_size:(fold+1)*fold_size]
    tr_idx = np.concatenate([indices[:fold*fold_size], indices[(fold+1)*fold_size:]])

    fold_model = MultipleLinearRegression()
    fold_model.fit(X_multi[tr_idx], y_multi[tr_idx])

    y_val_pred = fold_model.predict(X_multi[val_idx])
    y_val = y_multi[val_idx]
    fold_r2 = 1 - (np.sum((y_val - y_val_pred)**2) / np.sum((y_val - np.mean(y_val))**2))
    cv_scores.append(fold_r2)

print(f"5-fold CV R² scores: {[f'{s:.2f}' for s in cv_scores]}")
print(f"5-fold CV mean R²: {np.mean(cv_scores):.2f}")

plt.figure(figsize=(7,7))
plt.scatter(y_test, y_pred_test, color="blue", alpha=0.7, label="Test predictions")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color="red", label="Perfect prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Multiple Linear Regression: Predicted vs Actual (Housing data, all features)")
plt.legend()
plt.show()