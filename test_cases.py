import numpy as np
import pandas as pd
from linear_regression_1 import LinearRegression
from logistic_regression_2 import LogisticRegression
from sklearn.datasets import load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split

# Test linear regressor 
# X, y = load_boston(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# print("Predicted: ", model.predict(X_test))
# print("True: ", y_test)

# Test logistic classifier
# X, y = load_breast_cancer(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = np.array([[1, 2], 
					[0.9, 1.9], 
					[0.8, 2.1],
					[0.7, 2.2], 
					[0.8, 2.3], 
					[0.8, 1.8],
					[0.1, 1],
					[0.4, 1.7],
					[0.5, 1.9],
					[0.2, 1.5]])
X_test = X_train + np.random.uniform(-0.5,0.5,size=X_train.shape)
y_train = np.array([0,0,0,0,0,0,1,1,1,1])
y_test = y_train
model = LogisticRegression()
model.fit(X_train, y_train)
print("Predicted: ", model.predict(X_test))
print("True: ", y_test)



# Test PCA