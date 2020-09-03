"""
Source:
https://math.stackexchange.com/questions/477207/derivative-of-cost-function-for-logistic-regression
https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
"""

import numpy as np

class LogisticRegression():
	"""
	This class assumes numpy data and uses gradient descent to do MLE
	for logistic regression
	"""

	def __init__(self):
		self.iters = 2000
		self.alpha = 0.01

	def _predict(self, X):
		temp = -np.matmul(X, self.theta)
		y_pred = 1.0/(1 + np.exp(temp.reshape(-1)))
		return y_pred

	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		return self._predict(X)

	def cost(self, X, y):
		# Minimize the opposite of the 
		# negative log likelihood (we want maximum likelihood estimate, 
		# this is going to be negative, so closest number to 0)
		y_pred = self._predict(X)
		cost = - sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
		return cost

	def gradient_update(self, X, y):
		"""
		This calculates the gradient of cost with respect to theta. 
		Specifically, the cost of the binary cross entropy (aka negative log likelihood)
		"""
		# In layman terms: gradient of y wrt theta = X_T * (y_hat - y) where y_hat = X * theta
		y_pred = self._predict(X)
		gradient = np.dot(X.T, (y_pred - y)).reshape(-1,1)
		gradient /= len(X)
		gradient *= self.alpha
		return gradient


	def fit(self, X, y, verbose):
		"""
		Expects data in shape (n, f), where n is the number of observations, 
		and f is the number of features.
		e.g. X = [200, 4]
		theta: [4, 1]
		y: [200, 1]

		This function finds the weights w and bias b.
		"""
		# Weights (adding 1 to X for bias term)
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		self.theta = np.random.random_sample((X.shape[1], 1))

		# Iterations and learning weight
		for i in range(self.iters):
			self.theta -= self.gradient_update(X, y)
			cost = self.cost(X, y)
			# print("Epoch: {}, Loss: {}".format(i, cost))