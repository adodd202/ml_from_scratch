import numpy as np

class LinearRegression():
	"""
	This class assumes numpy data and uses gradient descent to fit a distribution,
	training a linear regressor.
	"""

	def __init__(self):
		pass

	def fit(self, X, y):
		"""
		Expects data in shape (n, f), where n is the number of observations, 
		and f is the number of features.

		This function finds the weights w and bias b.
		"""

		# Add ones to X for bias term
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

		# Weights and bias
		theta = np.random.random_sample((X.shape[1], 1))
		m = X.shape[0]

		# Iterations and learning weight
		iters = 1000
		alpha = 0.0003

		# Iterate and train
		for i in range(iters):
			y_pred = np.matmul(X, theta)
			y_delta = (y.reshape(-1,1) - y_pred).reshape((1, -1))
			grad_theta = - 1/m * np.transpose(np.matmul(y_delta, X))
			theta = theta - alpha * 1/m * grad_theta
			loss = 1/m * np.sum(np.square((y - y_pred)))
			print("Epoch: {}, Loss: {}".format(i, loss))

		self.theta = theta


	def predict(self, X):
		X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
		y_pred = np.matmul(X, self.theta).reshape(-1)
		return y_pred