import numpy as np
from numpy.linalg import eigh

class PCA():
	@staticmethod
	def transform(X, n, verbose):
		"""
		n = 1
		X = [10 x 2]
		output -> [10, 1]
		"""
		covariance = np.matmul(X.T, X)

		# Finds vectors and lambdas such that A*v = l*v, where A is original matrix
		# l is eigenvalue, v is eigenvector
		values, vectors = eigh(covariance)
		vectors = vectors[::-1]
		values = values[::-1]
		vectors = vectors[:n]
		print(X.shape, vectors.shape)

		variance = 100*sum(values[:n])/sum(values)
		if verbose:
			print("Percent of variance expressed in chosen number of eigenvectors is: {}%".format(np.round(variance,3)))
		return np.matmul(X, vectors.T)




