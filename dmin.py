"""@package docstring
Documentation for this module.

More details.
"""

import numpy as np

class DMIN:
	def __init__(self):
		self.data = np.array([])
		self.label = np.array([])
		self.n_label = 0

	def fit(self, data, label):
		"""
		Fit the DMIN model according to the given training data.

        Args:
			data : array-like
				Training vectors.
			label : array-like
				Target values (class labels in classification, real numbers in regression).

		Returns: self : object
        """
		self.data = data
		self.label = label
		self.n_label = len(set(label))
		return self

	def predict(self, data):
		"""
		Perform classification on samples in data.

        Args:
			data : array-like
				Test samples.

		Returns: array-like
			Class labels for samples in data.
        """
		return [self.label[np.argmin(np.sum(np.subtract(self.data, data[iterator]) ** 2, axis=1))] for iterator in range(0, len(data))]

	def score(self, data, label):
		"""
		Returns the mean accuracy on the given test data and labels.

		Args:
			data : array-like
				Test samples.
			label : array-like
				True labels for data.
		
		Returns: score : float
			Mean accuracy of self.predict(X) wrt. y 
		"""
		return np.count_nonzero(self.predict(data) == label) / len(data)

	def confusion(self, data, label):
		 """
		Returns the confusion matrix corresponding to the given test data and labels.

		Args:
			data : array-like
				Test samples.
			label : array-like
				True labels for data.

		Returns: matrix : matrix
			confusion matrix of self.predict(X) wrt. y 
		"""
		confusion_matrix = np.zeros((self.n_label, self.n_label))
		predictions = self.predict(data)
		for prediction in predictions:
			confusion_matrix[label][prediction] += 1
		return confusion_matrix