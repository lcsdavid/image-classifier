import numpy as np

class DMIN:
	def __init__(self):
		self.data = np.array([])
		self.label = np.array([])
		self.n_label = 0

	def fit(self, data, label):
		self.data = data
		self.label = label
		self.n_label = len(set(label))

	def predict(self, data):
		return [self.label[np.argmin(np.sum(np.subtract(self.data, data[iterator]) ** 2, axis=1))] for iterator in range(0, len(data))]

	def score(self, data, label):
		return np.count_nonzero(self.predict(data) == label) / len(data)