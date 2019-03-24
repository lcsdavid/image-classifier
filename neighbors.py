import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

# Nearest Neighbors

class Neighbors:
	def __init__(self, data, label, n_neighbors = 5, algorithm = 'auto'):
		self.data = data
		self.label = label
		self.n_label = len(set(self.label))
		self.neighbors = NearestNeighbors(n_neighbors, algorithm).fit(X)

	def score(self, data, label):
		if self.n_label != len(set(label)):
			raise ValueError("given label are not compatible with training set's labels")
		score, classes = 0, np.zeros(self.n_label)
		index = self.neighbors.kneighbors(data, return_distance = False)
		classes[self.label[index]] += 1
		# if (classes[np.argmax(classes)] == )

		return score / len(data)
			
		


nnc = NearestNeighborsClassifier(X, Y, 10)
#distances, indices = neighbors.kneighbors(reduced)

# for i in range(0, 1001):
	# print("found {}, expected {}".format(np.argmax(indices[i]), Y[i]))
	# img = X[i].reshape(28, 28)
	# plt.imshow(img, plt.cm.gray)
	# #plt.title("found {}, expected {}".format(np.argmax(indices[i]), Y[i]))
	# plt.show()
	# input("enter")

print("Score: {}%".format(nnc.score(X, Y) * 100))



		print('Testing Minimal Distance algorithm:')
		start = time.time()
		dmin = DMIN(reducedX, Y)
		score = dmin.score(reduceDevX, devY)
		print('Score: {}'.format(score))
		end = time.time()
		print('DMIN processed in {} sec...\n'.format(end - start))
		csvrow.append([score, end - start])
		
		print('Testing SVM algorithm:')
		
		print('Testing Nearest Neighbors algorithm:')
