import time
import numpy as np
import csv

from sklearn.decomposition import PCA

from dmin import DMIN
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

X = np.load('data/trn_img.npy')
Y = np.load('data/trn_lbl.npy')

devX = np.load('data/dev_img.npy')
devY = np.load('data/dev_lbl.npy')

testX = np.load('data/dev_img.npy')

def BenchmarkPCA(csvfile, n_range, algorithms):
	csvwriter = csv.writer(csvfile, delimiter=';')
	first_row = ['n_components (%)', 'n_components (dims)', 'Execution time PCA']
	for algorithm in algorithms:
		first_row.extend(['Score {}'.format(algorithm), 'Execution time {}'.format(algorithm)])
	csvwriter.writerow(first_row)

	for n in n_range:
		print('PCA with n_components={}:'.format(n))
		start = time.time()
		pca = PCA(n_components=n)
		reducedX = pca.fit_transform(X)
		reducedDevX = pca.transform(devX)
		end = time.time()
		print('PCA both fit and transform (on training and dev data) processed in {} sec...'.format(end - start))
		print('\tn_components={} reduce 784 dimensions to {} dimensions.\n'.format(n, np.shape(reducedX)[1]))
		csvrow = [n, np.shape(reducedX)[1], end - start]
		
		for algorithm in algorithms:
			print('Testing {} algorithm:'.format(algorithm))
			start = time.time()
			if algorithm == SVC:
				# Le mode par défaut dans la prochaine version est avec gamma='scale' et on voit la différence.
				algorithm_instance = algorithm(gamma='scale')
			else:
				algorithm_instance = algorithm()
			algorithm_instance.fit(reducedX, Y)
			algorithm_score = algorithm_instance.score(reducedDevX, devY)
			end = time.time()
			print('Score: {}'.format(algorithm_score))
			print('Execution time: {}\n'.format(end - start))
			csvrow.extend([algorithm_score, end - start])
		csvwriter.writerow(csvrow)

# BenchmarkPCA(open('generated/pca.csv', 'w'), np.array(range(5, 100, 5)) / 100, [DMIN, SVC, KNeighborsClassifier])
# BenchmarkPCA(open('generated/pca_precise.csv', 'w'), np.array(range(60, 75, 1)) / 100, [DMIN, SVC, KNeighborsClassifier])

# On génère le fichier de label solution.
clf = SVC(gamma='scale')
clf.fit(X, Y)
np.save('np.save', clf.predict(testX))