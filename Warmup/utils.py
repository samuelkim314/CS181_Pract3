import csv
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import operator

def importFruitData(filename="fruit.csv"):
    reader = csv.reader(open(filename, 'rb'))
    fruit = []
    xs = []
    for row in itertools.islice(reader, 1, None):
        fruit.append(float(row[0]))
        x1 = float(row[1])
        x2 = float(row[2])
        xs.append([x1, x2])
    return np.array(xs), np.array(fruit)

def plotRegression(x, t, w, y, K):
	plt.figure()
	plt.axis([4, 10, 4, 11])
	#scatter plot of the fruit data
	levels = [1,2,3]
	colors = ['red','orange','yellow']
	cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
	plt.scatter(x[:,0], x[:,1], c=t, s=144, alpha=0.75, cmap=cmap, norm=norm)
	#plot of decision boundaries
	dx = 50
	x1 = np.linspace(4, 10, dx)
	x2 = np.linspace(4, 11, dx)
	_w = w.reshape((K, x.shape[1] + 1))
	grid = np.zeros((dx, dx))
	for i in range(len(x1)):
		for j in range(len(x2)):
			index, value = max(enumerate(y(_w, [1, x1[i], x2[j]])), key=operator.itemgetter(1))
			grid[i][j] = index + 1
	plt.imshow(grid, extent=(4, 10, 4, 11), cmap=cmap)
	# for i in range(K):
	# 	m = _w[i]
	# 	plt.plot(_x, -(m[1]*_x + m[0])/m[2], color=colors[i])