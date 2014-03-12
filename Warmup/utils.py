import csv
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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

def plotRegression(x, t, w, K):
	plt.figure()
	plt.axis([4, 10, 0, 24])
	#scatter plot of the fruit data
	levels = [1,2,3]
	colors = ['red','orange','yellow']
	cmap, norm = mpl.colors.from_levels_and_colors(levels=levels, colors=colors, extend='max')
	plt.scatter(x[:,0], x[:,1], c=t, s=144, alpha=0.75, cmap=cmap, norm=norm)
	#plot of decision boundaries
	_x = np.linspace(4, 10, 50)
	_w = np.array_split(w, K - 1)
	for i in range(K - 1):
		m = _w[i]
		plt.plot(_x, -(m[1]*_x + m[0])/m[2], color=colors[i])