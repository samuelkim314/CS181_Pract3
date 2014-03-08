import csv
import sys
import itertools
import numpy as np

def importFruitData(filename="fruit.csv"):
    reader = csv.reader(open(filename, 'rb'))
    fruit = []
    xs = []
    for row in itertools.islice(reader,1,None):
        fruit.append(float(row[0]))
        x1 = float(row[1])
        x2 = float(row[2])
        xs.append([x1, x2])

    return np.array(xs), np.array(fruit)

def logRegress(xs, y):
    pass
