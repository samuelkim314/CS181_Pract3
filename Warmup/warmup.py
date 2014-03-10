from classification import *
from utils import *
import scipy.linalg

K = 3
xs, fruit = importFruitData()
N, mu = mean(xs, fruit, K)
print mu
print type(class_split(xs)[0])

w = logisticRegression(xs, fruit, K)
plotRegression(xs,fruit, w, K)

plt.show()