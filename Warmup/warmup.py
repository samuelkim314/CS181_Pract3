from classification import *
from utils import *
import scipy.linalg

K = 3
xs, fruit = importFruitData()

# w1 = logisticRegression(xs, fruit, K)
# plotRegression(xs, fruit, w1, y, K)

w2 = generativeClassifierWithBayesian(xs, fruit, K)
plotRegression(xs, fruit, w2, y, K)


plt.show()