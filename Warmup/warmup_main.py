import classification as classif
import utils

xs, classes = utils.importFruitData()
phi = classif.basisNone(xs)
t = classif.vectT(classes, 3)
w = classif.logRegress(phi, t)
utils.plotRegression(xs,classes, w, 3)
plt.show()