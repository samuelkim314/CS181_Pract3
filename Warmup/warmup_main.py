import classification as classif

xs, classes = classif.importFruitData()
phi = basisNone(xs)
t = vectT(classes)
logRegress(phi, t)