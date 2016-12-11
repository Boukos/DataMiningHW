import numpy as np
from sklearn import mixture

D = [[35], [41], [21], [20], [17], [55], [12], [33], [15], [18], [4],[51], [17], [46]]

gmm = mixture.GaussianMixture(2, init_params="random", n_init = 100).fit(D)
print gmm.means_
print gmm.covariances_
print gmm.weights_



