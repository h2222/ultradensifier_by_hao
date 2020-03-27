

import scipy
import numpy as np
from scipy.stats import special_ortho_group


x = special_ortho_group.rvs(3)

print(x)

# print(np.dot(x, x.T))

# print(np.eye(300, 1))



Q = np.matrix(scipy.stats.ortho_group.rvs(300, random_state=3))


# batch_sz = 50
X = np.random.rand(50, 300) * Q[0, :].reshape(300, 1)
x =np.absolute(X)


print(x[1][0,0])
