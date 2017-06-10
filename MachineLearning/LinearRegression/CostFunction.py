import numpy as np
import math

# shape -> x = [ [,] , [,] ]  |   y = q = [,]

def Cost(x, y, q):
    jQ = np.zeros_like(q)
    for i in range(0, len(x)):
        for j in range(0, len(y)):
            jQ[i] += math.pow(q[i]*x[i][j] - y[j], 2)
    return jQ / 2



