import numpy as np

def CrossEntropy(x, y):
    return -np.sum(np.multiply(x, np.log(y)))

def MSE(x, y):
    return np.mean(np.power((x-y), 2))

def SSE(x, y):
    return np.sum(np.power((x-y), 2))