import numpy as np

def Step(y, t = 0):
    temp = np.copy(y)
    out1 = y >= 0
    out2 = y < 0
    temp[out1] = 1
    temp[out2] = 0
    return temp

def Sign(y, t = 0):
    temp = np.copy(y)
    out1 = y > 0
    out2 = y == 0
    out3 = y < 0
    temp[out1] = 1
    temp[out2] = 0
    temp[out3] = -1
    return temp

def SaturatedLinear(t, y = 0 ):
    temp = np.copy(y)
    out1 = y >= 1
    out2 = y <= -1
    temp[out1] = 1
    temp[out2] = -1

def Sigmoid(y):
    temp = np.divide(1, (1 + np.exp(-y)))
    return temp

def Tanh(y):
    temp = np.divide((1 - np.exp(-2.0*y)), (1 + np.exp(-2.0*y)))
    return temp
