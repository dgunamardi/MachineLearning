import numpy as np
import activation as a
import util

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\hopfield.txt"
nl = "\n"
cut = "\n\n"

Y_ex = np.array([[ 1, 1, 1],
                 [-1,-1,-1]], dtype=float)
Y_img = np.array([ 1, 1, 1], dtype=float).reshape(1,3)
Y_ci = np.array([[1,-1,1,-1,1,-1],
                 [1,-1,1,1,-1,1],
                 [-1,-1,1,1,-1,-1]], dtype=float)

Y = Y_ci

x0 = np.array([-1,1,1], dtype=float)
x1 = np.array([-1,-1,1,1,-1,1], dtype=float)
x2 = np.array([1,1,-1,1,1,1], dtype=float)
X = x2

num_y = len(Y)
num_mem = Y.shape[1]

w = np.zeros((num_mem, num_mem), dtype=float)
T = np.full([1,num_mem], 0, dtype=float)


reportString += "input: " + nl + str(X) + nl
reportString += "memory: " + nl + str(Y) + nl



# --- Storage to weight ---
for i in range(0, num_y):
    y = Y[i].reshape(num_mem,1)
    w += y.dot(y.transpose())
W = w - np.diag(np.full(num_mem,num_y))
print(W)
print("\n")
reportString += "weight: " + nl + str(W) + nl

# --- Testing ---
t = T.reshape(num_mem,1)
for i in range(0, num_y):
    y = Y[i].reshape(num_mem,1)
    res = a.Sign(W.dot(y) - t)
    print(res.reshape(1,num_mem))

print("\n")

# --- Retrieval ---
epoch = 10
reportString += "epoch: " + str(epoch) + nl

t = T.reshape(num_mem, 1)
x = X.reshape(num_mem, 1)
for i in range (0, epoch):
    cont = False
    for j in range(0, num_y):
        y = Y[j].reshape(num_mem,1)
        res = a.Sign(W.dot(x) - t)
        if np.array_equal(y, res):
            counter = 2
        else:
            x = np.copy(res)
    reportString += "progression: " + nl + str(x.reshape(1,num_mem)) + nl

reportString += nl + "output: " + nl + str(x.reshape(1,num_mem)) + nl
util.WriteToFile(reportURL, reportString + cut)



