import numpy as np
import activation as a
import util

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\bam.txt"
nl = "\n"
cut = "\n\n"

#---EXAMPLES---
Y_b = np.array([[1,1,1,1,1,1],
                [-1,-1,-1,-1,-1,-1],
                [1,1,-1,-1,1,1],
                [-1,-1,1,1,-1,-1]])
Yx_b = np.array([[1,1,1],
                 [-1,-1,-1],
                 [1,-1,1],
                 [-1,1,-1]])

#---EXERCISE---
Y1 = np.array([[1,1,1,1],
               [-1,-1,-1,-1],
               [-1,1,1,-1]
])
Y2 = np.array([[-1,-1,-1,-1,-1],
               [-1,-1,1,1,1],
               [-1,-1,1,-1,-1]
])

Y = Y1
Yx = Y2

#---EXAMPLES---
x1 = np.array([-1,1,1,1,1,-1], dtype=float)
x2 = np.array([1,1,-1,1,1,-1], dtype=float)
x3 = np.array([1,-1,1,-1,1,-1], dtype=float)


#---EXERCISE---
x4 = np.array([1,1,-1,1],dtype=float)
x5 = np.array([-1,1,1,1], dtype=float)
x6 = np.array([-1,1,-1,1], dtype=float)


X = x6


num_y = len(Y)
num_yx = len(Yx)
num_mem = Y.shape[1]
num_memx = Yx.shape[1]

w = np.zeros((num_mem, num_memx), dtype=float)

reportString += "input: " + nl + str(X) + nl
reportString += "memory1: " + nl + str(Y) + nl
reportString += "memory2: " + nl + str(Yx) + nl

# --- Storage to weight ---
for i in range(0, num_y):
    y = Y[i].reshape(num_mem,1)
    yx = Yx[i].reshape(1,num_memx)
    w += y.dot(yx)
W = w
print(W)
print("\n")
reportString += "weight: " + nl + str(W) + nl

# --- Testing ---
print("---Testing---")
for i in range(0, num_y):
    y = Y[i].reshape(num_mem,1)
    res = a.Sign(W.transpose().dot(y))
    print(res.reshape(1,num_memx))

for i in range(0, num_yx):
    y = Yx[i].reshape(num_memx,1)
    res = a.Sign(W.dot(y))
    print(res.reshape(1,num_mem))
print("\n")


# --- Retrieval ---
print("Retrieval")
epoch = 2
reportString += "epoch: " + str(epoch) + nl

x = X.reshape(num_mem, 1)

for i in range(0, epoch):
    cont = False
    for j in range(0, num_y):
        y = Yx[j].reshape(num_memx,1)
        res = a.Sign(W.transpose().dot(x))
        print(res.reshape(1, num_memx))
        resx = a.Sign(W.dot(res))
        print(resx.reshape(1, num_mem))
        if np.array_equal(y, resx):
            counter = 10
        else:
            x = np.copy(resx)
        #print(x.reshape(1,num_mem))
    reportString += "progression: " + nl + str(x.reshape(1, num_mem)) + nl

reportString += nl + "output: " + nl + str(x.reshape(1,num_mem)) + nl
util.WriteToFile(reportURL, reportString + cut)




