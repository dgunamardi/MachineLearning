import numpy as np
import activation as a
import util
from time import sleep

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\ffbp.txt"
nl = "\n"
cut = "\n\n"


t = np.array([
    [0],
    [1],
    [1],
    [0]
])

alpha = 0.1
x = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])

w1 = np.array([
    [0.5, 0.9],
    [0.4, 1.0],
])

w2 = np.array([
    [-1.2],
    [1.1]
])

b1 = np.array([
    [0.8, -0.1]
])

b2 = np.array([
    [0.3]
])



epoch = 4

reportString += "input: " + nl + str(x) + nl
reportString += "weight layer 1: " + nl + str(w1) + nl
reportString += "bias layer 1: " + nl + str(b1) + nl
reportString += "weight layer 2: " + nl + str(w2) + nl
reportString += "bias layer 2: " + nl + str(b2) + nl
reportString += "learning rate: " + str(alpha) + nl
reportString += "target: " + nl + str(t) + nl
reportString += "epoch: " + str(epoch) + cut


for i in range(0, epoch):
    for j in range(0, len(t)):
        # -- Forward --
        x_sample = x[j].reshape(1, x[j].shape[0])
        t_sample = t[j].reshape(1, t[j].shape[0])
        z = a.Sigmoid(x_sample.dot(w1) - b1)
        y = a.Sigmoid(z.dot(w2) - b2)

        # -- Backprop --
        e2 = t_sample - y
        g2 = y * (1-y) * e2
        w2_delta = alpha * np.dot(z.transpose(), g2)
        b2_delta = alpha * -1.0 * g2
        w2 = w2 + w2_delta


        e1 = np.dot(w2, g2.transpose()).transpose()
        g1 = z * (1-z) * e1
        w1_delta = alpha * np.dot(x_sample.transpose(), g1)
        b1_delta = alpha * -1.0 * g1
        w1 = w1 + w1_delta

        # reportString += "epoch: " + str(i) + "  iteration: " + str(j) + nl
        # reportString += "update weight layer 1: " + str(w1) + nl
        # reportString += "update bias layer 1: " + str(b1) + nl
        # reportString += "update weight layer 2: " + str(w2) + nl
        # reportString += "update bias layer 1: " + str(b2) + nl

z = a.Sigmoid(x.dot(w1) - b1)
y = a.Sigmoid(z.dot(w2) - b2)

reportString += nl + "final weight layer 1: " + nl + str(w1) + nl
reportString += "final bias layer 1: " + nl + str(b1) + nl
reportString += "final weight layer 2: " + nl + str(w2) + nl
reportString += "final bias layer 1: " + nl + str(b2) + nl
reportString += "final output: " + nl + str(y) + nl
util.WriteToFile(reportURL, reportString + cut)



print(w1)
print(b1)
print(w2)
print(b2)












