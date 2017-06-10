import numpy as np
import activation as a
import util
from time import sleep

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\perceptron.txt"
nl = "\n"
cut = "\n\n"

y = np.array([ 0, 0, 1, 1])
alpha = .1
# [bias], [x1] , [x2]
x = np.array([
    [ 0, 0],
    [ 0, 1],
    [ 1, 0],
    [ 1, 1]
])

w = np.array([
    [[.1],
     [-.3]]
])

epoch = 2

reportString += "input: " + nl +str(x) + nl
reportString += "weight: " + nl + str(w) + nl
reportString += "learning rate: " + str(alpha) + nl
reportString += "target: " + nl + str(y) + nl
reportString += "total epoch: " + str(epoch) + cut

for i in range (0, epoch):
    print("---- EPOCH ---- ", i + 1)
    for j in range(0, len(x)):
        #--- START TO NEURONS ---
        x_cur = x[j].reshape([2,1])
        z_in = x_cur.transpose().dot(w[0])
        z = a.Step(z_in)
        for k in range(1, len(w)):
            z_in = z_in.dot(w[k])
            z = a.Step(z_in)

        #--- CHECK Y ---
        temp = np.copy(w)
        # index = np.full(Z[0].shape, True, dtype=bool)
        # index[0] = False
        e = y[j]- z
        w_delta = alpha * x_cur * e
        w = w + w_delta

        # reportString += "epoch: " + str(i) + "  iteration: "+ str(j) + nl
        # reportString += "weight update: " + nl + str(w) + nl


print(a.Step(x.dot(w)))

reportString += nl + "final weight: " + nl + str(w) + nl
reportString += "final output: " + nl + str(a.Step(x.dot(w))) + nl
util.WriteToFile(reportURL, reportString + cut)
