import numpy as np
import activation as a
import util
from time import sleep


reportString = "--- REGION --- \n"
reportURL = "\\Reports\\madaline.txt"
nl = "\n"
cut = "\n\n"

T = np.array([-1, 1, 1,-1])
alpha = .5
# [bias], [x1] , [x2]
x = np.array([
    [ 1, 1, 1],
    [ 1, 1,-1],
    [ 1,-1, 1],
    [ 1,-1,-1]
])

w = np.array([
    [[1, .3,.15],
     [0,.05, .1],
     [0, .2, .2]]
])

w_y = np.array([
    [[1, .5],
     [0, .5],
     [0, .5]]
])

reportString += "input: " + nl + str(x) + nl
reportString += "weight: " + nl + str(w) + nl
reportString += "weight to out: " + str(w_y) + nl
reportString += "learning rate: " + str(alpha) + nl
reportString += "target: " + nl + str(T) + nl

epoch = 2

reportString += "total epoch: " + str(epoch) + cut
for i in range( 0, epoch):
    print("---EPOCH--- ", i+1)
    for j in range(0, len(x)):
        #--- START TO NEURONS ---
        x_cur = x[j].reshape([3,1])
        z_in = x_cur.transpose().dot(w[0])
        Z = a.Sign(z_in)
        for k in range(1, len(w)):
            z_in = z_in.dot(w[k])
            Z = a.Sign(z_in)
        #--- TO OUTPUT ---
        y_in = Z.dot(w_y)
        Y = a.Sign(y_in)
        Y = Y.transpose()
        #--- CHECK Y ---
        if(Y[1] != T[j]):
            index = np.full(Z[0].shape, False, dtype=bool)
            if(T[j] == -1):
                index = Z[0] >= 0
            elif(T[j] == 1):
                index = Z[0] < 0
                #index[np.argmin(np.abs(z_in[0]))] = True
            index[0] = False

            w_delta = alpha * (x_cur.dot(T[j] - z_in))
            w_delta = w_delta.reshape(w.shape)
            w[:, :, index] = w[:, :, index] + w_delta[:, :, index]

        # reportString += "epoch: " + str(i) + "  iteration: " + str(j) + nl
        # reportString += "weight update: " + nl + str(w) + nl


reportString += nl + "final weight: " + nl + str(w) + nl
reportString += "final output: " + nl + str(a.Sign(x.dot(w))) + nl
util.WriteToFile(reportURL, reportString + cut)


