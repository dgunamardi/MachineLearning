import numpy as np
import util

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\som.txt"
nl = "\n"
cut = "\n\n"

x = np.array([
    [1,1,0,0],
    [0,0,0,1],
    [1,0,0,0],
    [0,0,1,1]
])

m = 2
a = 0.6
R = 0

w = np.array([
    [0.2, 0.6, 0.5, 0.9],
    [0.8, 0.4, 0.7, 0.3]
])

reportString += "input: " + nl + str(x) + nl
reportString += "weight: " + nl + str(w) + nl
reportString += "max cluster: " + str(m) + nl
reportString += "learning rate: " + str(a) + nl

# --- TRAINING ---
epoch = 2


for t in range(0, epoch):
    for i in range(0, len(x)):
        print("x ", i)
        res = w - x[i]
        res = res * res
        sum = np.sum(res, axis = 1, keepdims=True)
        print(sum)
        out = np.argmin(sum)
        print(sum[out])

        w[out] = w[out] + a * (x[i] - w[out])
        print(w)
        print("\n")
        reportString += "weight update: " + nl + str(w) + nl
    a = a/2
    print(a)
reportString += "final weight: " + nl + str(w) + nl
util.WriteToFile(reportURL, reportString + cut)