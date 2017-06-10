import numpy as np
import util

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\maxnet.txt"
nl = "\n"
cut = "\n\n"

a_exa = np.array([0.2, 0.4, 0.6, 0.8])
e_exa = np.full(a_exa.shape, -0.2)

a_exe = np.array([.7, .3, .5, .9, .1])
e_exe = np.full(a_exe.shape, -0.2)

a = np.copy(a_exe)
e = np.copy(e_exe)

reportString += "neurons: " + nl + str(a) + nl
reportString += "e: " + nl + str(e) + nl

while(True):
    a_old = np.copy(a)
    for i in range(0, len(a)):
        mask = np.full(a_old.shape, True, dtype=bool)
        mask[i] = False
        a[i] = max(0, a_old[i] + np.sum(np.multiply(e[mask], a_old[mask])))
    print(a)

    reportString += "progression = " + nl + str(a) + nl
    #check nonzero nodes
    out = a > 0
    res = a[out]
    if(len(res) == 1):
        break

reportString += nl + "result = " + nl + str(a) + nl
util.WriteToFile(reportURL, reportString + cut)