import numpy as np
import util

reportString = "--- REGION --- \n"
reportURL = "\\Reports\\mexicanhat.txt"
nl = "\n"
cut = "\n\n"

#1 = inner rad, 2 = outer rad
r1_exa = 1
r2_exa = 2
c1_exa = 0.6
c2_exa = -0.4

x_exa = np.array([.0, .5, .8, 1.0, .8, .5, .0])
x_max_exa = 2
t_max_exa = 2


r1_exe = 1
r2_exe = 3
c1_exe = 0.4
c2_exe = -0.5

x_exe = np.array([.5, 1.0, .3, .0, .4, .8, .2, .7, .9])
x_max_exe = 1.5
t_max_exe = 3


r1 = r1_exa
r2 = r2_exa
c1 = c1_exa
c2 = c2_exa
x = np.copy(x_exa)
x_max = x_max_exa
t_max = t_max_exa
x_end = len(x)


reportString += "neurons: " + nl + str(x) + nl
reportString += "epoch: " + nl + str(t_max) + nl
reportString += "x_max: " + str(x_max) + nl
reportString += "r1: " + str(r1) + " r2: " + str(r2) + nl
reportString += "c1: " + str(c1) + " c2: " + str(c2) + nl

for t in range(0, t_max):
    x_old = np.copy(x)
    for i in range(0, len(x_old)):
        start = 0 if i - r2 < 0 else i - r2
        end = x_end if i + r2 + 1 > x_end else i + r2 + 1
        x[i] = 0
        for j in range(start, end):
            if(abs(j - i) <= r1):
                x[i] += c1 * x_old[j]
            else:
                x[i] += c2 * x_old[j]
        x[i] = min(x_max, max(0,x[i]))
        reportString += "progression: " + nl + str(x) + nl

reportString += nl + "result: " + nl + str(x) + nl
util.WriteToFile(reportURL, reportString + cut)