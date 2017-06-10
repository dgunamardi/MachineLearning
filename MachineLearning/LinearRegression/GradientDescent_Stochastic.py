import numpy as np
import CostFunction as cf
import math
import Plotter as plot


def GD(x, y, q, alpha, cvg_rate, x_avg = 25, y_avg = 250):
    cost_record = np.empty(0)
    counter = 0
    temp = cf.Cost(x,y,q)
    cost_record = np.append(cost_record, temp)
    accepted = 0
    cont = True
    while(cont):
        for i in range (0, len(q)):
            sum = 0
            for j in range(0, len(y)):
                q[i] = q[i] + alpha * ((y[j] - q[i] * x[i][j]) * x[i][j])

                #check convergency
                print(math.fabs(temp - cf.Cost(x,y,q)))
                if (ConvergenceCheck(temp, cf.Cost(x, y, q), cvg_rate)):
                    accepted += 1
                else:
                    accepted = 0
                if (accepted > 3):
                    cont = False
                    break
                if (accepted > 0):
                    msg = '''converging'''
                else:
                    msg = '''adjusting'''

                #update cost
                # print(temp)
                # print(cf.Cost(x, y, q))
                temp = cf.Cost(x, y, q)
                cost_record = np.append(cost_record, temp)
                counter += 1

                #plot
                #plot.Plot2D(x,y,q,msg,[x_avg,y_avg])

            if (not cont):
                break
        if (not cont):
            break

    plot.Plot2D(x, y, q, '''Final''', [x_avg, y_avg])
    plot.plt.plot(cost_record, 'bo')
    plot.plt.show()

    print(counter)


def ConvergenceCheck(cost1, cost2, cvg_rate):
    if (math.fabs(cost1[0] - cost2[0]) > cvg_rate):
        return False
    # if (math.fabs(cost1[1] - cost2[1]) > cvg_rate):
    #     return False
    return True
