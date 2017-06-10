import CostFunction as cf
import GradientDescent_Batch as gdb
import GradientDescent_Stochastic as gds
import numpy as np

n = 1
m = 10

x_area = np.random.uniform(10,50,(m))
x_class = np.random.uniform(1,10,(m))
x = np.array([x_area],dtype=float)

y = np.random.uniform(100,500,(m))

q = np.random.rand(n)

#.00001 (unstable)
#.000001

gds.GD(x,y,q,.000001, .1)

gdb.GD(x,y,q,.00001, .1)
