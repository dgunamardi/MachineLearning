import numpy as np
from matplotlib import pyplot as plt

n = 10;
x = np.random.uniform(0.0, 5.0, (n, 2))
x = np.array([
    [1, 1],
    [1.2, 1.3],
    [1, 1.8],
    [2, 1.2],
    [.5, 2],
    [4, 4],
    [3.4, 2.7],
    [3.9, 4.3],
    [3.5, 2.6],
    [4.4, 4.5]
])
c = np.zeros([n,1])

k = 2;
miu = np.random.uniform(0.0, 5.0, (k, 2))

plt.plot(x[:,0],x[:,1], 'ro', miu[:,0], miu[:,1], 'bx')
plt.show()

reps = 5


for i in range(0, reps):
    for i in range(0, n):
        temp = np.zeros([k,1])
        for j in range(0, k):
            temp[j] = np.linalg.norm(miu[j] - x[i])
        c[i] = np.argmin(temp)

    print(x)
    print(miu)

    for j in range(0, k):
        indices = (c == j).reshape(n)
        miu[j] = sum(x[indices]) / len(x[indices])
        print(indices)
        print(miu[j])

    plt.plot(x[:,0],x[:,1], 'ro', miu[:,0], miu[:,1], 'bx')
    plt.show()