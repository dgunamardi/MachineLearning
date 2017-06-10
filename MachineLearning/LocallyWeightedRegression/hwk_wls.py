import numpy as np
import matplotlib.pyplot as plt

def hwk_wls():
    m = 500
    x = np.random.rand(m,1) * 20

    X = np.concatenate((np.ones([m,1]), x), axis=1)

    y = np.sin(x) / x

    epsilon = np.random.randn(m,1) * np.std(y) / 5                   # noise
    y = y + epsilon

    tau = .1                                                         # bandwidth

    xpredict = np.linspace(np.min(x), np.max(x), num=100)
    ypredict = np.zeros_like(xpredict)

    for i in range(0, len(xpredict)):
        xval = xpredict[i]
        wts = np.zeros([m,1])
        theta = np.zeros([2,1])

        z = xval - x
        wts = np.exp(-np.abs(z) / (2*tau))
        theta = np.linalg.pinv(X.transpose().dot(np.diag(wts[:,0])).dot(X)).dot(X.transpose()).dot(np.diag(wts[:,0])).dot(y)

        ypredict[i] = np.array([1, xval]).dot(theta)

        plt.plot(x, y, 'ro', xpredict, ypredict, 'b-')
        plt.show()

    return


hwk_wls()
