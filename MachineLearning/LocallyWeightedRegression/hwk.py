import numpy as np
import matplotlib.pyplot as plt

def hwk_wls():
    m = 10
    x = np.random.uniform(0, 1, [m,1]) * 20
    X = np.array([np.ones([m,1]), x])
    y = np.sin(x) / x
    epsilon = np.random.normal(0, 1, [m,1]) * np.std(y) / 5           # noise

    y += epsilon

    tau = .9                                                    # bandwidth

    xpredict = np.linspace(np.min(x), np.max(x))
    ypredict = np.zeros_like(xpredict)

    for i in range(1,len(xpredict)):
        xval = xpredict[i]

        wts = np.zeros([m,1])
        theta = np.zeros([2,1])

        #CODE HERE FIND WEIGHTS

        z = xval - x
        wts = np.exp(-np.sqrt(z*z) / (2*tau))

        #Wut
        theta = np.linalg.pinv(X.conj().transpose() * np.diag(wts) * X) * X.conj().transpose() * np.diag(wts) * y
        #---

        ypredict[i] = np.concatenate(1,xval) * theta

    plt.plot(x,y,'ro',xpredict, ypredict,'b-')
    plt.legend('training data', ['weighted least squares linear fit, tau = ', str(tau)])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    return

def hwk_sigmoid(z):
    g = 1 / (1 + np.exp(-z))
    return g

nargin = 0
def hwk_logistic_regression(method):
    if (nargin < 1):
        method = 'gradient_ascent'

    x = np.loadtxt('logistic_data_x.txt')
    y = np.loadtxt('logistic_data_y.txt')
    m = x.shape[0]
    X = np.array([np.ones([m,1]), x])
    n = X.shape[1]

    theta = np.zeros([n,1])

    grad = np.ones([n,1])
    H = np.ones([n,n])
    alpha = .01

    accuracy = 0

    iteration = 0

    while(accuracy < .83) and (np.linalg.norm(H) > 1e-5) & (np.linalg.norm(grad) > 1e-5):
        iteration = iteration + 1

        #CODE TO COMPUTE ACCURACY

        h = hwk_sigmoid(X * theta)
        t_h = (h >= .5)
        accuracy = 1 - np.mean(np.abs(t_h - y))

        for k in range(1,n):
            sigma = 0
            for i in range(1,m):
                sigma = sigma + ((y[i]-h[i]) * X[i][k])
            grad[k] = sigma

        #compute the Hessian, H
        # G = np.zeros([m,m])
        # for i in range(1,m):
        #     G[i][i] = hwk_sigmoid(X[i,:]*theta)

        G = np.diag(h)
        H = X.conj().transpose() * G * (np.eye(m) - G) * X

        print(np.linalg.norm(H))
        print(np.linalg.norm(grad))
        print(accuracy)

        if (method == 'newton'):
            theta = theta + np.linalg.inv(H) * grad
        elif (method == 'gradient_ascent'):
            theta = theta + alpha * grad
        else:
            print('unknown learning algorithm')

    return

hwk_logistic_regression(method='gradient ascent')

def hwk_mllr():
    m = 100
    x = np.random.uniform(0,1,[m,1])
    y = x * 3
    epsilon = np.random.normal(0,1,[m,1]) * .1
    y = y + epsilon


    #Wut
    ml_theta = np.linalg.inv(x.transpose() * x) * x.transpose() * y
    xplot = np.array([np.min(x), np.max(x)])
    #---

    plt.plot(x,y,'rx', xplot, xplot * ml_theta, 'b-')
    plt.legend('training data', 'maximum likelihood fit')
    plt.xlabel('x')
    plt.ylabel('y')
    return





