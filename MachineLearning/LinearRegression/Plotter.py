import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def Plot2D(x,y,q,msg, msg_loc):
    plt.plot(x[0],y, 'ro', x[0],x[0] * q, 'b-')
    plt.text(msg_loc[0], msg_loc[1], msg)
    plt.show()

def Plot3D(x,y,q):
    #---------------REVISE THIS---------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.scatter(x[0], x[1], y, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    #fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.plot(x[0], x[1], x[0]*q[0] + x[1]*q[1])
    plt.show()

