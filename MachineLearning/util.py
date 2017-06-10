import numpy as np
import cv2
import os

def LoadImage(url, ext, x, y):
    im_arr = np.empty((1,x*y), dtype=np.uint8)
    for u in url:
        u = os.getcwd() + u + ext
        im = cv2.imread(u, cv2.IMREAD_GRAYSCALE)
        im = ConvertDim(cv2.resize(im, (x, y)))
        im_arr = np.append(im_arr,im, axis=0)
    return im_arr

def SaveAsImage(data, url, ext, fx, fy, rx, ry):
    for i in range(0, data.shape[0]):
        u = os.getcwd() + url[i] + ext
        im = ConvertDim(data[i],rx,ry)
        im = cv2.resize(im, (fx, fy))
        cv2.imwrite(u, im)

def Display(data, fx, fy, rx, ry):
    for d in data:
        temp = ConvertDim(d, rx, ry).astype(np.uint8)
        cv2.namedWindow("test", flags= cv2.WINDOW_NORMAL)
        cv2.resizeWindow("test", fx, fy)
        cv2.imshow("test", temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def ConvertDim(data, dx = 0, dy = 0):
    if (dx == dy == 0):
        return data.reshape(1, data.shape[0] * data.shape[1])
    else:
        return data.reshape(dx, dy)


def Gaussian_Noise(data, alpha):
    data_scaled = Scale(data, 0, 255, -1, 1)
    n = np.random.normal(1, 0.5, data.shape).astype(np.float64)
    noise = alpha * Scale(n, n.min(), n.max(), -1, 1)
    data_unscaled = Scale((data_scaled + noise), -1, 1,0, 255)
    data_unscaled = Threshold(data_unscaled, 0, 255)

    return data_unscaled.astype(dtype=np.uint8)

def Scale(data, lower, upper, newmin, newmax):
    temp = np.copy(data)
    lower = np.full(data.shape, lower)
    upper = np.full(data.shape, upper)
    newmin = np.full(data.shape, newmin)
    newmax = np.full(data.shape, newmax)

    temp = newmin + np.divide(np.multiply((temp - lower), (newmax-newmin)), upper - lower)
    return temp.astype(np.float64);

def Threshold(data, lower, upper):
    out_lb = data < lower
    out_ub = data > upper
    temp = np.copy(data)
    temp[out_lb] = lower
    temp[out_ub] = upper
    return temp;

def WriteToFile(url, text):
    u = os.getcwd() + url
    f = open(u, 'a')
    f.write(text)
    f.close()
