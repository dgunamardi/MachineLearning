import numpy as np
import util
import math
import activation as a
import cost


reportString = "--- REGION --- \n"
reportURL = "\\Reports\\autoencoder.txt"
nl = "\n"
cut = "\n\n"


im_w = 100
im_h = 100

divisor = 4

im_noised_w = 50
im_noised_h = 50

#--config working--
# 100x100
# 25 it
# 0.0001 alpha

ext = ".jpg"
result_ext = "after.jpg"
noised_ext = "noised.jpg"
normal_ext = "normal.jpg"
code_ext = "code.jpg"

url = np.array(["\\AutoEncoder\\Datasets\\simf1",
                "\\AutoEncoder\\Datasets\\simf2",
                "\\AutoEncoder\\Datasets\\sim"])
#url = np.array(["\\AutoEncoder\\Datasets\\simf1"])

# --- INITIALIZATION ---
print("---init---")
# -- get 1d array from image --

data = util.LoadImage(url, ext, im_w, im_h)
data = np.delete(data, 0, axis=0)
util.SaveAsImage(data, url, normal_ext, 200, 200, im_w, im_h)
util.Display(data, 200, 200, im_w, im_h)

data_scaled = util.Scale(data, 0, 255, -1,1).astype(np.float64)

# -- apply noise --
noise_mult = 0.9
data_noised = util.Gaussian_Noise(data, noise_mult).astype(np.float64)
util.Display(data_noised, 200, 200, im_w, im_h)
util.SaveAsImage(data_noised, url, noised_ext, 200, 200, im_w, im_h)

# -- set input --
input = util.Scale(data_noised, 0, 255, -1, 1).astype(np.float64)  # tanh

# -- set neuron count --
n_input = np.int(im_w * im_h)
n_hidden = np.int(n_input / divisor)

# -- init weight --
bh = np.zeros([1,n_hidden])
wh = np.random.uniform((-1.0 / math.sqrt(n_input)), 1.0 / math.sqrt(n_input), (n_input, n_hidden)).astype(np.float64) # tanh

bo = np.zeros([1,n_input])
wo = wh.transpose()

# -- learning rate -- MUST BE ADJUSTED ACCORDINGLY TO IMAGE RESOLUTION
alpha = 0.0001

print("\n")

# --- TRAINING ITERATION ---
n_iteration = 100

reportString += "original url: " + nl + str(url) + nl
reportString += "input (clean) suffix: " + normal_ext + nl
reportString += "input (noised) suffix: " + noised_ext + nl
reportString += "output (training result) suffix: " + result_ext + nl
reportString += "noise multiplier: " + str(noise_mult) + nl
reportString += "Initial Error: " + str(cost.MSE(data_scaled, input)) + nl
reportString += "epoch: " + str(n_iteration) + nl
reportString += "learning rate: " + str(alpha) + nl

for i in range(0, n_iteration):
    n = np.random.randint(0, input.shape[0])
    sample = np.copy(input[n]).reshape(1, input[n].shape[0])
    clean = data_scaled[n].reshape(sample.shape)

    print(i, "iteration")
    # --- FORWARD PASS ---

    # - HIDDEN LAYER -
    code = a.Tanh(np.dot(sample, wh))
    im_code = util.Scale(code, -1, 1, 0, 255).astype(np.uint8)

    # - OUTPUT LAYER -
    output = a.Tanh(np.dot(code, wo))

    # --- BACKPROPAGATION ---
    w = np.copy(wh)


    e2 = clean - output
    g2 = (1 - np.power(output, 2)) * e2
    w_delta_2 = alpha * np.dot(code.transpose(), g2)

    e1 = np.dot(wo, g2.transpose()).transpose()
    g1 = (1 - np.power(code, 2)) * e1
    w_delta_1 = alpha * np.dot(clean.transpose(), g1)

    w_delta = w_delta_1 + w_delta_2.transpose()
    w = w + w_delta

    wh = np.copy(w)
    wo = wh.transpose()

# --- TESTING ---
print("\n --- TESTING ---")

y = a.Tanh(np.dot(data_scaled, wh))
# util.Display(im_code, 200, 200, im_noised_w, im_noised_h)
# util.SaveAsImage(im_code, url, code_ext, 200, 200, im_noised_w, im_noised_h)

# -- image for hidden 0 THIS IS YET WORKING --
# im_code = np.zeros((1, 2500), dtype=np.uint8)
# for i in range(0, 2500):
#     im_code[0,i] = np.divide(wh[0,i], np.sqrt(np.sum(np.power(w[0,i], 2)))).astype(np.uint8)
# util.Display(im_code, 200, 200, im_noised_w, im_noised_h)
# util.SaveAsImage(im_code, url, code_ext, 200, 200, im_noised_w, im_noised_h)

z = a.Tanh(np.dot(y, wh.transpose()))

im_z = util.Scale(z, -1, 1, 0, 255).astype(np.uint8)
util.Display(im_z,200,200,im_w,im_h)
util.SaveAsImage(im_z,url, result_ext, 200,200, im_w, im_h)


reportString += "Error Rate: " + str(cost.MSE(data_scaled, z)) + nl
util.WriteToFile(reportURL,reportString + cut)






