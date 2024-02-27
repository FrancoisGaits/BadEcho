import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig
import skimage.transform as tr
from skimage.restoration import denoise_nl_means

frame_spacing = 18
np.random.seed(21602426)

def gaus2d(x=0, y=0, mx=0, my=0, sx=0.6, sy=0.1):
    return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

x = np.linspace(-3, 3)
y = np.linspace(-3, 3)
x, y = np.meshgrid(x, y)
z = gaus2d(x, y)
z2 = gaus2d(x, y,my=0.05,sx=3,sy=2)

rd1 = None
rd2 = None

base = "frames/bad_apple_"
step = 0
start = 1
for frame in range(start,6563) :
    data = plt.imread("frames/bad_apple_{:03d}.png".format(frame))
    if len(data.shape) >= 3 :
        data = data[:,:,0]

    if data.shape[0] != 1080 or data.shape[1] != 1440 :
        data = tr.resize(data,(1080,1440))


    data = tr.rescale(data,0.5)


    max_energ = data.shape[0]*2.5

    print(frame)

    if frame == start :
        rd1 = np.reshape(np.random.randn(data.size),(data.shape[0],data.shape[1]))
        rd2 = np.reshape(np.random.randn(data.size),(data.shape[0],data.shape[1]))

    if step == frame_spacing :
        rd1 = rd2[:]
        rd2 = np.reshape(np.random.randn(data.size),(data.shape[0],data.shape[1]))
        step = 0

    rd = rd1 + (rd2-rd1) * (step/frame_spacing)
    scats = data * rd

    grad = np.gradient(data)
    grad = np.abs(grad[1]*0.01 + grad[0]*2.5)

    grad = sig.convolve2d(grad,z,mode="same",boundary="symm")


    mask = np.ones_like(data)
    for i in range(mask.shape[1]):
        count = max_energ
        for j in range(mask.shape[0]):
            if count > 0 :
                mask[j,i] = count/max_energ
                count -= 3*(j/mask.shape[0])+(abs(data[j,i])/4.5) + 0.75*grad[j,i]
            else :
                mask[j,i] = 0

    mask = np.clip(mask,0,1)
    res = sig.convolve2d(scats*mask*mask*mask, z, mode="same", boundary="symm")

    res = np.abs(res+((mask**4)*3*grad*res))

    res = abs(np.log(res+1))
    res = denoise_nl_means(res,h=0.24)
    res = tr.rescale(res,2)

    plt.imsave("res/bad_us_{:04d}.png".format(frame),res,cmap="gray")
    step += 1