from PIL import Image
import numpy as np
from sift import GuassianKernel, convolve, getDoG

image = Image.open("images/img1.ppm")
imgarr = np.asarray(image)

T, S, Sigma = 0.04, 3, 1.6
DoG_Pyramid, Guass_Pyramid, O = getDoG(img=imgarr,n=1, sigma0=Sigma, S=S)


extremas = []
for i in range(len(DoG_Pyramid)):
    octave = DoG_Pyramid[i]
    points = []
    for j in range(1, len(octave)):
        getExtrema()


