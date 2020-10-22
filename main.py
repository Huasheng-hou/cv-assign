from PIL import Image
import numpy as np
from sift import getDoG, getExtremas
from visual import plotPoints

image = Image.open("images/img1.ppm")
image = image.convert('L')
imgarr = np.asarray(image)

T, S, Sigma = 0.04, 5, 1.6
SIFT_FIXPT_SCALE = 1
edgeThreshold = 10

'''高斯差分金字塔'''
# DoG_Pyramid, Guass_Pyramid, O = getDoG(img=imgarr, n=1, sigma0=Sigma, S=S)
O = 6
DoG_Pyramid = np.load('intermediates/DoG.npy', allow_pickle=True)

'''局部极值点检测'''
extremas = getExtremas(DoG_Pyramid=DoG_Pyramid, T=T, S=S)
# extremas = np.load('intermediates/extremas.npy', allow_pickle=True)
plotPoints(extremas=extremas, imgarr=imgarr)

def locateKpoints(DoG_Pyramid=None, extremas=None, contrastThreshold=None, edgeThreshold=None, ):

    img_scale = 1.0 / (255 * SIFT_FIXPT_SCALE)
    deriv_scale = img_scale * .5
    second_deriv_scale = img_scale
    cross_deriv_scale = img_scale * .25

    SIFT_MAX_INTERP_STEPS = 5
    SIFT_IMG_BORDER = 5

    for o in range(O):
        octave = extremas[o]
        for s in range(1, S - 1):
            img = DoG_Pyramid[o][s]
            prev = DoG_Pyramid[o][s - 1]
            next = DoG_Pyramid[o][s + 1]
            points = octave[s - 1]

            for x, y in points:

                i = 0
                while i < SIFT_MAX_INTERP_STEPS:
                    dD = [(img[x, y + 1] - img[x, y - 1]) * deriv_scale,  # 对x求偏导
                          (img[x + 1, y] - img[x - 1, y]) * deriv_scale,
                          (next[x, y] - prev[x, y]) * deriv_scale]
                    v2 = img[x, y] * 2
                    dxx = (img[x, y + 1] + img[x, y - 1] - v2) * second_deriv_scale
                    dyy = (img[x + 1, y] + img[x - 1, y] - v2) * second_deriv_scale
                    dss = (next[x, y] + prev[x, y] - v2) * second_deriv_scale
                    dxy = (img[x + 1, y + 1] - img[x + 1, y - 1] - img[x - 1, y + 1] + img[
                        x - 1, y - 1]) * cross_deriv_scale
                    dxs = (next[x, y + 1] - next[x, y - 1] - prev[x, y + 1] + prev[x, y - 1]) * cross_deriv_scale
                    dys = (next[x + 1, y] - next[x - 1, y] - prev[x + 1, y] + prev[x - 1, y]) * cross_deriv_scale

                    H = [[dxx, dxy, dxs],
                         [dxy, dyy, dys],
                         [dxs, dys, dss]]

                    X = np.matmul(np.linalg.pinv(np.array(H)), np.array(dD))

                    xi = -X[2]
                    xr = -X[1]
                    xc = -X[0]

                    if np.abs(xi) < 0.5 and np.abs(xr) < 0.5 and np.abs(xc) < 0.5:
                        break

                    '''插值'''
                    y += int(np.round(xc))
                    x += int(np.round(xr))
                    s += int(np.round(xi))

                    i += 1

                if i >= SIFT_MAX_INTERP_STEPS:
                    continue

                if s < 1 or s > S or y < SIFT_IMG_BORDER or y >= img.shape[1] - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= \
                    img.shape[0] - SIFT_IMG_BORDER:
                    continue

                t = (np.array(dD)).dot(np.array([xc, xr, xi]))

                contr = img[x, y] * img_scale + t * 0.5

                # 确定极值点位置第四步：舍去低对比度的点 :|fx| < T/n

                if np.abs(contr) * S < contrastThreshold:
                    return None, x, y, s