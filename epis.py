
from cupyx.scipy.sparse import identity, csr_matrix, hstack, vstack
from cupyx.scipy.sparse.linalg import spsolve
import numpy as np
import cupy as cp
import cv2
import sys
import random


def rgb2srgb(pixel_rgb):
    srgb = np.zeros_like(pixel_rgb, dtype=float)
    
    for i in range(0, pixel_rgb.shape[0]):
        value = float(pixel_rgb[i]) / 255
        if value < 0:
            pass
        elif value >= 0 and value < 0.0031308:
            value = value * 12.92
        else:
            value = 1.055 * (value ** (1/2.4)) + -0.055
        srgb[i] = value * 255

    return srgb


def rgb2lab(pixel_rgb):
    rgb = np.zeros_like(pixel_rgb, dtype=float)
    xyz = np.zeros_like(pixel_rgb, dtype=float)
    lab = np.zeros_like(pixel_rgb, dtype=float)
    
    for i in range(0, pixel_rgb.shape[0]):
        value = float(pixel_rgb[i]) / 255
        if value > 0.04045:
            value = ((value + 0.055) / 1.055) ** 2.4
        else:
            value = value / 12.92
        rgb[i] = value * 100

    xyz[0] = ((rgb[0] * 0.4124) + (rgb[1] * 0.3576) + (rgb[2] * 0.1805)) / 95.047
    xyz[1] = ((rgb[0] * 0.2126) + (rgb[1] * 0.7152) + (rgb[2] * 0.0722)) / 100.0
    xyz[2] = ((rgb[0] * 0.0193) + (rgb[1] * 0.1192) + (rgb[2] * 0.9505)) / 108.883

    for i in range(0, xyz.shape[0]):
        value = xyz[i]
        if value > 0.008856:
            value = value ** (0.3333333333333333)
        else:
            value = (7.787 * value) + (16 / 116)
        xyz[i] = value

    lab[0] = (116 * xyz[1]) - 16
    lab[1] = 500 * (xyz[0] - xyz[1])
    lab[2] = 200 * (xyz[1] - xyz[2])

    return lab


def affinity(pi_lab, pj_lab, p_kappa, p_sigma):
    """ 
    pi_lab: np.ndarray(3)
    pj_lab: np.ndarray(3)
    
    Calculate the affinity between CIELAB pixel_i and pixel_j.
    """

    pi_lab[0] = pi_lab[0] * p_kappa
    pj_lab[0] = pj_lab[0] * p_kappa

    return np.exp(- ((np.linalg.norm(pi_lab - pj_lab) ** 2) / 2 * (p_sigma ** 2)))


def compute_L(img, p_h, p_kappa, p_sigma):
    img = cp.asnumpy(img)
    n = img.shape[0] * img.shape[1]
    m_l = n * (p_h ** 2)

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    img_lab[:,:,0] = img_lab[:,:,0] / 100.0 * 10.0
    img_lab[:,:,1] = img_lab[:,:,1] / 220.0 * 120.0
    img_lab[:,:,2] = img_lab[:,:,2] / 220.0 * 120.0

    pair1 = cp.zeros((m_l), dtype=cp.uint32)
    pair2 = cp.zeros((m_l), dtype=cp.uint32)
    val_pos = cp.zeros((m_l), dtype=cp.float32)
    val_neg = cp.zeros((m_l), dtype=cp.float32)

    # TODO: Implement a faster approach instead of iteration
    k = 0
    for y1 in range(0, img_lab.shape[0]):
        for x1 in range(0, img_lab.shape[1]):
            pixel_i = img_lab[y1, x1]
            i = y1 * img_lab.shape[0] + x1

            for y2 in range(y1, y1 + p_h):
                for x2 in range(x1, x1 + p_h):
                    if y2 < img_lab.shape[0] and x2 < img_lab.shape[1]:
                        pixel_j = img_lab[y2, x2]
                        j = y2 * img.shape[0] + x2

                        pair1[k] = i 
                        pair2[k] = j 

                        result = affinity(pixel_i, pixel_j, p_kappa, p_sigma)
                        val_pos[k] = result
                        val_neg[k] = -result

                        k += 1

    rows = cp.asarray(np.arange(start=0, stop=m_l, dtype=np.float32), dtype=cp.float32)
    rows = cp.hstack([rows, rows])
    cols = cp.hstack([pair1, pair2])
    vals = cp.hstack([val_pos, val_neg])

    M = csr_matrix((vals, (rows, cols)), shape=(m_l, n))
    print("M:", M[0:10,0:10])
    O = csr_matrix((m_l, n), dtype=cp.float32)

    return vstack(
        [
            hstack([M, O, O]), 
            hstack([O, M, O]), 
            hstack([O, O, M])
        ],
    )


def rgb2z(img):
    z_r = img[:,:,0].flatten()
    z_g = img[:,:,1].flatten()
    z_b = img[:,:,2].flatten()
    return cp.concatenate((z_r.T, z_g.T, z_b.T)).T


def z2rgb(z, y, x):
    len = z.shape[0] // 3
    z_r = z[0:len]
    z_g = z[len:2*len]
    z_b = z[2*len:]
    img = cp.zeros((y, x, 3), dtype=np.uint8)
    img[:,:,0] = z_r.reshape((y, x))
    img[:,:,1] = z_g.reshape((y, x))
    img[:,:,2] = z_b.reshape((y, x))
    return img


def shrink(y, p_gamma):
    y_norm = cp.linalg.norm(y)
    return (y / y_norm) * max(y_norm - p_gamma, 0)


def image_flatten(img, p_iter, p_h, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma):
    zin = rgb2z(cp.asarray(img))
    z = cp.zeros((3, zin.shape[0]))
    z[0] = zin

    L = compute_L(z2rgb(z[0], img.shape[0], img.shape[1]), p_h, p_kappa, p_sigma)
    print("L.shape:", L.shape)
    LT = L.T
    print("LT.shape:", LT.shape)
    LTL = LT @ L
    print("LTL.shape:", LTL.shape)

    d = cp.zeros((3, L.shape[0]))
    b = cp.zeros((3, L.shape[0]))
    print("d[0].shape:", d[0].shape)

    I = identity(LTL.shape[0], dtype=cp.float64, format='csr')

    i = 0
    print("\n---- OPTIMIZATION ------------------")
    while i < p_iter and cp.linalg.norm(z[1] - z[0]) > p_epsilon:
        print("Iteration {}".format(i))
        print("difference:", cp.linalg.norm(z[1] - z[0]))
        print("")

        A = p_theta * I + p_lambda * LTL
        v = p_theta * zin + cp.asarray(p_lambda * LT @ (d[1] - b[1]))

        z[2] = spsolve(A, v)

        d[2] = shrink(L @ z[2] + b[1], 1.0 / p_lambda)
        b[2] = b[1] + L @ z[2] - d[2]

        for j in range(0, 2):
            z[j] = z[j+1]
            d[j] = d[j+1]
            b[j] = b[j+1]

        i = i + 1

    return z2rgb(z[2], img.shape[0], img.shape[1])

if __name__ == "__main__":
    filename = "test_data/car_50_50"
    filetype = ".jpg"
    img = cv2.imread(cv2.samples.findFile(filename + filetype))
    assert img is not None, "file could not be read"

    # print(img[0,0])
    # print(rgb2srgb(img[0,0]))
    # print(rgb2lab(img[0,0]))
    # print(rgb2lab(rgb2srgb(img[0,0])))
    # lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # print(lab[0,0])

    p_iter = 5
    p_h = 5
    p_epsilon = 0.001
    p_theta = 2.5
    p_lambda = 30
    p_kappa = 0.3
    p_sigma = 0.5
    flat_img = cp.asnumpy(image_flatten(img, p_iter, p_h, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma))
    cv2.imwrite(filename + "_flattened" + filetype, flat_img)
