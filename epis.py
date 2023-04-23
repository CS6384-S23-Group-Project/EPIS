
from cupyx.scipy.sparse import identity, csr_matrix, hstack, vstack
from cupyx.scipy.sparse.linalg import spsolve
import numpy as np
import cupy as cp
import cv2
import sys
import random


def affinity(pi_lab, pj_lab, p_kappa, p_sigma):
    """ 
    pi_lab: np.ndarray(3)
    pj_lab: np.ndarray(3)
    
    Calculate the affinity between CIELAB pixel_i and pixel_j.
    """

    pi_lab[0] = pi_lab[0] * p_kappa
    pj_lab[0] = pj_lab[0] * p_kappa

    return np.exp(- np.linalg.norm(pi_lab - pj_lab) * p_sigma)


def compute_pairs(img_lab, p_h, p_kappa, p_sigma):
    n = img_lab.shape[0] * img_lab.shape[1]
    m_l = n * (p_h ** 2)

    pair1 = cp.zeros((m_l), dtype=cp.uint64)
    pair2 = cp.zeros((m_l), dtype=cp.uint64)
    val_pos = cp.zeros((m_l), dtype=cp.float64)
    val_neg = cp.zeros((m_l), dtype=cp.float64)
    
    print("img_lab.shape:", img_lab.shape)

    k = 0
    for y1 in range(0, img_lab.shape[0]):
        for x1 in range(0, img_lab.shape[1]):
            pixel_i = img_lab[y1, x1]

            i = y1 * img_lab.shape[1] + x1

            for y2 in range(y1, y1 + p_h):
                for x2 in range(x1, x1 + p_h):
                    if y2 < img_lab.shape[0] and x2 < img_lab.shape[1]:
                        pixel_j = img_lab[y2, x2]
                        j = y2 * img_lab.shape[1] + x2
                        
                        pair1[k] = i 
                        pair2[k] = j 

                        result = affinity(pixel_i, pixel_j, p_kappa, p_sigma)
                        val_pos[k] = result
                        val_neg[k] = -result

                        k += 1

    return (pair1, pair2, val_pos, val_neg)


def compute_L(img, p_h, p_kappa, p_sigma):
    img = cp.asnumpy(img)
    print("img.shape:", img.shape)
    n = img.shape[0] * img.shape[1]
    m_l = n * (p_h ** 2)

    img_lab = cp.array(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), dtype=cp.float64)

    img_lab[:,:,0] = (img_lab[:,:,0] / 100.0) * 10.0
    img_lab[:,:,1] = (img_lab[:,:,1] / 220.0) * 120.0
    img_lab[:,:,2] = (img_lab[:,:,2] / 220.0) * 120.0

    pair1, pair2, val_pos, val_neg = compute_pairs(img_lab, p_h, p_kappa, p_sigma)

    rows = cp.asarray(np.arange(start=0, stop=m_l, dtype=np.uint64), dtype=cp.uint64)
    rows = cp.hstack([rows, rows])
    cols = cp.hstack([pair1, pair2])
    vals = cp.hstack([val_pos, val_neg])

    print("m_l:", m_l)
    print("n:", n)
    print("rows.shape:", rows.shape)
    print("cols.shape:", cols.shape)
    print("cp.max(rows):", cp.max(rows))
    print("cp.max(cols):", cp.max(cols))

    M = csr_matrix((vals, (rows, cols)), shape=(m_l, n), dtype=cp.float64)
    print("M:", M[0:10,0:10])
    O = csr_matrix((m_l, n), dtype=cp.float64)

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


def shrink2(y, p_gamma):
    # x = cp.copy(y)
    # y = cp.max(cp.abs(y) - p_gamma, 0)
    # y = cp.where(x < 0, y, -1 * y)
    # return y
    pass


def image_flatten(img, p_iter, p_h, p_alpha, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma):
    zin = rgb2z(cp.asarray(img))
    z = cp.zeros((3, zin.shape[0]), dtype=cp.float64)
    z[0] = zin

    L = p_alpha * compute_L(z2rgb(z[0], img.shape[0], img.shape[1]), p_h, p_kappa, p_sigma)
    LT = L.T
    LTL = LT @ L

    d = cp.zeros((3, L.shape[0]), dtype=cp.float64)
    b = cp.zeros((3, L.shape[0]), dtype=cp.float64)

    I = identity(LTL.shape[0], dtype=cp.float64, format='csr')
    A = p_theta * I + p_lambda * LTL

    i = 0
    print("\n---- OPTIMIZATION ------------------")
    while i < p_iter and cp.linalg.norm(z[1] - z[0]) > p_epsilon:
        print("Iteration {}".format(i))
        print("difference:", cp.linalg.norm(z[1] - z[0]))
        print("")

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


def modular_flatten(img, width, height):
    img_flat = np.copy(img)

    p_iter = 4
    p_h = 5
    p_alpha = 20.0
    p_epsilon = 0.001
    p_theta = 50.0
    p_lambda = 128.0
    p_kappa = 1.0
    p_sigma = 0.9

    if (img.shape[0] == height) or (img.shape[1] == width):
        img_flat = cp.asnumpy(image_flatten(img, p_iter, p_h, p_alpha, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma))
    else:
        for h in range(0, (img.shape[0] // height) + 1):
            for w in range(0, (img.shape[1] // width) + 1):
                if (h * height == img.shape[0]) or (w * width == img.shape[1]):
                    continue

                print(range(w*width, (w+1)*width))
                print(range(h*height, (h+1)*height))
                cropped_img = img[h*height:(h+1)*height,w*width:(w+1)*width,:]
                print("cropped_img.shape:", cropped_img.shape)
                cropped_img_flat = cp.asnumpy(image_flatten(cropped_img, p_iter, p_h, p_alpha, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma))
                img_flat[h*height:(h+1)*height,w*width:(w+1)*width,:] = cropped_img_flat

    return img_flat


if __name__ == "__main__":
    filename = "test_data/FLIR_00010_640_512"
    filetype = ".png"
    img = cv2.imread(cv2.samples.findFile(filename + filetype))
    assert img is not None, "file could not be read"

    img_flat = modular_flatten(img, 160, 126)
    cv2.imwrite(filename + "_flattened" + filetype, img_flat)
