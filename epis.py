
from cupyx.scipy.sparse import identity, csr_matrix, hstack, vstack
from cupyx.scipy.sparse.linalg import spsolve
from pathlib import Path
import numpy as np
import cupy as cp
import cv2
import sys
import random


def affinity(pi_lab: np.ndarray, pj_lab: np.ndarray, p_kappa: float, p_sigma: float) -> float:
    """ 
    Compute the affinity between two CIE-LAB pixels. In other words, how similar they are. 
    A higher affinity indicates more similarity, vice-versa.

    Parameters:
        pi_lab - a pixel of shape (3)
        pj_lab - a pixel of shape (3)
        p_kappa - ?
        p_sigma - lightness weight
    
    Output:
        affinity - the affinity between CIE-LAB pixels p_i and p_j, value ranges from [0.0, 1.0]
    """
    pi_lab[0] = pi_lab[0] * p_kappa
    pj_lab[0] = pj_lab[0] * p_kappa

    return np.exp(-p_sigma * np.linalg.norm(pi_lab - pj_lab))


def compute_pairs(
    img_lab: np.ndarray, 
    p_h: int, 
    p_kappa: float, p_sigma: float
) -> (cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray):
    """
    Computes several helpful arrays used in the efficient construction of the global matrix (M) used in compute_L().
    This is orders of magnitude faster to compute than an iterative approach to constructing the sparse global matrix.

    Pixel_i is the kth pixel in the image. Pixel_j is a pixel within Pixel_i's neighborhood.

    val_pos[k] = affinity(pair1[k], pair2[k])
    val_neg[k] = -1 * affinity(pair1[k], pair2[k])

    TODO: compute pair1, pair2, val_pos, val_neg using matrix vectorization instead of iteratively.

    Parameters:
        img_lab - a np.ndarray of shape (height, width, 3) where dimension [2] is a CIE-LAB pixel.
        p_h - the size of local neighborhood used in creating the M matrix
        p_kappa - ? 
        p_sigma - lightness weight in affinity function

    Output:
        pair1 - an array of absolute positions of pixel_i corresponding to pair2
        pair2 - an array of absolute positions of pixel_j corresponding to pair1
        val_pos - an array of affinities between pixel_i and pixel_j corresponding to pair1 and pair2
        val_neg - an array of negative affinities between pixel_i and pixel_j corresponding to pair1 and pair2
    """
    n = img_lab.shape[0] * img_lab.shape[1]
    m_l = n * (p_h ** 2)

    pair1 = cp.zeros((m_l), dtype=cp.uint64)
    pair2 = cp.zeros((m_l), dtype=cp.uint64)
    val_pos = cp.zeros((m_l), dtype=cp.float64)
    val_neg = cp.zeros((m_l), dtype=cp.float64)

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


def compute_max_grad(img: np.ndarray, pair1: np.ndarray, pair2: np.ndarray) -> np.ndarray:
    """
    Computes the maximum gradient along the lines between pixel_i and pixel_j represented by pair1 and pair2 respectively.

    Parameters:
        img - a np.ndarray of shape (height, width, 3) where dimension [2] is a RGB pixel.
        pair1 - an array of absolute positions of pixel_i corresponding to pair2
        pair2 - an array of absolute positions of pixel_j corresponding to pair1

    Output:
        max_gradients - a np.ndarray of shape (len(pair1), 1) where dimension [1] is the maximum gradient along 
            the path between pixel_i and pixel_j.
    """
    img = img.copy()
    img = img / 255.0
    gradients_x = np.mean(cv2.Sobel(src=img, ddepth=-1, dx=1, dy=0, ksize=3), 2)
    gradients_y = np.mean(cv2.Sobel(src=img, ddepth=-1, dx=0, dy=1, ksize=3), 2)
    gradients = np.sqrt(np.square(gradients_x) + np.square(gradients_y))

    max_gradients = np.zeros_like(pair1)

    i = 0
    for (p1, p2) in zip(pair1, pair2):
        x1 = int(p1 % img.shape[1])
        y1 = int(p1 // img.shape[1])
        x2 = int(p2 % img.shape[1])
        y2 = int(p2 // img.shape[1])

        if x2 > x1 and y2 > y1:
            xstep = 1
            ystep = 1
        elif x2 > x1 and y2 < y1:
            xstep = 1
            ystep = -1
        elif x2 < x1 and y2 > y1:
            xstep = -1
            ystep = 1
        else:
            xstep = -1
            ystep = -1

        dx = x1
        dy = y1
        max_gradient_along_line = 0.0

        j = 0
        while dx < x2 and dy < y2:
            if j > 5:
                break
            if gradients[dy, dx] > max_gradient_along_line:
                max_gradient_along_line = gradients[dy, dx]
            dx += xstep
            dy += ystep
            j += 1

        max_gradients[i] = max_gradient_along_line
        i += 1

    return max_gradients


def compute_L(
    img: np.ndarray, 
    p_h: int, 
    p_kappa: float, p_sigma: float, p_eta: float, preserve_edges: bool
) -> csr_matrix:
    """
    Compute the local sparsity matrix (L) used in the split-bergman algorithm.

    Parameters:
        img - a np.ndarray of shape (height, width, 3) where dimension [2] is a RGB pixel.
        p_h - the size of local neighborhood for computing pairs
        p_kappa - ?
        p_sigma - lightness weight used in affinity()
        p_eta - weight for maximum gradient
        preserve_edges - controls whether to preserve edges

    Output:
        L - a csr_matrix that is an identity of M matrices.
    """
    img = cp.asnumpy(img)
    n = img.shape[0] * img.shape[1]
    m_l = n * (p_h ** 2)

    img_lab = cp.array(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), dtype=cp.float64)

    img_lab[:,:,0] = (img_lab[:,:,0] / 100.0) * 10.0
    img_lab[:,:,1] = (img_lab[:,:,1] / 220.0) * 120.0
    img_lab[:,:,2] = (img_lab[:,:,2] / 220.0) * 120.0

    pair1, pair2, val_pos, val_neg = compute_pairs(img_lab, p_h, p_kappa, p_sigma)
    
    if preserve_edges:
        max_gradients = cp.array(compute_max_grad(img, cp.asnumpy(pair1), cp.asnumpy(pair2)))
        max_gradients = np.exp((max_gradients * p_eta) * -p_sigma)

        val_pos = cp.maximum(val_pos, max_gradients)
        val_neg = cp.minimum(val_neg, -1 * max_gradients)

    rows = cp.asarray(np.arange(start=0, stop=m_l, dtype=np.uint64), dtype=cp.uint64)
    rows = cp.hstack([rows, rows])
    cols = cp.hstack([pair1, pair2])
    vals = cp.hstack([val_pos, val_neg])

    M = csr_matrix((vals, (rows, cols)), shape=(m_l, n), dtype=cp.float64)
    O = csr_matrix((m_l, n), dtype=cp.float64)

    return vstack(
        [
            hstack([M, O, O]), 
            hstack([O, M, O]), 
            hstack([O, O, M])
        ],
    )


def rgb2z(img: np.ndarray) -> np.ndarray:
    """
    convert a rgb image of shape (height, width, 3) into an array of shape (height * width * 3)
    where each of the R, G, and B values are concatenated to form a single 1-D array.

    Parameters:
        img - a np.ndarray of shape (height, width, 3)

    Output:
        z - a np.ndarray of shape (height * width * 3)
    """
    z_r = img[:,:,0].flatten()
    z_g = img[:,:,1].flatten()
    z_b = img[:,:,2].flatten()
    return cp.concatenate((z_r.T, z_g.T, z_b.T)).T


def z2rgb(z: np.ndarray, height, width):
    """
    convert an array of shape (height * width * 3) that is the concatenation of the R, G, and B values of a image
    back into an image of shape (height, width, 3)

    Parameters:
        z - a np.ndarray of shape (height * width * 3)
        height - the original height of the image 
        width - the original width of the image

    Output:
        img - a np.ndarray of shape (height, width, 3)
    """
    assert z.shape[0] == (height * width * 3)

    len = z.shape[0] // 3
    z_r = z[0:len]
    z_g = z[len:2*len]
    z_b = z[2*len:]

    img = cp.zeros((height, width, 3), dtype=np.uint8)
    img[:,:,0] = z_r.reshape((height, width))
    img[:,:,1] = z_g.reshape((height, width))
    img[:,:,2] = z_b.reshape((height, width))
    return img


def shrink(y, p_gamma):
    """
    ???

    Parameters:
        y - ?
        p_gamma - the regularization term weight
    
    Output:
        d - ?
    """
    y_norm = cp.linalg.norm(y)
    return (y / y_norm) * max(y_norm - p_gamma, 0)


def image_flatten(
    img: np.ndarray, 
    p_iter: int, p_h: int, 
    p_alpha: float, p_epsilon: float, p_theta: float, p_lambda: float, p_kappa: float, p_sigma: float, p_eta: float,
    preserve_edges: bool
) -> np.ndarray:
    """
    L1 Flatten an image using the provided parameters.

    Parameters:
        img - a np.ndarray of shape (height, width, 3)
        p_iter - the number of iterations for the L1 energy optimization algorithm
        p_h - the size of local neighborhood used in creating the M matrix
        p_alpha - the local sparseness weight
        p_epsilon - a threshold to control the L1 energy optimization algorithm to stop
        p_theta - the image approximation term
        p_lambda - the regularization term weight (controls how "flat" the image should become)
        p_kappa - ?
        p_sigma - lightness weight in affinity function
        p_eta - weight for maximum gradient (controls the prevalence of the gradients used for edge-preserving)
        preserve_edges - controls whether edge preserving is applied

    Output:
        img_flat - a np.ndarray of shape (height, width, 3) that is the flattened version of `img`
    """
    zin = rgb2z(cp.asarray(img))
    z = cp.zeros((3, zin.shape[0]), dtype=cp.float64)
    z[0] = zin

    L = p_alpha * compute_L(z2rgb(z[0], img.shape[0], img.shape[1]), p_h, p_kappa, p_sigma, p_eta, preserve_edges)
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
        print("Difference: {}\n".format(cp.linalg.norm(z[1] - z[0])))

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
    if len(sys.argv) != 2:
        print("Usage: python epis.py [filepath of image]")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    if not filepath.exists():
        print("Filepath `{}` does not exist.".format(filepath))
        sys.exit(1)
    if filepath.is_dir():
        print("Filepath `{}` is a directory. Filepath must be a file.".format(filepath))
        sys.exit(1)
    if not filepath.suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        print("Filepath `{}` is not a supported filetype.".format(filepath))
        sys.exit(1)

    img = cv2.imread(cv2.samples.findFile(str(filepath)))
    assert img is not None, "File `{}` could not be read".format(filepath)

    # default parameters
    p_iter = 4
    p_h = 5
    p_alpha = 20.0
    p_epsilon = 0.01
    p_theta = 50.0
    p_lambda = 256.0
    p_kappa = 1.0
    p_sigma = 0.9
    p_eta = 0.1
    preserve_edges = False

    img_flat = cp.asnumpy(image_flatten(
        img, p_iter, p_h, p_alpha, p_epsilon, p_theta, p_lambda, p_kappa, p_sigma, p_eta, preserve_edges
    ))

    if preserve_edges:
        cv2.imwrite(filename + "_smoothed_256" + filetype, img_flat)
    else:
        cv2.imwrite(filename + "_flattened_256" + filetype, img_flat)        
