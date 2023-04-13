
from cupyx.scipy.sparse import identity, csr_matrix
from cupyx.scipy.sparse.linalg import spsolve
import numpy as np
import cupy as cp
import cv2
import sys
import random


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


def affinity(pi_lab, pj_lab):
    """ 
    pi_lab: np.ndarray(3)
    pj_lab: np.ndarray(3)
    
    Calculate the affinity between CIELAB pixel_i and pixel_j.
    """

    # constants
    kappa = 0.3     # sensitivity to luminance variations
    sigma = 1.0     # metric for pixel affinity

    pi_lab[0] = pi_lab[0] * kappa
    pj_lab[0] = pj_lab[0] * kappa

    return np.exp(- (np.square(np.linalg.norm(pi_lab - pj_lab)) / 2 * (sigma ** 2)))


# Source: https://gist.github.com/pghazanfari/b23739a0a057e165fa8b591a64ca181d
def surrounding(x, idx, radius=1, fill=0):
    """ 
    Gets surrounding elements from a numpy array 
  
    Parameters: 
    x (ndarray of rank N): Input array
    idx (N-Dimensional Index): The index at which to get surrounding elements. If None is specified for a particular axis,
        the entire axis is returned.
    radius (array-like of rank N or scalar): The radius across each axis. If None is specified for a particular axis, 
        the entire axis is returned.
    fill (scalar or None): The value to fill the array for indices that are out-of-bounds.
        If value is None, only the surrounding indices that are within the original array are returned.
  
    Returns: 
    ndarray: The surrounding elements at the specified index
    """
    
    assert len(idx) == len(x.shape)
    
    if np.isscalar(radius): radius = tuple([radius for i in range(len(x.shape))])
    
    slices = []
    paddings = []
    for axis in range(len(x.shape)):
        if idx[axis] is None or radius[axis] is None:
            slices.append(slice(0, x.shape[axis]))
            paddings.append((0, 0))
            continue
            
        r = radius[axis]
        l = idx[axis] - r 
        r = idx[axis] + r
        
        pl = 0 if l > 0 else abs(l)
        pr = 0 if r < x.shape[axis] else r - x.shape[axis] + 1
        
        slices.append(slice(max(0, l), min(x.shape[axis], r+1)))
        paddings.append((pl, pr))
    
    if fill is None: return x[slices]
    return np.pad(x[slices[0], slices[1], slices[2]], paddings, 'constant', constant_values=fill)


def local_neighborhood(img, y, x, h):
    """
    img: np.ndarray(i, j, 3)
    y: int < i
    x: int < j
    h: int < min(i, j)

    return the (h x h) local neighborhood of pixel_yx.
    """
    assert y < img.shape[0], "pixel out of range"
    assert x < img.shape[1], "pixel out of range"
    assert h < min(img.shape[0], img.shape[1]), "neighborhood too big"

    return surrounding(img, (y, x, None), (h, h, None))

def compute_L(img, h=11):
    img = cp.asnumpy(img)
    m_l = (2*h + 1) ** 2
    n = img.shape[0] * img.shape[1]

    print("m_l:", m_l)
    print("n:", n)

    img_lab = np.zeros_like(img, dtype=float)
    for y1 in range(0, img.shape[0]):
        for x1 in range(0, img.shape[1]):
            img_lab[y1, x1] = rgb2lab(img[y1, x1])

    M = np.zeros((m_l, n))

    for y1 in range(0, img.shape[0]):
        for x1 in range(0, img.shape[1]):
            window = local_neighborhood(img_lab, y1, x1, h)
            middle_y = window.shape[0] // 2
            middle_x = window.shape[1] // 2

            for y2 in range(0, window.shape[0]):
                for x2 in range(0, window.shape[1]):
                    pixel_j = window[y2, x2]

                    # map local neighborhood coordinates to global 
                    gy2 = y1 - middle_y + y2
                    gx2 = x1 - middle_x + x2
                    
                    k = y2 * window.shape[0] + x2
                    i = y1 * img.shape[0] + x1
                    j = gy2 * img.shape[0] + gx2

                    if j < img.shape[0] * img.shape[1]:
                        result = affinity(img_lab[y1, x1], pixel_j)
                        M[k, i] = result
                        M[k, j] = -result

    M = cp.asarray(M)
    O = cp.zeros_like(M)

    return cp.concatenate(
        [
            cp.concatenate([M, O, O], axis=1), 
            cp.concatenate([O, M, O], axis=1), 
            cp.concatenate([O, O, M], axis=1)
        ], 
        axis=0
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


def image_flatten(img, p_h, p_epsilon, p_beta, p_lambda):
    zin = rgb2z(cp.asarray(img))
    z = cp.zeros((3, zin.shape[0]))
    z[0] = zin

    print("zin.shape:", zin.shape)

    L = csr_matrix(compute_L(img, p_h))
    d = cp.zeros((3, L.shape[0]))
    b = cp.zeros((3, L.shape[0]))

    print("L.shape:", L.shape)
    LT = L.transpose()
    print("LT.shape:", LT.shape)
    LTL = LT.multiply(L.toarray())
    print("LTL.shape:", LTL.shape)
    I = identity(LTL.shape[0], dtype='float', format='csr')
    print("I.shape:", I.shape)

    i = 0
    print("\n---- OPTIMIZATION ------------------")
    while (cp.linalg.norm(z[1] - z[0]) ** 2) > p_epsilon:
        print("Iteration {}".format(i))
        print("difference:", cp.linalg.norm(z[1] - z[0]) ** 2)
        print("")

        A = p_beta * I + p_lambda * LTL
        v = csr_matrix(p_beta * zin) + p_lambda * LT.multiply(csr_matrix(d[1] - b[1]))

        z[2] = spsolve(A, v.toarray())

        d[2] = shrink(L.multiply(z[2]) + b[1], 1 / p_lambda)
        b[2] = b[1] + L.multiply(z[2]) - d[2]

        for i in range(0, 2):
            z[i] = z[i+1]
            d[i] = d[i+1]
            b[i] = b[i+1]

        i = i + 1

    return z2rgb(z[2], img.shape[0], img.shape[1])

if __name__ == "__main__":
    filename = "test_data/car_50_50"
    filetype = ".jpg"
    img = cp.asarray(cv2.imread(cv2.samples.findFile(filename + filetype)))
    assert img is not None, "file could not be read"

    flat_img = cp.asnumpy(image_flatten(img, 11, 1.0, 2.5, 5.0))
    cv2.imwrite(filename + "_flattened" + filetype, flat_img)
