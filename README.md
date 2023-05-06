# EPIS
An implementation of the Edge-Preserving Image Smoothing (EPIS) algorithm discussed in this [paper](https://cseweb.ucsd.edu/~bisai/papers/SIGGRAPH15_IntrinsicDecomposition.pdf).

Our goal is to measure the impact of L1 piece-wise flattening with edge-preserving on image compression for various image formats.

## Showcase

EPIS transforms an image (left) into a flattened image (right).

![EPIS showcase](./README_data/epis_showcase.png)

## Usage

```bash
python epis.py [filepath to image]
```

## Requirements

Libraries used and their respective versions. The minimum versions for each library are unknown. A detailed list can be found in [library_versions.txt](https://github.com/CS6384-S23-Group-Project/EPIS/blob/main/library_versions.txt).

* Python 3.10.0
* Numpy 1.24.2
* CuPy 12.0.0
* CuSPARSE 0.4.0
* CUDA 12.1

## Hardware

EPIS relies heavily on GPU computation, specifically sparse matrix multiplication and sparse matrix solving.
EPIS uses incredibly large sparse matrixes which necessitates a large pool of VRAM. This is the limiting factor
when flattening images. Downscaling images is required depending on the hardware used.

A GPU with 8GB can flatten images with dimensions of about 180x180 pixels.

The hardware used for `test_data` are:
* AMD Ryzen 5 5600x 6-Core 12-Thread
* 8GB Nvidia RTX 3070
* 32 GB DDR4-3600 RAM