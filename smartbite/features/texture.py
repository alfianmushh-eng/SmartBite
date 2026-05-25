from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


class GLCMFeatures:
    def __init__(self, distances: list[int] | None = None, angles: list[float] | None = None):
        self.distances = distances or [1, 3, 5]
        self.angles = angles or [0, np.pi/4, np.pi/2, 3*np.pi/4]

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        glcm = graycomatrix(gray, self.distances, self.angles, symmetric=True, normed=True)
        features = []
        for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
            vals = graycoprops(glcm, prop).flatten()
            features.extend([float(v) for v in vals])
        return np.array(features, dtype=np.float32)


class LBPFeatures:
    def __init__(self, radius: int = 3, n_points: int = 24):
        self.radius = radius
        self.n_points = n_points

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        lbp = local_binary_pattern(gray, self.n_points, self.radius, method="uniform")
        n_bins = self.n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
        return hist.astype(np.float32) / hist.sum()


class GaborFilterBank:
    def __init__(self, scales: int = 5, orientations: int = 8):
        self.kernels = []
        for scale in range(scales):
            for orientation in range(orientations):
                theta = orientation * np.pi / orientations
                ksize = 31 + scale * 10
                kernel = cv2.getGaborKernel((ksize, ksize), 4 + scale, theta, 10 + scale * 5, 0.5, 0)
                self.kernels.append(kernel)

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        features = []
        for kernel in self.kernels:
            filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
            features.extend([float(filtered.mean()), float(filtered.std())])
        return np.array(features, dtype=np.float32)
