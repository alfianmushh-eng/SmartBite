from __future__ import annotations
import numpy as np
import cv2
from sklearn.cluster import KMeans


class ColorHistogram:
    def extract(self, image: np.ndarray, bins: int = 32) -> np.ndarray:
        hist_features = []
        for i in range(3):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            hist_features.extend(hist)
        return np.array(hist_features, dtype=np.float32)


class ColorMoments:
    def extract(self, image: np.ndarray) -> np.ndarray:
        moments = []
        for i in range(3):
            channel = image[:, :, i].astype(np.float32)
            moments.append(float(np.mean(channel)))
            moments.append(float(np.std(channel)))
            moments.append(float(np.mean((channel - np.mean(channel)) ** 3)))
        return np.array(moments, dtype=np.float32)


class DominantColors:
    def __init__(self, n_colors: int = 5):
        self.n_colors = n_colors

    def extract(self, image: np.ndarray) -> np.ndarray:
        pixels = image.reshape(-1, 3).astype(np.float32)
        kmeans = KMeans(n_clusters=self.n_colors, n_init=1, random_state=42)
        kmeans.fit(pixels)
        counts = np.bincount(kmeans.labels_, minlength=self.n_colors)
        proportions = counts.astype(np.float32) / counts.sum()
        features = np.empty(self.n_colors * 4, dtype=np.float32)
        for i in range(self.n_colors):
            features[i*4:i*4+3] = kmeans.cluster_centers_[i] / 255.0
            features[i*4+3] = proportions[i]
        return features
