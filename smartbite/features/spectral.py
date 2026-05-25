from __future__ import annotations
import numpy as np
import cv2


class SpectralSignature:
    def extract(self, image: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32))
        fshift = np.fft.fftshift(f)
        mag = np.abs(fshift)
        h, w = mag.shape
        ch, cw = h // 2, w // 2
        bands = []
        radii = [5, 15, 30, 60, 120]
        for r in radii:
            mask = np.zeros_like(mag)
            cv2.circle(mask, (cw, ch), r, 1, -1)
            bands.extend([float(mag[mask > 0].mean()), float(mag[mask > 0].std())])
        return np.array(bands, dtype=np.float32)


class NearInfraredEstimator:
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        ndvi_like = (rgb[:, :, 2].astype(float) - rgb[:, :, 0].astype(float)) /                     (rgb[:, :, 2] + rgb[:, :, 0] + 1e-6)
        return np.clip(ndvi_like, -1, 1)
