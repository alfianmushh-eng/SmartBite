"""Inference cache with SHA256 key and LRU eviction."""

import pickle
import hashlib
from pathlib import Path


class InferenceCache:
    def __init__(self, cache_dir="./cache", max_size_mb=500):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_mb * 1024 * 1024

    def _hash(self, data):
        return hashlib.sha256(data).hexdigest()[:16]

    def get(self, image_bytes):
        key = self._hash(image_bytes)
        path = self.cache_dir / f"{key}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        return None

    def put(self, image_bytes, result):
        key = self._hash(image_bytes)
        path = self.cache_dir / f"{key}.pkl"
        with open(path, "wb") as f:
            pickle.dump(result, f)
        self._evict_if_needed()

    def _evict_if_needed(self):
        total = sum(f.stat().st_size for f in self.cache_dir.iterdir() if f.is_file())
        if total > self.max_size_bytes:
            files = sorted(self.cache_dir.iterdir(), key=lambda f: f.stat().st_mtime)
            while total > self.max_size_bytes * 0.8 and files:
                f = files.pop(0)
                total -= f.stat().st_size
                f.unlink()

    def clear(self):
        for f in self.cache_dir.iterdir():
            if f.is_file():
                f.unlink()