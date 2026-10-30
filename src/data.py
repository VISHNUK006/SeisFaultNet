import numpy as np
import torch
from torch.utils.data import Dataset

def normalize_section(section):
    """Robust per-section amplitude normalization using finite values and 1-99 percentiles."""
    x = np.asarray(section, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x)
    x = np.where(finite, x, np.nanmedian(x[finite]))
    lo, hi = np.percentile(x, [1, 99])
    x = np.clip(x, lo, hi)
    mean, std = float(x.mean()), float(x.std())
    return ((x - mean) / std if std > 1e-8 else x - mean).astype(np.float32)

class SeismicDataset(Dataset):
    def __init__(self, seismic, fault_mask, augment=False):
        if seismic.ndim != 3 or fault_mask.ndim != 3:
            raise ValueError("Expected arrays shaped (N,H,W).")
        if seismic.shape != fault_mask.shape:
            raise ValueError("seismic and fault_mask must have identical shapes.")
        self.seismic = seismic
        self.fault_mask = fault_mask
        self.augment = augment

    def __len__(self):
        return len(self.seismic)

    def __getitem__(self, i):
        x = normalize_section(self.seismic[i])
        y = (np.nan_to_num(self.fault_mask[i], nan=0) > 0).astype(np.float32)
        if self.augment:
            if np.random.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            if np.random.random() < 0.25:
                x, y = x[::-1, :], y[::-1, :]
        return torch.from_numpy(x.copy()[None]), torch.from_numpy(y.copy()[None])
