import numpy as np
import torch
from src.data import normalize_section
from src.metrics import binary_metrics
from src.model import UNet2D

def test_normalization():
    y=normalize_section(np.array([[1.,np.nan],[2.,3.]]))
    assert y.shape==(2,2) and np.isfinite(y).all()

def test_metrics():
    y=np.array([[0,1],[1,0]])
    m=binary_metrics(y,y.astype(float))
    assert m["f1"]>.99 and m["iou"]>.99

def test_model_shape():
    assert UNet2D()(torch.randn(1,1,64,80)).shape==(1,1,64,80)
