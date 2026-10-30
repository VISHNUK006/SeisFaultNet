from pathlib import Path
import io
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.data import normalize_section
from src.model import UNet2D
from src.metrics import binary_metrics

st.set_page_config(
    page_title="SeisFaultNet | Seismic Fault Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
.small-note {color:#666; font-size:0.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("SeisFaultNet")
st.subheader("Deep-learning-assisted seismic fault detection")
st.write(
    "A research prototype for converting seismic amplitude sections into "
    "interpretable fault-probability maps using a compact U-Net."
)

with st.sidebar:
    st.header("Workflow")
    st.markdown(
        "1. Load seismic data\n"
        "2. Inspect amplitude\n"
        "3. Load a trained model\n"
        "4. Generate fault probability\n"
        "5. Compare against labels\n"
    )
    st.divider()
    st.caption("Model: 2D U-Net • Loss: BCE + Dice • Validation: group-held-out")
    st.caption("⚠️ Research prototype — predictions require geoscientist review.")

@st.cache_resource
def load_checkpoint(raw_bytes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(io.BytesIO(raw_bytes), map_location=device, weights_only=False)
    model = UNet2D()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, device, checkpoint

def load_npz(raw):
    z = np.load(io.BytesIO(raw), allow_pickle=False)
    if "seismic" not in z.files:
        raise ValueError("NPZ must contain a 'seismic' array.")
    seismic = z["seismic"]
    masks = z["fault_mask"] if "fault_mask" in z.files else None
    groups = z["groups"] if "groups" in z.files else None
    if seismic.ndim == 2:
        seismic = seismic[None]
    if seismic.ndim != 3:
        raise ValueError("Seismic data must be (N,H,W) or (H,W).")
    return seismic, masks, groups

def load_segy_bytes(raw):
    try:
        import segyio
    except ImportError as exc:
        raise RuntimeError("SEG-Y support requires `pip install -r requirements-segyio.txt`.") from exc
    with tempfile.NamedTemporaryFile(suffix=".sgy", delete=False) as f:
        f.write(raw)
        path = f.name
    try:
        with segyio.open(path, "r", ignore_geometry=True) as f:
            cube = segyio.tools.cube(f)
        # Common SEG-Y cube order is inline, crossline, sample. Display inline slices.
        return np.asarray(cube, dtype=np.float32), None, None
    finally:
        Path(path).unlink(missing_ok=True)

st.markdown("### 1. Data")
data_file = st.file_uploader(
    "Upload labelled NPZ, seismic NPZ, or SEG-Y",
    type=["npz", "sgy", "segy"],
    help="NPZ: seismic + optional fault_mask + groups. SEG-Y: amplitude volume; labels must be supplied separately.",
)
model_file = st.file_uploader("Trained SeisFaultNet checkpoint (.pt)", type=["pt"])

if not data_file:
    st.info("Start with `data/sample/synthetic_demo.npz` to verify the complete workflow.")
    st.stop()

try:
    if data_file.name.lower().endswith(".npz"):
        seismic, masks, groups = load_npz(data_file.getvalue())
    else:
        seismic = load_segy_bytes(data_file.getvalue())
        masks, groups = None, None
except Exception as exc:
    st.error(f"Could not load the dataset: {exc}")
    st.stop()

st.success(f"Loaded {seismic.shape[0]} section(s) with {seismic.shape[1]} × {seismic.shape[2]} samples.")

idx = st.slider("Section / inline index", 0, len(seismic)-1, 0)
section = np.asarray(seismic[idx])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sections", f"{len(seismic):,}")
c2.metric("Samples", f"{section.shape[0]} × {section.shape[1]}")
c3.metric("Amplitude mean", f"{np.nanmean(section):.3f}")
c4.metric("Amplitude σ", f"{np.nanstd(section):.3f}")

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(section, cmap="gray", aspect="auto")
ax.set_title("Seismic amplitude")
ax.set_xlabel("Trace / lateral sample")
ax.set_ylabel("Time / depth sample")
fig.colorbar(im, ax=ax, shrink=.8, label="Amplitude")
st.pyplot(fig)
plt.close(fig)

if masks is not None:
    if masks.ndim == 2:
        masks = masks[None]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.imshow(section, cmap="gray", aspect="auto")
    ax.imshow(masks[idx], cmap="autumn", alpha=.55, aspect="auto")
    ax.set_title("Reference fault annotation")
    st.pyplot(fig)
    plt.close(fig)

if not model_file:
    st.warning("Upload a trained checkpoint to generate a fault-probability map.")
    st.stop()

try:
    model, device, checkpoint = load_checkpoint(model_file.getvalue())
    x = normalize_section(section)
    with torch.no_grad():
        tensor = torch.from_numpy(x[None, None]).float().to(device)
        probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
except Exception as exc:
    st.error(f"Could not run inference: {exc}")
    st.stop()

st.markdown("### 2. Fault probability")
threshold = st.slider("Interpretation threshold", 0.05, 0.95, 0.50, 0.05)

left, right = st.columns(2)
with left:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(section, cmap="gray", aspect="auto")
    ax.imshow(probability, cmap="magma", alpha=.60, vmin=0, vmax=1, aspect="auto")
    ax.set_title("Seismic + fault probability")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time / depth")
    st.pyplot(fig)
    plt.close(fig)

with right:
    prediction = probability >= threshold
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(section, cmap="gray", aspect="auto")
    ax.contour(prediction, levels=[.5], colors="cyan", linewidths=.8)
    ax.set_title("Thresholded fault candidates")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time / depth")
    st.pyplot(fig)
    plt.close(fig)

m1, m2, m3 = st.columns(3)
m1.metric("Mean probability", f"{probability.mean():.3f}")
m2.metric("Max probability", f"{probability.max():.3f}")
m3.metric("Candidate fault pixels", f"{prediction.mean():.2%}")

if masks is not None:
    metrics = binary_metrics(masks[idx], probability, threshold)
    st.markdown("### 3. Label-based evaluation")
    a,b,c,d = st.columns(4)
    a.metric("Precision", f"{metrics['precision']:.3f}")
    b.metric("Recall", f"{metrics['recall']:.3f}")
    c.metric("F1", f"{metrics['f1']:.3f}")
    d.metric("IoU", f"{metrics['iou']:.3f}")

out = io.BytesIO()
np.save(out, probability.astype("float32"))
st.download_button(
    "Download fault-probability map (.npy)",
    data=out.getvalue(),
    file_name=f"fault_probability_section_{idx}.npy",
    mime="application/octet-stream",
)

st.divider()
st.caption(
    "Interpretation note: high probability indicates model confidence, not geological certainty. "
    "Review continuity, reflector character, acquisition artefacts and structural context before interpretation."
)
