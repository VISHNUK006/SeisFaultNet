# SeisFaultNet
## Deep Learning-Based Seismic Fault Detection

> **Portfolio project for applied geoscience + machine learning**

SeisFaultNet is a reproducible research prototype that uses a compact **2D U-Net** to generate pixel-wise **fault probability maps** from seismic amplitude sections.

The project combines seismic interpretation concepts with machine-learning engineering: preprocessing, class-imbalance handling, augmentation, group-held-out validation, quantitative segmentation metrics, and an interactive interpretation interface.

### What the project demonstrates

- **Geoscience:** seismic amplitude, structural discontinuities, fault interpretation, annotation uncertainty.
- **Machine learning:** U-Net segmentation, BCE + Dice loss, class weighting, augmentation, early stopping.
- **Scientific validation:** survey/volume-level grouping to reduce spatial leakage.
- **Software engineering:** modular Python code, tests, CI, model card, data card and reproducible commands.
- **Deployment:** Streamlit application for interactive fault-probability visualization.
- **Real-data pathway:** optional SEG-Y loading through `segyio`.

## Architecture

```text
Seismic amplitude
      │
      ▼
Robust normalization
      │
      ▼
2D U-Net encoder/decoder
      │
      ▼
Fault logits
      │
      ▼
Sigmoid probability map
      │
      ├── probability overlay
      ├── thresholded candidates
      └── precision / recall / F1 / IoU
```

## Important scientific note

The bundled dataset is **synthetic** and is included only as a software smoke test. It does not establish real-world fault-detection performance.

A credible geological result requires a real labelled seismic dataset, documented label provenance, and testing on a genuinely independent seismic volume/survey.

## Quick start — Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

python make_demo_data.py
python -m pytest -q
python train.py --data data/sample/synthetic_demo.npz --epochs 10
streamlit run app.py
```

Then upload:
`data/sample/synthetic_demo.npz`

and the checkpoint:
`models/seisfaultnet.pt`

## SEG-Y support

SEG-Y loading is intentionally separated because it adds a specialized dependency.

```powershell
pip install -r requirements-segyio.txt
```

The Streamlit app can then accept `.sgy` / `.segy` files and display the loaded seismic volume section-by-section.

For production-grade seismic interpretation, geometry, inline/crossline orientation, sample interval, coordinate system and header semantics should be verified for each dataset rather than assumed.

## Real labelled data workflow

Prepare an NPZ archive:

```python
np.savez_compressed(
    "labelled_sections.npz",
    seismic=seismic_sections.astype("float32"),
    fault_mask=fault_masks.astype("uint8"),
    groups=survey_ids,
)
```

Required:
- `seismic`: `(N,H,W)`
- `fault_mask`: `(N,H,W)`
- `groups`: `(N,)`

All slices from one seismic volume should share the same group ID.

### Why group validation?

Randomly splitting neighbouring seismic slices can make a model look much better than it really is because nearly identical geological information can occur in both train and test sets.

SeisFaultNet therefore uses **group-held-out validation**. The strongest version is:

```text
Survey A + Survey B → training
Survey C             → validation/test
```

rather than:

```text
Random slices from A+B+C → train/test
```

## Metrics

The project reports:

- **Precision** — how many predicted fault pixels are supported by the reference mask.
- **Recall** — how much of the reference fault mask is detected.
- **F1-score** — balance between precision and recall.
- **IoU** — intersection-over-union of predicted and reference masks.
- **Specificity** — non-fault pixel rejection.

For sparse fault labels, accuracy alone is misleading.

## Project outputs

The Streamlit interface provides:

1. seismic amplitude view
2. reference annotation overlay, when available
3. fault probability heatmap
4. thresholded fault candidate contours
5. quantitative metrics when labels are available
6. downloadable probability maps

## Recommended real-data development path

### Phase 1 — completed baseline
Synthetic seismic → U-Net → probability map → Streamlit.

### Phase 2 — real seismic
Use a properly licensed/publicly accessible fault-labelled dataset.

### Phase 3 — geophysical input
Add SEG-Y ingestion and verify geometry/header information.

### Phase 4 — stronger validation
Hold out an entire seismic volume or survey.

### Phase 5 — research extension
Compare:
- U-Net
- classical edge/semblance-style baseline
- different thresholds
- different annotation sources

### Phase 6 — 3D extension
Only after the 2D workflow is validated, investigate 3D U-Net or 2.5D context.

## Dataset research leads

Potential resources to investigate include:

- **CNN for Fault Interpretation:** https://github.com/anyuzoey/CNNforFaultInterpretation
- **CRACKS seismic fault annotation resources:** https://alregib.ece.gatech.edu/software-and-datasets/cracks-crowdsourcing-resources-for-analysis-and-categorization-of-key-subsurface-faults/
- **Microsoft Seismic Deep Learning:** https://github.com/microsoft/seismic-deeplearning

Always verify the current licence, access terms, annotation type and redistribution rules before using or publishing data. A seismic benchmark with facies or horizon labels is **not automatically a fault-label dataset**.

## Repository quality checklist

- [x] modular source code
- [x] reproducible demo data
- [x] automated tests
- [x] GitHub Actions CI
- [x] `.gitignore`
- [x] model card
- [x] data card
- [x] citation metadata
- [x] group-held-out validation
- [x] interactive Streamlit app
- [x] optional SEG-Y pathway

## Limitations

This is a baseline, not a production interpreter. Fault labels can be subjective, faults may be discontinuous, and seismic acquisition/processing artefacts can resemble structural features. Domain shift between basins and surveys can be substantial.

The model should therefore be presented as **AI-assisted fault candidate detection**, not automatic geological interpretation.

## License

MIT for the project code. Third-party datasets remain subject to their own licences and terms.
