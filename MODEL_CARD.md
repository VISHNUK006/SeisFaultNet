# SeisFaultNet Model Card

## Intended use
SeisFaultNet is an educational/research baseline for detecting candidate fault pixels on 2D seismic amplitude sections. It is intended for experimentation, visualization and workflow development.

## Not intended for
- autonomous geological interpretation
- drilling or production decisions
- reservoir-risk decisions
- claims of universal performance across basins
- use without expert quality control

## Model
A compact 2D U-Net produces one logit per seismic pixel. A sigmoid converts logits into fault probability.

## Training objective
The enhanced baseline combines:
- weighted binary cross entropy for sparse positive labels
- soft Dice loss for overlap and class imbalance

## Validation
The training script uses `GroupShuffleSplit`. Sections belonging to the same group are kept on one side of the split. For a meaningful scientific experiment, a group should represent an independent seismic volume/survey rather than an arbitrary slice.

## Known limitations
- 2D context only
- annotation uncertainty
- acquisition and processing artefacts
- domain shift between surveys
- probability calibration has not been established
- synthetic demonstration data do not represent real geological complexity

## Reporting rule
Do not report synthetic-demo metrics as real-world performance. Report the dataset, label provenance, held-out survey IDs, preprocessing, threshold, metrics and uncertainty.
