This branch trains a small baseline DeePoint model using a limited subset of the DP Dataset.
The goal is to produce a CPU-feasible baseline that can be directly compared to an augmented model.

────────────────────────
WHY A SMALL BASELINE
────────────────────────

The full DP Dataset contains millions of frames and is not feasible to train on a CPU-only laptop.
For a controlled academic experiment, a small subset is sufficient if:

- the same data subset is used consistently
- training time and settings are fixed
- comparisons are made relative to that baseline

────────────────────────
DATASET SETUP
────────────────────────

Only a single venue is used (example):
2023-01-17-livingroom

The following folders are required locally:
data/frames/2023-01-17-livingroom/
data/labels/2023-01-17-livingroom/
data/keypoints/

These folders are not committed to GitHub.

────────────────────────
TRAINING SETTINGS
────────────────────────

Key constraints:
- CPU-only training
- very small batch size
- very small shrink_rate
- short training duration (e.g. 1 epoch)

Example command:
python src/main.py task=train hardware.gpus=1 hardware.bs=2 hardware.nworkers=0 shrink_rate=0.001

This produces:
- TensorBoard logs in lightning_logs/
- a trained checkpoint saved automatically by PyTorch Lightning

────────────────────────
BASELINE OUTPUTS
────────────────────────

Using the trained checkpoint:
- demo.py is run on the same demo video
- a processed video is generated
- per-frame pointing probabilities are recorded
- probability plots are generated from the terminal output

These results define the baseline for comparison.
