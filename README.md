This repository is a structured reproduction and experimental extension of the DeePoint model
(Visual Pointing Recognition and Direction Estimation, ICCV 2023).

The project is organized into three branches:
- main: reproduction of the original demo using the authors’ pretrained weights
- baseline-small: training a small baseline model on a limited subset of the DP Dataset
- augmented-small: training a small model with modified data augmentation using the same subset

The purpose of this organization is to clearly separate:
1) exact reproduction of the original work,
2) a controlled small-data baseline experiment,
3) an augmentation-based improvement experiment.

────────────────────────
REPOSITORY STRUCTURE
────────────────────────

Top-level folders:

conf/
  Configuration files used by Hydra.
  - base.yaml controls global experiment settings.
  - data/DP.yaml controls dataset paths and venue selection.

src/
  Core code.
  - main.py: training and evaluation entry point (PyTorch Lightning)
  - demo.py: demo pipeline for running a trained model on a video
  - dataset.py: dataset loading, preprocessing, and token generation
  - model/: transformer-based pointing network
  - pl_module.py: Lightning module wrapper

demo/
  Demo inputs and outputs.
  - example.mp4: input demo video
  - processed demo videos are written here when running demo.py

results/
  Used in experimental branches to store outputs and checkpoints
  (not populated in main by default).

lightning_logs/
  Automatically created during training.
  Contains TensorBoard logs and trained checkpoints.
  Not committed by default.

data/
  Not committed.
  Expected local structure when training:
  - frames/: raw video frames from DP Dataset
  - labels/: CSV + metadata from DP Dataset
  - keypoints/: preprocessed pose/keypoint data

────────────────────────
MAIN BRANCH PURPOSE
────────────────────────

The main branch reproduces the authors’ demo pipeline using the original pretrained weights.
No model retraining is performed here.

This branch answers:
- Can the released DeePoint model be executed on a CPU-only Windows machine?
- Can the demo video pipeline be reproduced as described in the paper?

────────────────────────
HOW THE MODEL WORKS (OVERVIEW)
────────────────────────

For each short temporal window of frames:
1) A human pose estimator (OpenPifPaf) extracts 2D keypoints.
2) Keypoints are normalized relative to body size and chest position.
3) Joint positions are converted into token sequences.
4) A transformer-based network predicts:
   - pointing vs non-pointing probability
   - a 3D pointing direction vector
5) In demo mode, the direction is visualized as an arrow overlaid on the video.

The demo script also prints the pointing probability for each frame window.

────────────────────────
WHAT WAS CHANGED FOR REPRODUCTION
────────────────────────

Only compatibility and execution changes were made:

1) Dependency installation and version alignment
   Some required packages were not listed explicitly and were installed as needed
   (einops, matplotlib, openpifpaf).

2) CPU-only execution
   The pretrained checkpoint was saved from a CUDA environment.
   On a CPU-only system, checkpoints must be loaded with CPU mapping to avoid runtime errors.

3) No architectural or algorithmic changes
   The model structure, weights, and inference logic remain unchanged.

────────────────────────
HOW TO RUN THE DEMO (MAIN BRANCH)
────────────────────────

Activate environment:
conda activate deepoint
cd path/to/deepoint

Run demo:
python src/demo.py movie=./demo/example.mp4 lr=l ckpt=path/to/pretrained.ckpt

Outputs:
- processed demo videos saved in demo/
- per-frame pointing probabilities printed to terminal

────────────────────────
EXPERIMENTAL BRANCHES
────────────────────────

For training experiments, see:
- baseline-small branch README
- augmented-small branch README

These branches build on the same codebase but modify configuration and training behavior.
