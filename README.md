This branch trains a small DeePoint model with modified data augmentation.
The purpose is to measure whether augmentation improves performance relative to the baseline-small branch.

────────────────────────
EXPERIMENT RULES
────────────────────────

To ensure a fair comparison:
- the same dataset subset is used
- the same training duration is used
- the same demo video is used
- only augmentation behavior is changed

────────────────────────
AUGMENTATION STRATEGY
────────────────────────

Augmentation is applied in joint space rather than image space.
This avoids expensive image transformations and is suitable for CPU-only training.

Examples of augmentation applied:
- small Gaussian noise added to joint coordinates
- applied only during training
- disabled during validation and demo inference

This simulates natural pose estimation noise and encourages robustness.

────────────────────────
TRAINING
────────────────────────

Training is run using the same command structure as baseline-small,
with augmentation enabled via configuration or dataset logic.

The resulting checkpoint is stored separately from baseline weights.

────────────────────────
EVALUATION
────────────────────────

Evaluation uses:
- the same demo video
- the same demo script
- the same probability extraction process

Outputs:
- augmented demo video
- per-frame pointing probabilities
- probability distribution plots

These results are compared directly against the baseline-small outputs.

────────────────────────
INTERPRETATION
────────────────────────

Any change in:
- probability distribution
- pointing stability
- direction consistency

can be attributed to augmentation alone, since all other variables are held constant.
