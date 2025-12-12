import numpy as np
import torch
import dataset
from tqdm import tqdm
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from model import build_pointing_network


@hydra.main(version_base=None, config_path="../conf", config_name="base")
def main(cfg: DictConfig) -> None:
    import logging

    logging.info(
        "Successfully loaded settings:\n"
        + "==================================================\n"
        + f"{OmegaConf.to_yaml(cfg)}"
        + "==================================================\n"
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cpu":
        logging.warning("Running DeePoint with CPU takes a long time.")

    # Basic argument checks
    assert cfg.movie is not None, "Please specify movie path: movie=./demo/example.mp4"
    assert cfg.lr is not None, "Please specify lr=l or lr=r"
    assert cfg.ckpt is not None, "Please specify ckpt=path/to/checkpoint.ckpt"

    # Make things as simple / light as possible (without changing model behavior)
    cfg.hardware.bs = 1       # 1 clip per batch
    cfg.hardware.nworkers = 0 # no multiprocessing dataloader workers

    tlength = cfg.model.tlength
    print(f"Using temporal length tlength={tlength}")

    # Dataset: same MovieDataset used by the original demo
    ds = dataset.MovieDataset(cfg.movie, cfg.lr, tlength, DEVICE)
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
    )

    # Build model and load checkpoint
    network = build_pointing_network(cfg, DEVICE)
    ckpt = torch.load(cfg.ckpt, map_location=DEVICE)

    # Strip off 'model.' prefix from Lightning checkpoints
    cleaned_state_dict = {}
    for key, val in ckpt["state_dict"].items():
        if key.startswith("model."):
            cleaned_state_dict[key[len("model."):]] = val
        else:
            cleaned_state_dict[key] = val

    network.load_state_dict(cleaned_state_dict, strict=False)
    network.to(DEVICE)
    network.eval()

    print("\n=== Running DeePoint (probabilities only, no video output) ===\n")

    with torch.no_grad():
        for batch in tqdm(dl):
            # Forward pass
            result = network(batch)

            # action logits: shape (2,) for [not-pointing, pointing]
            logits = result["action"][0]
            probs = torch.softmax(logits, dim=0)
            prob_pointing = float(probs[1])

            # 3D pointing direction
            direction = result["direction"][0].cpu().numpy()

            # approximate frame index of last frame in the clip
            idx = int(batch["idx"][0].item())
            frame_index = idx + (tlength - 1)

            print(
                f"[frame {frame_index:4d}] "
                f"prob_pointing={prob_pointing:.4f} "
                f"direction=[{direction[0]:+.3f}, {direction[1]:+.3f}, {direction[2]:+.3f}]"
            )

    print("\n=== Done. All probabilities printed above. ===\n")


if __name__ == "__main__":
    main()
