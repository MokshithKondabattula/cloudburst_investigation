# ============================================================
# FILE: inference.py
# ============================================================

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
import torch
from torch.amp import autocast
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from model import PrecipNowcastModel


# ====================================================================== #
# PREDICTOR
# ====================================================================== #

class Predictor:
    """
    Standalone inference class for the precipitation nowcasting model.

    Usage
    -----
    predictor = Predictor("checkpoints/best.pt")
    pred_array = predictor.predict(list_of_12_wv_paths)   # → (320, 320) numpy
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[Config] = None,
    ):
        if config is None:
            config = Config()
        self.config = config

        # ---- Device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Predictor] Device: {self.device}")

        # ---- Model ----
        self.model = PrecipNowcastModel(config)

        # ---- Load checkpoint ----
        print(f"[Predictor] Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        # Checkpoints may have been saved with or without DataParallel
        state_dict = ckpt.get("model_state", ckpt)
        # Strip 'module.' prefix if saved under DataParallel
        cleaned_state: Dict[str, torch.Tensor] = {}
        for k, v in state_dict.items():
            new_key = k[len("module."):] if k.startswith("module.") else k
            cleaned_state[new_key] = v

        self.model.load_state_dict(cleaned_state)
        self.model.to(self.device)
        self.model.eval()
        print("[Predictor] Model loaded and set to eval mode.")

    # ------------------------------------------------------------------ #
    # SINGLE PREDICTION
    # ------------------------------------------------------------------ #

    def predict(self, wv_image_paths: List) -> np.ndarray:
        """
        Produce a precipitation prediction from 12 water-vapour images.

        Parameters
        ----------
        wv_image_paths : list of str or Path, length == 12
            Ordered paths to grayscale WV PNG images (t-11 … t).

        Returns
        -------
        np.ndarray, shape (320, 320), dtype float32, values in [0, 1]
        """
        assert len(wv_image_paths) == 12, (
            f"Exactly 12 WV images required, got {len(wv_image_paths)}."
        )

        frames: List[torch.Tensor] = []
        for path in wv_image_paths:
            img = Image.open(str(path)).convert("L")
            arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
            tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, 320, 320)
            frames.append(tensor)

        # Stack → (12, 1, 320, 320) → add batch dim → (1, 12, 1, 320, 320)
        wv_tensor = torch.stack(frames, dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            with autocast("cuda", enabled=(self.device.type == "cuda")):
                output = self.model(wv_tensor)   # (1, 1, 320, 320)

        prediction = output[0, 0].cpu().numpy()   # (320, 320)
        return prediction

    # ------------------------------------------------------------------ #
    # PREDICT AND SAVE
    # ------------------------------------------------------------------ #

    def predict_and_save(
        self,
        wv_image_paths: List,
        output_dir: str,
        prefix: str = "pred",
    ) -> Dict[str, str]:
        """
        Run prediction and persist both a PNG visualisation and a .npy array.

        Parameters
        ----------
        wv_image_paths : list of str/Path  (length == 12)
        output_dir     : str — directory where outputs are written
        prefix         : str — filename prefix (no extension)

        Returns
        -------
        dict with keys "png" and "npy" mapping to the saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)

        pred = self.predict(wv_image_paths)   # (320, 320)

        # ---- PNG ----
        png_path = os.path.join(output_dir, f"{prefix}.png")
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(pred, cmap="hot", vmin=0.0, vmax=1.0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("Predicted Precipitation", fontsize=12, fontweight="bold")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # ---- NumPy array ----
        npy_path = os.path.join(output_dir, f"{prefix}.npy")
        np.save(npy_path, pred)

        print(f"[Predictor] Saved PNG → {png_path}")
        print(f"[Predictor] Saved NPY → {npy_path}")

        return {"png": png_path, "npy": npy_path}

    # ------------------------------------------------------------------ #
    # PREDICT ENTIRE EVENT
    # ------------------------------------------------------------------ #

    def predict_event(
        self,
        event_dir: str,
        output_dir: str,
    ) -> List[Dict[str, str]]:
        """
        Predict for all valid 12-image windows in an event directory.

        Parameters
        ----------
        event_dir  : str — root of the event (must contain wv_images/)
        output_dir : str — where to store predictions

        Returns
        -------
        List of dicts returned by predict_and_save.
        """
        wv_dir = os.path.join(event_dir, self.config.wv_subdir)
        if not os.path.isdir(wv_dir):
            raise FileNotFoundError(f"WV directory not found: {wv_dir}")

        # Gather and sort WV files by their 12-digit timestamp
        pattern = re.compile(r"(\d{12})\.png$")
        wv_files = []
        for fname in os.listdir(wv_dir):
            m = pattern.search(fname)
            if m:
                wv_files.append((m.group(1), os.path.join(wv_dir, fname)))

        wv_files.sort(key=lambda x: x[0])   # sort by timestamp string

        if len(wv_files) < 12:
            print(
                f"[Predictor] Only {len(wv_files)} WV images found in {wv_dir}; "
                f"need at least 12. No predictions made."
            )
            return []

        total_windows = len(wv_files) - 12 + 1
        print(
            f"[Predictor] Found {len(wv_files)} WV images → "
            f"{total_windows} prediction windows."
        )

        results: List[Dict[str, str]] = []
        for i in range(total_windows):
            window_paths = [path for (_, path) in wv_files[i: i + 12]]
            prefix = f"pred_{i:04d}"
            result = self.predict_and_save(window_paths, output_dir, prefix=prefix)
            results.append(result)
            print(f"  [{i + 1:04d}/{total_windows}] Predicted → {result['png']}")

        print(f"[Predictor] Done. {len(results)} predictions saved to {output_dir}")
        return results


# ====================================================================== #
# CLI ENTRY POINT
# ====================================================================== #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precipitation nowcasting inference"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained checkpoint .pt file (e.g. checkpoints/best.pt).",
    )
    parser.add_argument(
        "--wv_dir",
        required=True,
        help="Directory containing exactly 12 WV PNG images (sorted by name).",
    )
    parser.add_argument(
        "--output_dir",
        default="inference_output",
        help="Directory where prediction outputs will be saved (default: inference_output).",
    )
    args = parser.parse_args()

    # ---- Collect WV images ----
    wv_paths = sorted(Path(args.wv_dir).glob("*.png"))

    if len(wv_paths) < 12:
        print(f"Error: need at least 12 WV images, found {len(wv_paths)} in '{args.wv_dir}'")
        exit(1)

    # Use only the first 12 (chronologically earliest, assuming name-sorted order)
    wv_paths = wv_paths[:12]
    print(f"Using {len(wv_paths)} WV images from '{args.wv_dir}'")

    # ---- Run inference ----
    predictor = Predictor(args.checkpoint)
    result = predictor.predict_and_save(
        wv_image_paths=[str(p) for p in wv_paths],
        output_dir=args.output_dir,
        prefix="pred",
    )

    print("Inference complete:", result)