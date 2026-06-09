# ============================================================
# FILE: dataset.py
# ============================================================

import os
import re
from typing import List, Dict, Any, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from config import Config


class PrecipNowcastDataset(Dataset):
    """
    Dataset for precipitation nowcasting.

    Given a list of event directories, builds sliding-window sequences of
    12 consecutive water-vapour images paired with the precipitation image
    at the next timestamp.

    Parameters
    ----------
    event_dirs : List[str]
        Full paths to event root directories (each contains wv_images/ and
        precipitation_images/ sub-directories).
    config : Config
        Project configuration.
    augment : bool
        If True, apply random horizontal / vertical flips at load time.
    """

    def __init__(self, event_dirs: List[str], config: Config, augment: bool = False):
        self.event_dirs = event_dirs
        self.config = config
        self.augment = augment
        self.sequences: List[Dict[str, Any]] = self._build_sequences()
        print(f"  → Total sequences: {len(self.sequences)}")

    # ------------------------------------------------------------------ #
    # SEQUENCE BUILDING
    # ------------------------------------------------------------------ #

    def _build_sequences(self) -> List[Dict[str, Any]]:
        """
        Scan every event directory and build the list of valid sample dicts.

        Each dict contains:
            wv_paths   : list of 12 full file paths (WV inputs, t-11 … t)
            target_path: full file path for the precipitation target (t+1)
            event      : event directory name
            timestamp  : timestamp string of the last WV image (t)
        """
        sequences: List[Dict[str, Any]] = []
        pattern = re.compile(r"(\d{12})\.png$")

        for event_dir in self.event_dirs:
            wv_dir = os.path.join(event_dir, self.config.wv_subdir)
            pr_dir = os.path.join(event_dir, self.config.precip_subdir)

            if not os.path.isdir(wv_dir):
                print(f"  [WARN] WV directory not found: {wv_dir}")
                continue
            if not os.path.isdir(pr_dir):
                print(f"  [WARN] Precip directory not found: {pr_dir}")
                continue

            # Map timestamp → file path for WV images
            wv_by_ts: Dict[str, str] = {}
            for fname in os.listdir(wv_dir):
                m = pattern.search(fname)
                if m:
                    wv_by_ts[m.group(1)] = os.path.join(wv_dir, fname)

            # Map timestamp → file path for precipitation images
            pr_by_ts: Dict[str, str] = {}
            for fname in os.listdir(pr_dir):
                m = pattern.search(fname)
                if m:
                    pr_by_ts[m.group(1)] = os.path.join(pr_dir, fname)

            # Timestamps present in BOTH directories
            common_ts = sorted(set(wv_by_ts.keys()) & set(pr_by_ts.keys()))

            if len(common_ts) < self.config.seq_len + 1:
                print(
                    f"  [WARN] {os.path.basename(event_dir)}: only {len(common_ts)} "
                    f"common timestamps (need ≥{self.config.seq_len + 1}), skipping."
                )
                continue

            # Sliding window: window size = seq_len + 1, stride = config.stride
            window_size = self.config.seq_len + 1  # 13
            event_count = 0

            for i in range(0, len(common_ts) - window_size + 1, self.config.stride):
                window_ts = common_ts[i: i + window_size]  # 13 timestamps

                wv_ts_list = window_ts[: self.config.seq_len]   # first 12
                target_ts = window_ts[self.config.seq_len]       # 13th

                # Verify all WV files exist
                wv_paths = [wv_by_ts[ts] for ts in wv_ts_list]
                all_exist = all(os.path.isfile(p) for p in wv_paths)

                # Verify target file exists
                target_path = pr_by_ts[target_ts]
                all_exist = all_exist and os.path.isfile(target_path)

                if not all_exist:
                    continue

                sequences.append({
                    "wv_paths": wv_paths,
                    "target_path": target_path,
                    "event": os.path.basename(event_dir),
                    "timestamp": wv_ts_list[-1],   # timestamp of the last WV image (t)
                })
                event_count += 1

            print(f"  {os.path.basename(event_dir)}: {event_count} sequences")

        return sequences

    # ------------------------------------------------------------------ #
    # DATASET INTERFACE
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.sequences[idx]

        # ---- Load 12 WV images ----
        wv_frames: List[np.ndarray] = []
        for path in sample["wv_paths"]:
            img = Image.open(path).convert("L")  # grayscale
            arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]
            arr = arr[np.newaxis, ...]             # (1, 320, 320)
            wv_frames.append(arr)

        wv_array = np.stack(wv_frames, axis=0)    # (12, 1, 320, 320)

        # ---- Load precipitation target ----
        tgt_img = Image.open(sample["target_path"]).convert("L")
        tgt_arr = np.array(tgt_img, dtype=np.float32) / 255.0  # [0, 1]
        tgt_arr = tgt_arr[np.newaxis, ...]         # (1, 320, 320)

        # ---- Augmentation (consistent across all 13 images) ----
        if self.augment:
            # Horizontal flip
            if np.random.rand() < 0.5:
                wv_array = wv_array[:, :, :, ::-1].copy()
                tgt_arr = tgt_arr[:, :, ::-1].copy()

            # Vertical flip
            if np.random.rand() < 0.5:
                wv_array = wv_array[:, :, ::-1, :].copy()
                tgt_arr = tgt_arr[:, ::-1, :].copy()

        # ---- Convert to tensors ----
        wv_tensor = torch.from_numpy(wv_array)    # (12, 1, 320, 320)
        tgt_tensor = torch.from_numpy(tgt_arr)    # (1, 320, 320)

        return {
            "input": wv_tensor,
            "target": tgt_tensor,
            "event": sample["event"],
            "timestamp": sample["timestamp"],
        }


# ------------------------------------------------------------------ #
# DATALOADER FACTORY
# ------------------------------------------------------------------ #

def create_dataloaders(
    config: Config,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from a Config.

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
        (train_loader, val_loader, test_loader)
    """
    print("Building TRAIN dataset …")
    train_ds = PrecipNowcastDataset(
        config.get_event_paths("train"), config, augment=True
    )

    print("Building VAL dataset …")
    val_ds = PrecipNowcastDataset(
        config.get_event_paths("val"), config, augment=False
    )

    print("Building TEST dataset …")
    test_ds = PrecipNowcastDataset(
        config.get_event_paths("test"), config, augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print(
        f"\nDataLoader sizes  —  "
        f"Train: {len(train_loader)} batches ({len(train_ds)} samples)  |  "
        f"Val: {len(val_loader)} batches ({len(val_ds)} samples)  |  "
        f"Test: {len(test_loader)} batches ({len(test_ds)} samples)"
    )

    return train_loader, val_loader, test_loader
