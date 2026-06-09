# ============================================================
# FILE: config.py
# ============================================================

import os
import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # ------------------------------------------------------------------ #
    # DATA
    # ------------------------------------------------------------------ #
    data_root: str = "E:/data/"

    events_train: List[str] = field(default_factory=lambda: [
        "1st_event", "2nd_event", "3rd_event", "4th_event",
        "5th_event", "6th_event", "7th_event", "8th_event",
        "9th_event", "10th_event", "11th_event", "12th_event",
        "13th_event", "14th_event",
    ])

    events_val: List[str] = field(default_factory=lambda: [
        "15th_event", "16th_event", "17th_event",
    ])

    events_test: List[str] = field(default_factory=lambda: [
        "18th_event", "19th_event", "20th_event",
    ])

    wv_subdir: str = "wv_images"
    precip_subdir: str = "precipitation_images"
    image_size: Tuple[int, int] = (320, 320)
    seq_len: int = 12
    stride: int = 1

    # ------------------------------------------------------------------ #
    # TRAINING
    # ------------------------------------------------------------------ #
    batch_size: int = 2
    num_workers: int = 4
    max_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    mixed_precision: bool = False

    # ------------------------------------------------------------------ #
    # MODEL
    # ------------------------------------------------------------------ #
    encoder_channels: List[int] = field(default_factory=lambda: [1, 32, 64, 128, 256])
    convlstm_hidden: List[int] = field(default_factory=lambda: [256, 128])
    convlstm_kernel: int = 3

    # ------------------------------------------------------------------ #
    # PATHS
    # ------------------------------------------------------------------ #
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    prediction_dir: str = "predictions"

    # ------------------------------------------------------------------ #
    # POST-INIT: create directories
    # ------------------------------------------------------------------ #
    def __post_init__(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.prediction_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # SERIALISATION
    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        """Return the config as a plain dictionary."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Construct a Config from a plain dictionary."""
        return cls(**d)

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #
    def get_event_paths(self, split: str) -> List[str]:
        """
        Return full paths for a given split.

        Parameters
        ----------
        split : str
            One of "train", "val", or "test".

        Returns
        -------
        List[str]
            Absolute paths to the event directories.
        """
        split = split.lower()
        if split == "train":
            event_names = self.events_train
        elif split == "val":
            event_names = self.events_val
        elif split == "test":
            event_names = self.events_test
        else:
            raise ValueError(f"Unknown split '{split}'. Choose 'train', 'val', or 'test'.")

        return [os.path.join(self.data_root, name) for name in event_names]
