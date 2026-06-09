# ============================================================
# FILE: train.py
# ============================================================

import os
import csv
import pathlib
from typing import Tuple, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from config import Config
from dataset import create_dataloaders
from model import PrecipNowcastModel
from losses import CombinedLoss
from metrics import compute_all_metrics


# ====================================================================== #
# TRAINER
# ====================================================================== #

class Trainer:
    """
    Manages the full training lifecycle:
    - builds data loaders, model, optimiser, scheduler, and loss
    - trains per-epoch with mixed-precision and gradient clipping
    - validates with quantitative metrics and visualisation panels
    - saves checkpoints (latest + best) and a CSV training log
    - plots training curves at the end of training
    """

    # ------------------------------------------------------------------ #
    # INITIALISATION
    # ------------------------------------------------------------------ #

    def __init__(self, config: Config):
        self.config = config

        # ---- Device ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # ---- Data ----
        print("\nBuilding data loaders …")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)

        # ---- Model ----
        print("\nBuilding model …")
        self.model: nn.Module = PrecipNowcastModel(config).to(self.device)
        if torch.cuda.device_count() > 1:
            print(f"  Using DataParallel across {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)
        self.model.count_parameters() if not isinstance(self.model, nn.DataParallel) \
            else self.model.module.count_parameters()

        # ---- Optimiser & scheduler ----
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=1e-6,
        )

        # ---- Loss ----
        self.criterion = CombinedLoss(mae_weight=0.5, ssim_weight=0.5)

        # ---- Mixed-precision scaler ----
        self.scaler = GradScaler("cuda", enabled=config.mixed_precision)

        # ---- State tracking ----
        self.best_val_loss: float = float("inf")
        self.no_improve: int = 0
        self.start_epoch: int = 0

        # ---- CSV log ----
        self.log_path = os.path.join(config.log_dir, "train_log.csv")
        self._init_csv_log()

        # ---- Attempt to resume from latest checkpoint ----
        latest_ckpt = os.path.join(config.checkpoint_dir, "latest.pt")
        if os.path.isfile(latest_ckpt):
            print(f"\nFound checkpoint at '{latest_ckpt}'. Resuming …")
            self.load_checkpoint(latest_ckpt)
        else:
            print("\nNo checkpoint found — training from scratch.")

    # ------------------------------------------------------------------ #
    # CSV LOG
    # ------------------------------------------------------------------ #

    def _init_csv_log(self):
        """Create the CSV file and write the header if it does not yet exist."""
        if not os.path.isfile(self.log_path):
            header = [
                "epoch", "train_loss", "val_loss",
                "MAE", "RMSE", "PSNR", "Pearson", "SSIM",
                "CSI_0.2", "POD_0.2", "FAR_0.2",
                "CSI_0.4", "POD_0.4", "FAR_0.4",
                "CSI_0.6", "POD_0.6", "FAR_0.6",
                "lr",
            ]
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def _write_csv_row(self, epoch: int, train_loss: float, val_loss: float,
                       metrics: Dict[str, float]):
        """Append one row to the CSV log."""
        current_lr = self.optimizer.param_groups[0]["lr"]
        row = [
            epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
            f"{metrics.get('MAE', 0):.6f}",
            f"{metrics.get('RMSE', 0):.6f}",
            f"{metrics.get('PSNR', 0):.6f}",
            f"{metrics.get('Pearson', 0):.6f}",
            f"{metrics.get('SSIM', 0):.6f}",
            f"{metrics.get('CSI_0.2', 0):.6f}",
            f"{metrics.get('POD_0.2', 0):.6f}",
            f"{metrics.get('FAR_0.2', 0):.6f}",
            f"{metrics.get('CSI_0.4', 0):.6f}",
            f"{metrics.get('POD_0.4', 0):.6f}",
            f"{metrics.get('FAR_0.4', 0):.6f}",
            f"{metrics.get('CSI_0.6', 0):.6f}",
            f"{metrics.get('POD_0.6', 0):.6f}",
            f"{metrics.get('FAR_0.6', 0):.6f}",
            f"{current_lr:.2e}",
        ]
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

    # ------------------------------------------------------------------ #
    # TRAINING
    # ------------------------------------------------------------------ #

    def train_one_epoch(self, epoch: int) -> float:
        """Run one training epoch and return the mean loss."""
        self.model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            inputs  = batch["input"].to(self.device, non_blocking=True)   # (B,12,1,320,320)
            targets = batch["target"].to(self.device, non_blocking=True)  # (B,1,320,320)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.config.mixed_precision):
                pred = self.model(inputs)                          # (B,1,320,320)
                loss, components = self.criterion(pred, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += components["total"]
            num_batches  += 1

            # Guard: if loss went nan this batch, stop accumulating silently
            if not torch.isfinite(loss):
                print(f"\n  [WARN] Non-finite loss detected in batch {num_batches}. Skipping grad update.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            pbar.set_postfix({
                "loss":  f"{components['total']:.4f}",
                "mae":   f"{components['mae']:.4f}",
                "ssim_l": f"{components['ssim_loss']:.4f}",
            })

        return running_loss / max(num_batches, 1)

    # ------------------------------------------------------------------ #
    # VALIDATION
    # ------------------------------------------------------------------ #

    def validate(self, epoch: int) -> Tuple[float, Dict[str, float]]:
        """Run validation and return (mean_val_loss, metrics_dict)."""
        self.model.eval()

        all_preds   = []
        all_targets = []
        val_losses  = []

        last_batch: Optional[Dict[str, Any]] = None
        last_pred_tensor: Optional[torch.Tensor] = None

        with torch.no_grad():
            pbar = tqdm(
                self.val_loader,
                desc=f"Epoch {epoch:03d} [val]  ",
                leave=False,
                dynamic_ncols=True,
            )
            for batch in pbar:
                inputs  = batch["input"].to(self.device, non_blocking=True)
                targets = batch["target"].to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.config.mixed_precision):
                    pred = self.model(inputs)
                    loss, components = self.criterion(pred, targets)

                val_losses.append(components["total"])

                all_preds.append(pred.cpu().detach().numpy())
                all_targets.append(targets.cpu().detach().numpy())

                last_batch       = batch
                last_pred_tensor = pred.cpu().detach()

        mean_val_loss = float(np.mean(val_losses))

        pred_array   = np.concatenate(all_preds,   axis=0)   # (N,1,320,320)
        target_array = np.concatenate(all_targets, axis=0)   # (N,1,320,320)

        metrics = compute_all_metrics(pred_array, target_array)

        # Save a visualisation panel every 5 epochs
        if epoch % 5 == 0 and last_batch is not None and last_pred_tensor is not None:
            self.save_prediction_panel(last_batch, last_pred_tensor, epoch)

        return mean_val_loss, metrics

    # ------------------------------------------------------------------ #
    # VISUALISATION
    # ------------------------------------------------------------------ #

    def save_prediction_panel(
        self,
        batch: Dict[str, Any],
        pred_tensor: torch.Tensor,
        epoch: int,
    ):
        """
        Save a 4-panel matplotlib figure for the first sample in the batch:
            Last WV Input | Ground Truth | Predicted | Absolute Error
        """
        last_wv = batch["input"][0, -1, 0].cpu().numpy()   # (320,320)
        gt      = batch["target"][0, 0].cpu().numpy()       # (320,320)
        pr      = pred_tensor[0, 0].numpy()                 # (320,320)
        err     = np.abs(pr - gt)                           # (320,320)

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        titles    = ["Last WV Input", "Ground Truth", "Predicted", "Abs Error"]
        data      = [last_wv, gt, pr, err]
        cmaps     = ["gray", "hot", "hot", "RdBu_r"]

        for ax, title, d, cmap in zip(axes, titles, data, cmaps):
            im = ax.imshow(d, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.suptitle(f"Epoch {epoch:03d}", fontsize=13, y=1.01)
        plt.tight_layout()

        save_path = os.path.join(self.config.prediction_dir, f"epoch_{epoch:03d}.png")
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved prediction panel → {save_path}")

    # ------------------------------------------------------------------ #
    # CHECKPOINTING
    # ------------------------------------------------------------------ #

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: Dict[str, float],
        is_best: bool,
    ):
        """Save the current training state to disk."""
        # Unwrap DataParallel if necessary
        model_state = (
            self.model.module.state_dict()
            if isinstance(self.model, nn.DataParallel)
            else self.model.state_dict()
        )

        ckpt = {
            "epoch":           epoch,
            "model_state":     model_state,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state":    self.scaler.state_dict(),
            "val_loss":        val_loss,
            "metrics":         metrics,
        }

        latest_path = os.path.join(self.config.checkpoint_dir, "latest.pt")
        torch.save(ckpt, latest_path)
        print(f"  Checkpoint saved → {latest_path}")

        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, "best.pt")
            torch.save(ckpt, best_path)
            print(f"  Best checkpoint updated → {best_path}")

    def load_checkpoint(self, path: str):
        """Load a saved checkpoint and restore all training states."""
        ckpt = torch.load(path, map_location=self.device)

        # Restore model weights (handle DataParallel)
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt["model_state"])
        else:
            self.model.load_state_dict(ckpt["model_state"])

        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.scaler.load_state_dict(ckpt["scaler_state"])

        self.start_epoch   = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["val_loss"]

        print(
            f"  Loaded epoch {ckpt['epoch']}, "
            f"best val loss = {self.best_val_loss:.6f}"
        )

    # ------------------------------------------------------------------ #
    # PLOTTING TRAINING CURVES
    # ------------------------------------------------------------------ #

    def plot_training_curves(self):
        """Read the CSV log and produce four summary plots."""
        # Try to use a nice style; fall back to default if unavailable
        try:
            plt.style.use("seaborn-v0_8-darkgrid")
        except OSError:
            plt.style.use("default")

        # ---- Read CSV ----
        rows = []
        with open(self.log_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if not rows:
            print("CSV log is empty — skipping curve plots.")
            return

        def col(key: str):
            return [float(r[key]) for r in rows]

        epochs = [int(r["epoch"]) for r in rows]

        # -- Plot 1: Loss curves --
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, col("train_loss"), label="Train Loss",  linewidth=2)
        ax.plot(epochs, col("val_loss"),   label="Val Loss",    linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, "loss_curve.png"), dpi=150)
        plt.close(fig)

        # -- Plot 2: MAE & RMSE --
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, col("MAE"),  label="MAE",  linewidth=2)
        ax.plot(epochs, col("RMSE"), label="RMSE", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error")
        ax.set_title("MAE & RMSE")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, "error_metrics.png"), dpi=150)
        plt.close(fig)

        # -- Plot 3: SSIM & Pearson --
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, col("SSIM"),    label="SSIM",    linewidth=2)
        ax.plot(epochs, col("Pearson"), label="Pearson", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("SSIM & Pearson Correlation")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, "correlation_metrics.png"), dpi=150)
        plt.close(fig)

        # -- Plot 4: CSI at three thresholds --
        fig, ax = plt.subplots(figsize=(8, 4))
        for thresh in ["0.2", "0.4", "0.6"]:
            ax.plot(epochs, col(f"CSI_{thresh}"), label=f"CSI@{thresh}", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("CSI")
        ax.set_title("Critical Success Index at Multiple Thresholds")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.log_dir, "csi_curves.png"), dpi=150)
        plt.close(fig)

        print("Training curve plots saved to", self.config.log_dir)

    # ------------------------------------------------------------------ #
    # MAIN TRAINING LOOP
    # ------------------------------------------------------------------ #

    def run(self):
        """Execute the full training loop with early stopping."""
        print(f"\n{'=' * 60}")
        print(f"  Starting training from epoch {self.start_epoch}")
        print(f"  Max epochs: {self.config.max_epochs}")
        print(f"  Early-stop patience: {self.config.early_stop_patience}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.start_epoch, self.config.max_epochs):
            # ---- Train ----
            train_loss = self.train_one_epoch(epoch)

            # ---- Validate ----
            val_loss, metrics = self.validate(epoch)

            # ---- LR scheduler step ----
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # ---- Early stopping bookkeeping ----
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.no_improve = 0
            else:
                self.no_improve += 1

            # ---- Save checkpoint ----
            self.save_checkpoint(epoch, val_loss, metrics, is_best)

            # ---- Write CSV row ----
            self._write_csv_row(epoch, train_loss, val_loss, metrics)

            # ---- Console summary ----
            best_marker = "  ★ BEST" if is_best else ""
            print(
                f"Epoch {epoch:03d}/{self.config.max_epochs - 1}  |  "
                f"Train: {train_loss:.5f}  Val: {val_loss:.5f}  |  "
                f"MAE: {metrics.get('MAE', 0):.4f}  "
                f"SSIM: {metrics.get('SSIM', 0):.4f}  "
                f"LR: {current_lr:.2e}"
                f"{best_marker}"
            )

            # ---- Early stopping ----
            if self.no_improve >= self.config.early_stop_patience:
                print(
                    f"\nEarly stopping triggered after {self.no_improve} "
                    f"epochs without improvement."
                )
                break

        # ---- Final plots ----
        self.plot_training_curves()
        print("\nTraining complete.")


# ====================================================================== #
# ENTRY POINT
# ====================================================================== #

if __name__ == "__main__":
    cfg = Config()
    trainer = Trainer(cfg)
    trainer.run()