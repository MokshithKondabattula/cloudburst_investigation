# ============================================================
# FILE: inference.py
# ============================================================

import os
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.amp import autocast

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from config import Config
from model import PrecipNowcastModel


# ============================================================
# PREDICTOR
# ============================================================

class Predictor:

    def __init__(self,checkpoint_path):

        self.config=Config()

        self.device=torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"\nUsing device: {self.device}")

        self.model=PrecipNowcastModel(
            self.config
        )

        ckpt=torch.load(
            checkpoint_path,
            map_location=self.device
        )

        if "model_state" in ckpt:

            self.model.load_state_dict(
                ckpt["model_state"]
            )

        else:

            self.model.load_state_dict(
                ckpt
            )

        self.model.to(self.device)

        self.model.eval()

        print("Checkpoint loaded")


    # ========================================================
    # LOAD WV SEQUENCE
    # ========================================================

    def load_wv_sequence(self,wv_paths):

        frames=[]

        for fp in wv_paths:

            img=Image.open(fp).convert("L")

            arr=np.array(
                img,
                dtype=np.float32
            ) / 255.0

            tensor=torch.from_numpy(
                arr
            ).unsqueeze(0)

            frames.append(tensor)

        x=torch.stack(
            frames,
            dim=0
        )

        x=x.unsqueeze(0)

        return x.to(self.device)


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self,wv_paths):

        x=self.load_wv_sequence(
            wv_paths
        )

        with torch.no_grad():

            with autocast(
                "cuda",
                enabled=(
                    self.device.type=="cuda"
                )
            ):

                out=self.model(x)

        pred=out[0,0].cpu().numpy()

        pred=np.clip(
            pred,
            0,
            1
        )

        return pred


    # ========================================================
    # SAVE PANEL
    # ========================================================

    def save_panel(
        self,
        last_wv,
        gt,
        pred,
        save_path
    ):

        # ====================================================
        # MATCH PRED SCALE TO GT
        # ====================================================

        gt_min=gt.min()

        gt_max=gt.max()

        pred_vis=np.clip(
            pred,
            gt_min,
            gt_max
        )

        if gt_max-gt_min>1e-8:

            pred_vis=(
                pred_vis-gt_min
            ) / (
                gt_max-gt_min
            )

        # ====================================================
        # ERROR
        # ====================================================

        err=np.abs(
            pred_vis-gt
        )

        # ====================================================
        # FIGURE
        # ====================================================

        fig,axs=plt.subplots(
            1,
            4,
            figsize=(20,5)
        )

        fig.suptitle(
            "WV → Future Precipitation Prediction",
            fontsize=22,
            fontweight="bold"
        )

        # ====================================================
        # WV
        # ====================================================

        im0=axs[0].imshow(
            last_wv,
            cmap="gray"
        )

        axs[0].set_title(
            "Last WV Input",
            fontsize=16,
            fontweight="bold"
        )

        axs[0].axis("off")

        plt.colorbar(
            im0,
            ax=axs[0],
            fraction=0.046,
            pad=0.04
        )

        # ====================================================
        # GT
        # ====================================================

        im1=axs[1].imshow(
            gt,
            cmap="gray_r"
        )

        axs[1].set_title(
            "Ground Truth",
            fontsize=16,
            fontweight="bold"
        )

        axs[1].axis("off")

        plt.colorbar(
            im1,
            ax=axs[1],
            fraction=0.046,
            pad=0.04
        )

        # ====================================================
        # PREDICTION
        # ====================================================

        im2=axs[2].imshow(
            pred_vis,
            cmap="gray_r"
        )

        axs[2].set_title(
            "Predicted",
            fontsize=16,
            fontweight="bold"
        )

        axs[2].axis("off")

        plt.colorbar(
            im2,
            ax=axs[2],
            fraction=0.046,
            pad=0.04
        )

        # ====================================================
        # ERROR
        # ====================================================

        im3=axs[3].imshow(
            err,
            cmap="coolwarm"
        )

        axs[3].set_title(
            "Abs Error",
            fontsize=16,
            fontweight="bold"
        )

        axs[3].axis("off")

        plt.colorbar(
            im3,
            ax=axs[3],
            fraction=0.046,
            pad=0.04
        )

        plt.tight_layout()

        plt.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print(f"Saved → {save_path}")


# ============================================================
# MAIN
# ============================================================

if __name__=="__main__":

    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        required=True
    )

    parser.add_argument(
        "--wv_dir",
        required=True
    )

    parser.add_argument(
        "--gt_dir",
        required=True
    )

    parser.add_argument(
        "--output_dir",
        default="predictions"
    )

    args=parser.parse_args()

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # ========================================================
    # FILES
    # ========================================================

    wv_files=sorted(
        Path(args.wv_dir).glob("*.png")
    )

    gt_files=sorted(
        Path(args.gt_dir).glob("*.png")
    )

    print(f"WV images : {len(wv_files)}")

    print(f"GT images : {len(gt_files)}")

    predictor=Predictor(
        args.checkpoint
    )

    # ========================================================
    # TOTAL SEQUENCES
    # ========================================================

    total=min(
        len(wv_files)-12,
        len(gt_files)-12
    )

    print(f"Generating {total} predictions")

    # ========================================================
    # LOOP
    # ========================================================

    for i in range(total):

        try:

            # ================================================
            # INPUT WV SEQUENCE
            # ================================================

            seq=wv_files[
                i:i+12
            ]

            # ================================================
            # FUTURE GT
            # ================================================

            gt_path=gt_files[
                i+12
            ]

            # ================================================
            # LOAD LAST WV
            # ================================================

            last_wv=Image.open(
                seq[-1]
            ).convert("L")

            last_wv=np.array(
                last_wv,
                dtype=np.float32
            ) / 255.0

            # ================================================
            # LOAD GT
            # ================================================

            gt=Image.open(
                gt_path
            ).convert("L")

            gt=np.array(
                gt,
                dtype=np.float32
            ) / 255.0

            # ================================================
            # PREDICT
            # ================================================

            pred=predictor.predict(
                seq
            )

            # ================================================
            # SAVE PANEL
            # ================================================

            out_name=f"{i:04d}.png"

            out_path=os.path.join(
                args.output_dir,
                out_name
            )

            predictor.save_panel(
                last_wv,
                gt,
                pred,
                out_path
            )

        except Exception as e:

            print(f"\nERROR at index {i}")

            print(e)

    print("\nDONE")