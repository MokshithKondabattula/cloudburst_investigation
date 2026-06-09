# ============================================================
# FILE: model.py
# ============================================================

from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


# ====================================================================== #
# ConvLSTM CELL
# ====================================================================== #

class ConvLSTMCell(nn.Module):
    """
    A single ConvLSTM cell.

    Parameters
    ----------
    in_channels : int
        Number of input feature channels.
    hidden_channels : int
        Number of hidden-state channels.
    kernel_size : int
        Spatial kernel size for the convolutional gate (default 3).
    """

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        # Single convolution that computes all 4 gates at once
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x      : (B, C_in, H, W)
        h_prev : (B, C_hidden, H, W)
        c_prev : (B, C_hidden, H, W)

        Returns
        -------
        h, c : each (B, C_hidden, H, W)
        """
        combined = torch.cat([x, h_prev], dim=1)   # (B, C_in+C_hidden, H, W)
        gates = self.conv(combined)                 # (B, 4*C_hidden, H, W)

        # Split into four gate tensors along channel axis
        i_gate, f_gate, o_gate, g_gate = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i_gate)   # input gate
        f = torch.sigmoid(f_gate)   # forget gate
        o = torch.sigmoid(o_gate)   # output gate
        g = torch.tanh(g_gate)      # cell gate

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c

    def init_hidden(
        self,
        batch_size: int,
        h: int,
        w: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return zero-initialised (h, c) state tensors."""
        zeros_h = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        zeros_c = torch.zeros(batch_size, self.hidden_channels, h, w, device=device)
        return zeros_h, zeros_c


# ====================================================================== #
# ConvLSTM (multi-layer)
# ====================================================================== #

class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM.

    Parameters
    ----------
    in_channels : int
        Input channel count (fed to layer 0).
    hidden_channels_list : List[int]
        Hidden channel counts for each layer.
    kernel_size : int
        Spatial kernel size passed to every ConvLSTMCell.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels_list: List[int],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_layers = len(hidden_channels_list)
        self.hidden_channels_list = hidden_channels_list

        cells = []
        for layer_idx, hidden_ch in enumerate(hidden_channels_list):
            layer_in_ch = in_channels if layer_idx == 0 else hidden_channels_list[layer_idx - 1]
            cells.append(ConvLSTMCell(layer_in_ch, hidden_ch, kernel_size))

        self.cells = nn.ModuleList(cells)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Run the full sequence through all layers.

        Parameters
        ----------
        x_seq : (B, T, C, H, W)

        Returns
        -------
        h_last : (B, hidden_channels_list[-1], H, W)
            Hidden state of the last layer at the last timestep.
        """
        B, T, C, H, W = x_seq.shape
        device = x_seq.device

        # Initialise hidden and cell states for every layer
        states: List[Tuple[torch.Tensor, torch.Tensor]] = [
            cell.init_hidden(B, H, W, device) for cell in self.cells
        ]

        # Iterate over time
        for t in range(T):
            layer_input = x_seq[:, t]   # (B, C, H, W)

            new_states: List[Tuple[torch.Tensor, torch.Tensor]] = []
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = states[layer_idx]
                h, c = cell(layer_input, h_prev, c_prev)
                new_states.append((h, c))
                layer_input = h  # feed this layer's output to the next

            states = new_states

        # Return the hidden state of the last layer at the last timestep
        h_last, _ = states[-1]
        return h_last   # (B, hidden_channels_list[-1], H, W)


# ====================================================================== #
# ENCODER BLOCK
# ====================================================================== #

class DoubleConv(nn.Sequential):
    """Conv → BN → ReLU → Conv → BN → ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    """
    One encoder stage: double conv (produces skip) + max-pool (produces next input).

    Parameters
    ----------
    in_ch  : int  — input channels
    out_ch : int  — output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.double_conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        skip   : (B, out_ch, H, W)     — before pooling (used by decoder)
        pooled : (B, out_ch, H/2, W/2) — after max-pool
        """
        skip = self.double_conv(x)
        pooled = self.pool(skip)
        return skip, pooled


# ====================================================================== #
# DECODER BLOCK
# ====================================================================== #

class DecoderBlock(nn.Module):
    """
    One decoder stage: transposed-conv upsample → cat with skip → double conv.

    Parameters
    ----------
    in_ch   : int — channels of the incoming (low-resolution) feature map
    skip_ch : int — channels of the skip connection from the encoder
    out_ch  : int — output channels
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, in_ch, H, W)
        skip : (B, skip_ch, 2H, 2W)

        Returns
        -------
        out  : (B, out_ch, 2H, 2W)
        """
        x = self.up(x)                        # (B, out_ch, 2H, 2W)

        # Handle potential size mismatch due to odd input dimensions
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)       # (B, out_ch+skip_ch, 2H, 2W)
        return self.conv(x)                   # (B, out_ch, 2H, 2W)


# ====================================================================== #
# FULL MODEL
# ====================================================================== #

class PrecipNowcastModel(nn.Module):
    """
    Precipitation nowcasting model:
        CNN Encoder (per-frame, shared weights)
        → ConvLSTM (temporal modelling at 20×20)
        → U-Net Decoder (with skip connections from last frame encoder)
        → Conv1×1 + Sigmoid → (B, 1, 320, 320)

    Input  : (B, 12, 1, 320, 320)
    Output : (B,  1, 320, 320)
    """

    def __init__(self, config: Config):
        super().__init__()

        enc_ch = config.encoder_channels   # [1, 32, 64, 128, 256]
        lstm_hidden = config.convlstm_hidden   # [256, 128]
        lstm_kernel = config.convlstm_kernel   # 3

        # ---- Encoder (shared across timesteps) ----
        self.enc1 = EncoderBlock(enc_ch[0], enc_ch[1])   # 1   → 32,  320→160
        self.enc2 = EncoderBlock(enc_ch[1], enc_ch[2])   # 32  → 64,  160→80
        self.enc3 = EncoderBlock(enc_ch[2], enc_ch[3])   # 64  → 128, 80→40
        self.enc4 = EncoderBlock(enc_ch[3], enc_ch[4])   # 128 → 256, 40→20

        # ---- Project to ConvLSTM input channels ----
        self.project_conv = nn.Conv2d(enc_ch[4], enc_ch[4], kernel_size=1)

        # ---- ConvLSTM (2 layers) at 20×20 ----
        self.convlstm = ConvLSTM(
            in_channels=enc_ch[4],           # 256
            hidden_channels_list=lstm_hidden, # [256, 128]
            kernel_size=lstm_kernel,
        )

        # ---- U-Net Decoder ----
        # Input spatial sizes and channel counts (bottom-up):
        #   h_last: (B, 128, 20, 20)  skip4: (B, 256, 20, 20)
        #   after dec1: (B, 128, 40, 40)  skip3: (B, 128, 40, 40)
        #   after dec2: (B,  64, 80, 80)  skip2: (B,  64, 80, 80)
        #   after dec3: (B,  32,160,160)  skip1: (B,  32,160,160)
        #   after dec4: (B,  32,320,320)
        self.dec1 = DecoderBlock(lstm_hidden[-1], enc_ch[4], 128)  # 128,256 → 128
        self.dec2 = DecoderBlock(128,             enc_ch[3], 64)   # 128,128 → 64
        self.dec3 = DecoderBlock(64,              enc_ch[2], 32)   # 64, 64  → 32
        self.dec4 = DecoderBlock(32,              enc_ch[1], 32)   # 32, 32  → 32

        # ---- Output head ----
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    # ------------------------------------------------------------------ #

    def _encode_frame(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a single frame through the 4 encoder blocks.

        Parameters
        ----------
        frame : (B, 1, 320, 320)

        Returns
        -------
        skip1, skip2, skip3, skip4, encoded
            skip1 : (B,  32, 160, 160)
            skip2 : (B,  64,  80,  80)
            skip3 : (B, 128,  40,  40)
            skip4 : (B, 256,  20,  20)
            encoded: (B, 256,  20,  20)  — projected
        """
        skip1, x = self.enc1(frame)   # 320→160
        skip2, x = self.enc2(x)       # 160→80
        skip3, x = self.enc3(x)       # 80→40
        skip4, x = self.enc4(x)       # 40→20
        encoded = self.project_conv(x)
        return skip1, skip2, skip3, skip4, encoded

    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 12, 1, 320, 320)

        Returns
        -------
        out : (B, 1, 320, 320)
        """
        B, T, C, H, W = x.shape   # T = 12

        # ---- Encode every timestep ----
        encoded_list = []
        # We only need the skip connections from the LAST timestep for the decoder
        last_skips = None

        for t in range(T):
            frame = x[:, t]   # (B, 1, 320, 320)
            skip1, skip2, skip3, skip4, encoded = self._encode_frame(frame)
            encoded_list.append(encoded)       # (B, 256, 20, 20)
            if t == T - 1:
                last_skips = (skip1, skip2, skip3, skip4)

        # Stack encoded features → (B, T, 256, 20, 20)
        enc_seq = torch.stack(encoded_list, dim=1)

        # ---- ConvLSTM over time ----
        h_last = self.convlstm(enc_seq)   # (B, 128, 20, 20)

        # ---- Decode with skip connections from last encoder frame ----
        skip1, skip2, skip3, skip4 = last_skips

        out = self.dec1(h_last, skip4)    # (B, 128, 40, 40)
        out = self.dec2(out, skip3)       # (B,  64, 80, 80)
        out = self.dec3(out, skip2)       # (B,  32,160,160)
        out = self.dec4(out, skip1)       # (B,  32,320,320)

        out = self.final_conv(out)        # (B,   1,320,320)
        out = self.sigmoid(out)           # (B,   1,320,320)

        return out

    # ------------------------------------------------------------------ #

    def count_parameters(self) -> int:
        """Print and return the number of trainable parameters."""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable parameters: {total:,}")
        return total
