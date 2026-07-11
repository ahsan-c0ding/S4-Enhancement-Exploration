import torch
import torch.nn as nn


class CNNStem(nn.Module):
    """
    Convolutional stem for the CNN-stem -> S4D hybrid classifier.

    Does local feature extraction and spatial downsampling *before* the
    sequence ever reaches S4D, so the S4D backbone processes a much shorter
    sequence than the pure Hilbert-scan-> S4D baseline. This is the lever
    that tests whether galaxy morphology classification on this dataset is
    local-structure-dominated (arm curvature, edge sharpness, blob shape) or
    long-range-dependency-dominated.

    Two variants, selected via `reduction`:
      - reduction=16 (default, primary): two stride-2 conv blocks.
            64x64 -> 32x32 -> 16x16   => 16x sequence-length cut (4096->256)
      - reduction=4  (milder fallback): a single stride-2 conv block.
            64x64 -> 32x32            => 4x sequence-length cut  (4096->1024)

    No BatchNorm/LayerNorm anywhere in this module. BatchNorm needs a
    running mean/var (and a sqrt) at train time, which complicates an
    eventual bare-metal (RISC-V) port of this repo -- the whole point of
    this codebase is portability to targets with no transcendentals beyond
    what GELU/S4D already need. Plain conv + GELU only.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (1 grayscale, 3 RGB).
    d_model : int, optional
        Output channel count of the stem. Feeds directly into the S4D
        backbone's d_model, so the stem's last conv doubles as the channel
        projection the pure-S4D baseline needs a separate `uproject` Linear
        for. Default 64.
    mid_channels : int, optional
        Hidden channel width of the first conv block. Only used by the
        reduction=16 variant, which has two conv layers. Default 16.
    reduction : int, optional
        Spatial / sequence-length reduction factor. One of {4, 16}.
        Default 16.

    Input
    -----
    x : torch.Tensor, shape (B, in_channels, 64, 64)

    Returns
    -------
    torch.Tensor
        shape (B, d_model, 16, 16) if reduction=16,
        shape (B, d_model, 32, 32) if reduction=4.
    """

    def __init__(self, in_channels, d_model=64, mid_channels=16, reduction=16):
        super().__init__()
        if reduction not in (4, 16):
            raise ValueError(f"reduction must be 4 or 16, got {reduction}")
        self.reduction = reduction

        if reduction == 16:
            # Two stride-2 blocks, each halving H and W (4x area reduction
            # per block): 64 -> 32 -> 16. Second conv projects straight to
            # d_model, so no separate uproject layer is needed downstream.
            self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=2, padding=1)  # 64->32
            self.act1 = nn.GELU()
            self.conv2 = nn.Conv2d(mid_channels, d_model, kernel_size=3, stride=2, padding=1)       # 32->16
            self.act2 = nn.GELU()
        else:
            # Single stride-2 block, projecting straight to d_model: 64->32.
            self.conv1 = nn.Conv2d(in_channels, d_model, kernel_size=3, stride=2, padding=1)        # 64->32
            self.act1 = nn.GELU()
            self.conv2 = None
            self.act2 = None

    def forward(self, x):
        # x: (B, in_channels, 64, 64)
        x = self.act1(self.conv1(x))
        if self.conv2 is not None:
            x = self.act2(self.conv2(x))
        return x
