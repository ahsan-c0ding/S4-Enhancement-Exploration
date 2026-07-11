import torch
import torch.nn as nn

from .cnn_stem import CNNStem
from .hilbert import HilbertScan
from .tlts import TakeLastTimestep
from .s4d_recurrent import S4D


class GalaxyClassifierCNNS4D(nn.Module):
    """
    CNN-stem -> S4D hybrid galaxy classifier.

    Companion/competitor to GalaxyClassifierS4D (model/gclassifier.py). The
    baseline assumes long-range pixel dependency matters for galaxy
    morphology (it scans the full 4096-pixel image straight into S4D). This
    model tests the opposite hypothesis: that morphology is dominated by
    local structure (arm curvature, edge sharpness, blob shape), so a small
    CNN stem can do local feature extraction + spatial downsampling first,
    handing S4D a much shorter sequence, while preserving accuracy.

    image (B,C,64,64)
      -> CNNStem                          -> (B, d_model, grid, grid)
      -> HilbertScan(n=grid)              -> (B, grid*grid, d_model)
      -> S4D(d_model, d_state) s4_1       -> (B, grid*grid, d_model)
      -> GELU
      -> S4D(d_model, d_state) s4_2       -> (B, grid*grid, d_model)
      -> GELU
      -> TakeLastTimestep                 -> (B, d_model)
      -> Linear(d_model, num_classes) fc  -> (B, num_classes)
      -> softmax (or raw logits, matching GalaxyClassifierS4D's API exactly)

    With the default stem_reduction=16, grid=16, so seq_len=256 -- a 16x cut
    from the baseline's 4096, and therefore ~16x fewer S4D-loop ops (S4D's
    per-layer cost is O(d_model * seq_len * d_state/2), linear in seq_len --
    see the FLOPS comment in model/gclassifier.py for the exact op-count
    formula this scales).

    The CNN stem's last conv already projects channels up to d_model, so --
    unlike GalaxyClassifierS4D -- there is no separate `uproject` Linear
    here; the stem's output channel dim *is* the projection.

    S4D itself (model/s4d_recurrent.py) is reused unmodified: d_state stays
    64 per the project's fixed constraints, only seq_len shrinks, and only
    because of what happens upstream in the stem.

    Parameters
    ----------
    s4_state : int, optional
        Hidden state dimension for the S4D layers (default 64).
    d_model : int, optional
        Output feature dimension of the CNN stem / S4D layers (default 64).
    num_classes : int, optional
        Number of output classes (default 4).
    colored : bool, optional
        If True, expects RGB input images (3 channels); if False, expects
        grayscale images (1 channel) (default True, matching
        GalaxyClassifierS4D's default so the two classes are drop-in
        compatible).
    stem_reduction : int, optional
        Sequence-length reduction factor applied by the CNN stem before
        Hilbert-scanning, one of {4, 16}:
          - 16 (default): primary variant, two stride-2 conv blocks,
            64x64 -> 16x16, seq_len 4096 -> 256.
          - 4: milder fallback, single stride-2 conv block,
            64x64 -> 32x32, seq_len 4096 -> 1024. Kept as a fallback data
            point in case the 16x cut degrades accuracy too much to be a
            fair comparison against the baseline.
    mid_channels : int, optional
        Hidden channel width inside the stem (only used by the
        stem_reduction=16 variant, which has two conv layers). Default 16.

    Attributes
    ----------
    seq_len : int
        Sequence length after the CNN stem + Hilbert scan (256 for
        stem_reduction=16, 1024 for stem_reduction=4).
    d_model : int
        Dimension of the S4D output features.
    hilbert_channels : int
        Number of input image channels (1 for grayscale, 3 for RGB).
    cnn_stem : CNNStem
        Conv stem doing local feature extraction, downsampling, and
        channel projection to d_model.
    hilbert_scan : HilbertScan
        Scans the stem's (B, d_model, grid, grid) feature map into a 1D
        sequence via a Hilbert curve over the grid.
    s4_1, s4_2 : S4D
        Stacked S4D layers (unmodified from the baseline).
    act1, act2 : nn.GELU
    take_last : TakeLastTimestep
    fc : nn.Linear
    softmax : nn.Softmax
    """

    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True,
                 stem_reduction=16, mid_channels=16):
        super().__init__()
        if stem_reduction not in (4, 16):
            raise ValueError(f"stem_reduction must be 4 or 16, got {stem_reduction}")

        self.hilbert_channels = 1 if not colored else 3
        self.d_model = d_model
        self.stem_reduction = stem_reduction

        # Spatial side of the feature grid after the stem: 64 -> 64/sqrt(reduction)
        # reduction=16 -> two stride-2 blocks -> /4 side reduction -> grid=16
        # reduction=4  -> one stride-2 block   -> /2 side reduction -> grid=32
        grid = 64 // (4 if stem_reduction == 16 else 2)
        self.seq_len = grid * grid

        # CNN stem: local feature extraction + downsampling. Its last conv
        # projects channels to d_model, so no separate uproject Linear is
        # needed (unlike the baseline).
        self.cnn_stem = CNNStem(
            in_channels=self.hilbert_channels,
            d_model=d_model,
            mid_channels=mid_channels,
            reduction=stem_reduction,
        )

        # Hilbert scan over the downsampled feature grid (not the raw image)
        self.hilbert_scan = HilbertScan(n=grid)

        # S4D layers -- reused unmodified from the baseline; only seq_len
        # shrinks, because of what happens upstream in the stem.
        self.s4_1 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act1 = nn.GELU()

        self.s4_2 = S4D(d_model=d_model, d_state=s4_state, transposed=False)
        self.act2 = nn.GELU()

        # Take last timestep
        self.take_last = TakeLastTimestep()

        # Classifier
        self.fc = nn.Linear(d_model, num_classes)

        # Softmax for output probabilities
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_logits=False):
        """
        Forward pass of the CNN-stem -> S4D hybrid model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, 64, 64), where B is the batch size
            and C is the number of channels (1 for grayscale, 3 for RGB).
        return_logits : bool, optional
            If True, returns raw logits instead of softmax probabilities
            (default is False).

        Returns
        -------
        output : torch.Tensor
            If return_logits=True: Output logits of shape (B, num_classes).
            If return_logits=False: Output probabilities of shape
            (B, num_classes), the softmax distribution over classes.
        """
        B, C, H, W = x.shape
        assert H == 64 and W == 64, "Expected 64x64"
        assert C == self.hilbert_channels, f"Expected {self.hilbert_channels} channels"

        # 1. CNN stem: local feature extraction + spatial downsampling,
        #    also projects channels -> d_model
        feat = self.cnn_stem(x)  # (B, d_model, grid, grid)

        # 2. Hilbert scan: 2D feature map -> 1D sequence
        x_seq = self.hilbert_scan(feat)  # (B, seq_len, d_model)

        # 3. S4D layer 1 + GELU
        s4_out1, _ = self.s4_1(x_seq)
        a1 = self.act1(s4_out1)  # (B, seq_len, d_model)

        # 4. S4D layer 2 + GELU
        s4_out2, _ = self.s4_2(a1)
        a2 = self.act2(s4_out2)  # (B, seq_len, d_model)

        # 5. Take last timestep
        last = self.take_last(a2)  # (B, d_model)

        # 6. Classifier: d_model -> num_classes
        logits = self.fc(last)  # (B, num_classes)

        # Return logits or softmax
        if return_logits:
            return logits
        return self.softmax(logits)
