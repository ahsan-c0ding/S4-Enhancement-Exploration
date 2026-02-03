import torch   
import torch.nn as nn


class HilbertScan(nn.Module):
    """
    Reorders pixels according to a Hilbert Curve for multi-channel images.
    
    The Hilbert curve is a space-filling curve that preserves spatial locality
    when mapping 2D coordinates to 1D sequences. This module applies the same
    Hilbert curve pattern to each channel independently, then reorganizes the
    output so the sequence dimension comes first.
    
    Supports grayscale (C=1) or RGB (C=3) images.
    
    Attributes
    ----------
    indices : torch.LongTensor
        Precomputed Hilbert curve indices for a 64×64 grid, stored as a
        non-trainable buffer.
    
    Input
    -----
    x : torch.Tensor
        Input tensor of shape (B, C, H, W), where
        B : batch size
        C : number of channels (1 for grayscale, 3 for RGB)
        H : height (64)
        W : width (64)
    
    Returns
    -------
    out : torch.Tensor
        Reordered tensor of shape (B, seq_len, C) where seq_len = H*W = 4096.
        Pixels are arranged according to the Hilbert curve traversal order.
    """
    def __init__(self):
        """Initialize HilbertScan with precomputed indices for 64x64 images."""
        super().__init__()
        indices = self.get_hilbert_indices(64)
        self.register_buffer('indices', indices)

    def _d2xy(self, n, d):
        """
        Convert 1D Hilbert curve distance to 2D coordinates.
        
        This implements the Hilbert curve mapping algorithm that converts
        a linear distance along the curve to (x, y) coordinates.
        
        Parameters
        ----------
        n : int
            Size of the grid (must be a power of 2).
        d : int
            Distance along the Hilbert curve (0 to n²-1).
        
        Returns
        -------
        tuple of int
            (x, y) coordinates in the grid.
        """
        # TODO: Implement the d2xy conversion algorithm
        raise NotImplementedError("This method should be implemented to convert d to (x, y).")

    def get_hilbert_indices(self, n):
        """
        Generate Hilbert curve indices for an n x n grid.
        
        Creates a lookup table that maps Hilbert curve positions to
        flattened array indices for a 2D grid.
        
        Parameters
        ----------
        n : int
            Grid size (must be a power of 2).
        
        Returns
        -------
        torch.LongTensor
            Tensor of shape (n²,) containing flattened indices following
            the Hilbert curve traversal order.
        """
        indices = []
        for d in range(n * n):
            x, y = self._d2xy(n, d)
            # GalaxyMNIST is 64x64, power of 2
            if x < 64 and y < 64:
                indices.append(y * 64 + x)
        return torch.LongTensor(indices)

    def forward(self, x):
        """
        Apply Hilbert curve reordering to input images.
        
        Parameters
        ----------
        x : torch.Tensor
            Input images of shape (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Reordered tensor of shape (B, seq_len, C) where seq_len = H*W,
            with pixels arranged in Hilbert curve order.
        """
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x = x.view(B, C, -1)           # Flatten each channel: (B, C, H*W)
        x = x[:, :, self.indices]      # Reorder according to Hilbert: (B, C, H*W)
        x = x.permute(0, 2, 1)         # (B, seq_len, C) so sequence dimension is 1D
        return x
