import torch
import torch.nn as nn

from .hilbert import HilbertScan
from .tlts import TakeLastTimestep
from .s4d import S4D

class GalaxyClassifierS4D(nn.Module):
    """
    Galaxy classifier using Hilbert Scan and S4 sequence modeling.
    
    This model scans 2D galaxy images into a 1D Hilbert sequence, projects
    the multi-channel pixel values to a higher-dimensional feature space,
    processes the sequence with stacked S4 layers with GELU activations, 
    takes the final timestep as a summary representation, and applies a 
    linear classifier to predict galaxy types.
    
    Parameters
    ----------
    s4_state : int, optional
        Hidden state dimension for the S4 layers (default is 64).
    d_model : int, optional
        Output feature dimension of the S4 layers (default is 64).
    num_classes : int, optional
        Number of output classes (default is 4).
    colored : bool, optional
        If True, expects RGB input images (3 channels); if False, expects
        grayscale images (1 channel) (default is True).
    
    Attributes
    ----------
    seq_len : int
        Sequence length after Hilbert scan (64*64 = 4096).
    d_model : int
        Dimension of the S4 output features.
    hilbert_channels : int
        Number of input channels (1 for grayscale, 3 for RGB).
    hilbert_scan : HilbertScan
        Layer that converts 2D images into 1D sequences using a Hilbert scan.
    uproject : nn.Linear
        Linear projection mapping hilbert_channels to d_model dimensions.
    s4_1 : S4D
        First S4 layer.
    act1 : nn.GELU
        GELU activation after the first S4 layer.
    s4_2 : S4D
        Second S4 layer.
    act2 : nn.GELU
        GELU activation after the second S4 layer.
    take_last : TakeLastTimestep
        Layer that extracts the last timestep from the sequence.
    fc : nn.Linear
        Linear classifier mapping S4 features to output classes.
    softmax : nn.Softmax
        Softmax layer for output probabilities.
    """
    def __init__(self, s4_state=64, d_model=64, num_classes=4, colored=True):
        super().__init__()
        self.seq_len = 64 * 64 
        self.d_model = d_model

        # Hilbert Scan layer
        self.hilbert_scan = HilbertScan()
        self.hilbert_channels = 1 if not colored else 3

        self.uproject = nn.Linear(self.hilbert_channels, d_model)

        # S4 layers
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
        Forward pass of the PixelS4Galaxy model.
        
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
            If return_logits=True: Output logits of shape (B, num_classes),
            representing unnormalized scores for each galaxy class.
            If return_logits=False: Output probabilities of shape (B, num_classes),
            representing the softmax probability distribution over classes.
        """
        # TODO: Implement the forward pass
        raise NotImplementedError("Forward method not implemented yet.")