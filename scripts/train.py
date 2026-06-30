#!/usr/bin/env python
# coding: utf-8

# # GalaxyMNIST S4 Model Training, Evaluation, and Weight Export (PyTorch Version)

# ## About GalaxyMNIST
# 
# GalaxyMNIST is a dataset of galaxy morphology images designed as an astronomy-specific alternative to traditional benchmark datasets like MNIST. Created by Mike Walmsley and colleagues, it contains 10,000 galaxies from the Galaxy Zoo project, each labeled as one of four morphological types:
# 
# - **Smooth Round**: Elliptical galaxies with smooth, featureless light distributions
# - **Smooth Cigar**: Elongated elliptical galaxies viewed edge-on
# - **Edge-on Disk**: Spiral galaxies viewed edge-on, showing a thin disk structure
# - **Unbarred Spiral**: Face-on spiral galaxies with visible spiral arm patterns
# 
# Each image is 64×64 pixels with 3 color channels (RGB), derived from SDSS imaging data. The dataset presents a more challenging and scientifically relevant classification task compared to handwritten digits, with real-world astronomical noise, varying brightness scales, and subtle morphological differences.
# 
# **References:**
# - Walmsley, M., et al. (2022). "Galaxy Zoo DECaLS: Detailed visual morphology measurements from volunteers and deep learning for 314,000 galaxies." *Monthly Notices of the Royal Astronomical Society*, 509(3), 3966-3988.
# - GalaxyMNIST Repository: https://github.com/mwalmsley/galaxy_mnist
# 
# ---
# 
# **This notebook** demonstrates training a Structured State Space (S4) model for galaxy morphology classification. We convert RGB images to grayscale, flatten them using a Hilbert curve to preserve spatial locality, and process them as 1D sequences of 4,096 pixels. The S4 architecture's ability to capture long-range dependencies makes it well-suited for this task, achieving competitive performance without traditional convolutional layers.

# ## Preliminary Setup
# 
# Note: Python version 3.11.7 is used in this notebook.

# In[ ]:


# Check if GPU is available


# In[ ]:


# If you have a GPU, prefer installing the CUDA version of PyTorch
# Refer to https://pytorch.org/get-started/locally/ for specific instructions.
# For example for CUDA 13.0, you can use the following command:

# For CPU-only installation, you can use the following command:
# %pip install torch torchvision

# Other dependencies


# In[ ]:


# Install GalaxyMNIST from source
# The specific commit used is: https://github.com/mwalmsley/galaxy_mnist/tree/c1fe9853a00bc34b2ff082585c6bb1654d34d239


# ## 1. Imports and Configurations

# In[ ]:


# Standard library
import csv
import random

# Numerical / plotting
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary

# Classifier
from model import GalaxyClassifierS4D
from model.functions import export_model_parameters, load_data

from utils import set_pbar_style


# In[ ]:


set_pbar_style(bar_fill_color="#FFFFFF", text_color="#FFFFFF") # Make progress bars look good in notebooks
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Set device

CLASS_NAMES =  ["Smooth Round", "Smooth Cigar", "Edge-on Disk", "Unbarred Spiral"] # Class names for GalaxyMNIST

# Whether to use colored images
COLORED = False  # Start with grayscale

# Set RNG seeds for reproducibility
# Use your ERP id...
RNG_SEED = 42 # TODO: Replace with your ERP id for actual experiments

# Set seeds
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(RNG_SEED)

print(f"Using RNG seed: {RNG_SEED}")
print(f"Using device: {DEVICE}")


# In[ ]:


# Visualization inside the jupyter

# Load the "autoreload" extension so that code can change

# ----------
# Plot
# ----------
# graph style
sns.set_style("darkgrid")
plt.style.use('fivethirtyeight')

# ----------
# Seaborn rcParams
# ----------
rc={'savefig.dpi': 500, 
    'figure.autolayout': True, 
    'figure.figsize': [17, 12], 
    'axes.labelsize': 18,
    'axes.titlesize': 18, 
    'font.size': 10, 
    'lines.linewidth': 1.0, 
    'lines.markersize': 8, 
    'legend.fontsize': 15,
    'xtick.labelsize': 10, 
    'ytick.labelsize': 10}

sns.set_theme(context='notebook',  # notebook
        style='darkgrid',
        palette='deep',
        color_codes=True, 
        rc=rc)


# ## 2. Load and Preprocess the GalaxyMNIST Dataset
# 
# We load the GalaxyMNIST dataset and preprocess it by converting RGB images to grayscale (averaging across channels) and normalizing pixel values to the [0, 1] range. The labels are converted to one-hot encoding for compatibility with the cross-entropy loss function.

# In[ ]:


X, y_onehot, y = load_data(root="./data", download=True, train=True, colored=COLORED)
NUM_CLASSES = y_onehot.shape[1]


# In[ ]:


# Verify the new dataset size
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y_onehot shape: {y_onehot.shape}")
print(f"Number of classes: {NUM_CLASSES}")


# ### 2.2 Prepare the Test and Train Datasets
# 
# We split the dataset into training (80%) and validation (20%) sets using stratified sampling to maintain class balance. PyTorch DataLoaders are created with a batch size of 64 for efficient mini-batch training.

# In[ ]:


BATCH_SIZE = 16

# Heads up: GalaxyClassifierS4D now uses the recurrent S4D layer (model/s4d_recurrent.py),
# not the old FFT one. Forward is fine either way, but backward through it keeps an autograd
# node for every one of the L=4096 steps, x2 stacked layers -- memory scales with L*B, not
# just B. BATCH_SIZE=16 OOM'd outright on a small (~4GB) box; if training crashes on yours,
# drop this before anything else, and check GPU/RAM headroom either way.

# Split into train/validation sets
x_train, x_val, y_train_onehot, y_val_onehot = train_test_split(X, y_onehot, test_size=0.2, random_state=RNG_SEED, stratify=y)

# Create TensorDatasets
train_ds = TensorDataset(x_train, y_train_onehot)
val_ds = TensorDataset(x_val, y_val_onehot)

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)


# ### 2.3 Save Sample Images for Later Use in C/RISC-V Programs
# 
# We export 100 random training samples to a CSV file for testing inference implementations in lower-level languages. Each row contains the true label followed by the flattened 4,096 pixel values.

# In[ ]:


# This currently makes a CSV dump. 
# **S4 Sequence Processing:**
# We stack two S4D (diagonal state space) layers, each with:
# - State dimension: $d_{state} = 64$ (controls the model's memory capacity)
# - Model dimension: $d_{model} = 64$ (output feature dimension)
# - Activation: GELU after each S4 layer
# 
# The S4 layers model the sequential dependencies across the 4,096-pixel sequence, learning to identify morphological patterns that distinguish galaxy types.
# 
# **Classification Head:**
# - Extract final timestep: Take the last hidden state $(B, 64)$ as the sequence summary
# - Fully connected layer: Map to 4 class logits $(B, 4)$
# - Softmax layer: Convert logits to probability distribution over classes
# 
# **Mathematical Flow:**
# 
# $$X_{img} \in \mathbb{R}^{C \times 64 \times 64} \xrightarrow{\text{Hilbert}} X_{seq} \in \mathbb{R}^{4096 \times C}$$
# 
# $$X_{seq} \xrightarrow{\text{Linear}} X_{proj} \in \mathbb{R}^{4096 \times 64}$$
# 
# $$X_{proj} \xrightarrow{\text{S4D}_1} Z_1 \in \mathbb{R}^{4096 \times 64} \xrightarrow{\text{GELU}} A_1$$
# 
# $$A_1 \xrightarrow{\text{S4D}_2} Z_2 \in \mathbb{R}^{4096 \times 64} \xrightarrow{\text{GELU}} A_2$$
# 
# $$A_2[:, -1, :] \in \mathbb{R}^{64} \xrightarrow{\text{Linear}} Y_{logits} \in \mathbb{R}^{4} \xrightarrow{\text{Softmax}} Y_{probs}$$

# In[ ]:


# Instantiate model
model = GalaxyClassifierS4D(num_classes=NUM_CLASSES, colored=COLORED).to(DEVICE)
model_sum = summary(model, input_size=(2, 1 if not COLORED else 3, 64, 64)) # Summarize model
print(model_sum)


# ## 5. Compile and Train the Model
# 
# We train the model using the Adam optimizer with a learning rate of 0.001 and cross-entropy loss. 

# In[ ]:


optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
loss_fn = nn.CrossEntropyLoss()

# Global training history
# Persistent across training runs
history = {
    "loss": [],
    "val_accuracy": []
}


# In[ ]:


def train(train_loader, val_loader, model, optimizer, loss_fn, epochs, device, verbose=True):
    """Train the model and validate after each epoch.

    Parameters:
    -----------
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    model : nn.Module
        The neural network model to train.
    optimizer : torch.optim.Optimizer
        Optimizer for updating model parameters.
    loss_fn : nn.Module
        Loss function to compute training loss.
    epochs : int
        Number of training epochs.
    device : torch.device
        Device to run the training on (CPU or GPU).

    Returns:
    --------
    history : dict
        Dictionary containing training loss and validation accuracy history.
    """

    history = {
        "loss": [],
        "val_accuracy": []
    }

    ebar = tqdm(range(epochs), desc="Training Progress", disable=verbose)

    # If verbose, don't show outer pbar
    for epoch in ebar:
        model.train()
        running_loss = 0.0

        # show pbar only if verbose
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", disable=not verbose)

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs, return_logits=True)
            loss = loss_fn(outputs, torch.argmax(targets, dim=1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            pbar.set_postfix({"Batch Loss": loss.item()})

        epoch_loss = running_loss / len(train_loader)
        history["loss"].append(epoch_loss)

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, return_logits=True)

                predicted = torch.argmax(outputs, dim=1)
                target = torch.argmax(targets, dim=1)

                correct += (predicted == target).sum().item()
                total += targets.size(0)

        val_accuracy = correct / total
        history["val_accuracy"].append(val_accuracy)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f}")

        ebar.set_postfix({"Loss": epoch_loss, "Val Acc": val_accuracy})

    return history


# In[ ]:


# Change this to atleast 10 for meaningful training
EPOCHS = 10

train_hist = train(train_loader, val_loader, model, optimizer, loss_fn, EPOCHS, DEVICE, verbose=True)

# append to existing history
# This way keep training history across multiple runs
history["loss"].extend(train_hist["loss"])
history["val_accuracy"].extend(train_hist["val_accuracy"])


# ## 6. Evaluate the Model

# ### 6.1 Plot the Training History
# 
# Visualize the training loss curve to assess model convergence and identify potential overfitting or underfitting.

# In[ ]:


# TODO: Plot training loss, and validation accuracy


# ### 6.2 Evaluate the Model on the Test Set
# 
# Compute classification accuracy on the test set to quantify model performance.

# #### 6.2.1 Load the Test set

# In[ ]:


# TODO: Load the test data
X_test, y_test_onehot, y_test = load_data(root="./data", download=True, train=False, colored=COLORED)

test_ds = TensorDataset(X_test, y_test_onehot)
test_loader = DataLoader(test_ds, batch_size=64)


# In[ ]:


model.eval()

correct = 0
total = 0
all_preds = []
with torch.no_grad():
    for imgs, labels in test_loader:

        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        logits = model(imgs, return_logits=True)

        preds = torch.argmax(logits, dim=1)
        target = torch.argmax(labels, dim=1)

        correct += (preds == target).sum().item()
        total += labels.size(0)

        all_preds.extend(preds.cpu().numpy())

test_accuracy = correct / total
print(f"Validation accuracy: {test_accuracy:.4f}")


# In[ ]:


# TODO: Plot confusion matrix


# ### 7. Interactive Galaxy Explorer GUI
# 
# This interactive visualization tool allows you to browse through the validation set and examine the 
# model's predictions in real-time. The GUI displays each galaxy image using the Magma colormap 
# (commonly used in astronomy visualization) alongside the model's softmax probability distribution
# across all four classes.
# 
# Controls
# --------
# - **LEFT/RIGHT Arrow Keys:** Navigate through validation samples
# - **R Key:** Jump to a random sample
# - **M Key:** Toggle Magma colormap on/off
# - **Q Key:** Quit the application
# 
# The visualization highlights the predicted class with a green bar, making it easy to spot correct 
# classifications and identify failure cases where the model might confuse similar morphologies 
# (e.g., smooth round vs. smooth cigar galaxies).
# 
# Assuming `pygame` is installed, if not, you can install it by creating a new code cell in your Jupyter notebook and running:
# ```bash
# %pip install pygame
# ```

# In[ ]:


from model.gui import GalaxyExplorerGUI


# In[ ]:


# Ensure the model is in evaluation mode
model.eval()

# Instantiate the GUI
# x_val and y_val are the tensors we created from the GalaxyMNIST dataset
explorer = GalaxyExplorerGUI(
    model=model, 
    x_val=x_val, 
    y_val=y_val_onehot, 
    device=DEVICE
)
explorer.run()


# ## 8. Export Model Weights for C/RISC-V Programs
# 
# Finally, we export the trained model parameters to files that can be loaded by C or RISC-V implementations. This enables deployment of the model on embedded systems or custom hardware accelerators without requiring a Python runtime.
# 
# The exporter utility serializes all weight matrices, biases, and S4 state space parameters to a format compatible with low-level implementations.

# In[ ]:


export_model_parameters(model, "model_params")

# Save for other python programs (e.g., GUI)
torch.save(model.state_dict(), f"model_params/galaxys4{'-colored' if COLORED else ''}-{RNG_SEED}.pth")


# In[ ]:




