import torch
from model.gclassifier import GalaxyClassifierS4D  # adjust path if needed

# create model
model = GalaxyClassifierS4D(colored=True)

# Test RGB input
x_rgb = torch.randn(1, 3, 64, 64)
out_rgb = model(x_rgb)
print("RGB input shape:", x_rgb.shape, "-> output shape:", out_rgb.shape)

# Test grayscale input
model_gray = GalaxyClassifierS4D(colored=False)
x_gray = torch.randn(1, 1, 64, 64)
out_gray = model_gray(x_gray)
print("Grayscale input shape:", x_gray.shape, "-> output shape:", out_gray.shape)
