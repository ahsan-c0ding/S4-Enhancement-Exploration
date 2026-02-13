import torch
from model.gclassifier import GalaxyClassifierS4D

# Create model
model = GalaxyClassifierS4D(num_classes=4, colored=False)
model.eval()

# Create dummy input: 2 images, 1 channel, 64x64
dummy_input = torch.randn(2, 1, 64, 64)

try:
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    if output.shape == (2, 4):
        print("✅ SUCCESS: Forward pass implementation is correct!")
except Exception as e:
    print(f"❌ ERROR in implementation: {e}")
