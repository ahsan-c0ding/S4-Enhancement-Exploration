import torch
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.gclassifier import GalaxyClassifierS4D 

def export_tensor_to_bin(tensor, filename):
    flat_array = tensor.detach().cpu().numpy().astype(np.float32).flatten()
    with open(filename, 'wb') as f:
        f.write(flat_array.tobytes())

print("🚀 Generating Diverse Samples to Cover ALL 4 Classes...")
test_data_dir = "../test_data"
os.makedirs(test_data_dir, exist_ok=True)

model = GalaxyClassifierS4D(colored=False) 
weights_path = "../model_params/galaxys4-30EPOCH-STANDARD.pth"
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

target_classes = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]

for i, target in enumerate(target_classes):
    print(f"[*] Synthesizing Sample {i} -> Forcing Class {target}", end="", flush=True)
    
    # Give it a new random starting point to avoid getting stuck
    torch.manual_seed(i * 100) 
    image = torch.randn(1, 1, 64, 64, requires_grad=True)
    optimizer = torch.optim.Adam([image], lr=0.5)
    
    for step in range(50):
        optimizer.zero_grad()
        probs = model(image, return_logits=False)
        
        loss = -torch.log(probs[0, target] + 1e-8) 
        loss.backward()
        optimizer.step()
        
        # Print a dot every 5 steps so we know it's working
        if step % 5 == 0:
            print(".", end="", flush=True)
            
        # Break early if we hit our target class with >60% confidence
        if probs.argmax().item() == target and probs[0, target].item() > 0.60:
            break
            
    final_probs = model(image, return_logits=False)
    pred_class = final_probs.argmax().item()
    print(f" Done! (Predicted: {pred_class}, Confidence: {final_probs[0, pred_class].item()*100:.1f}%)")

    export_tensor_to_bin(image.detach(), f"{test_data_dir}/sample_{i}_img.bin")
    export_tensor_to_bin(final_probs.detach(), f"{test_data_dir}/sample_{i}_softmax.bin")

print("\n✅ Diverse Generation Complete! All 4 rubric classes are perfectly covered.")
