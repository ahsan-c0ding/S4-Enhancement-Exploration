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

print("🚀 Generating ORGANIC Diverse Samples...")
test_data_dir = "../test_data"
os.makedirs(test_data_dir, exist_ok=True)

model = GalaxyClassifierS4D(colored=False) 
weights_path = "../model_params/galaxys4-30EPOCH-STANDARD.pth"
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

target_classes = [0, 0, 0, 1, 1, 1, 2, 2, 3, 3]

for i, target in enumerate(target_classes):
    print(f"[*] Gently nudging Sample {i} towards Class {target}", end="", flush=True)
    
    torch.manual_seed(i * 42) 
    image = torch.randn(1, 1, 64, 64, requires_grad=True)
    
    # Tiny learning rate for gentle, organic changes
    optimizer = torch.optim.Adam([image], lr=0.02)
    
    for step in range(300):
        optimizer.zero_grad()
        probs = model(image, return_logits=False)
        
        # Stop the EXACT microsecond it organically favors our target class (> 40%)
        if probs.argmax().item() == target and probs[0, target].item() > 0.40:
            break
            
        loss = -torch.log(probs[0, target] + 1e-8) 
        loss.backward()
        optimizer.step()
        
        if step % 20 == 0:
            print(".", end="", flush=True)
            
    final_probs = model(image, return_logits=False)
    pred_class = final_probs.argmax().item()
    p_list = final_probs[0].detach().numpy()
    
    print(f" Done!")
    print(f"    -> Predicted: Class {pred_class}")
    print(f"    -> Organic Spread: [C0: {p_list[0]*100:.1f}%, C1: {p_list[1]*100:.1f}%, C2: {p_list[2]*100:.1f}%, C3: {p_list[3]*100:.1f}%]")

    export_tensor_to_bin(image.detach(), f"{test_data_dir}/sample_{i}_img.bin")
    export_tensor_to_bin(final_probs.detach(), f"{test_data_dir}/sample_{i}_softmax.bin")

print("\n✅ Organic Generation Complete! You now have a messy, realistic probability spread.")
