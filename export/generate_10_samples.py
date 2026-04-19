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

print("Generating 10 Samples and Synchronizing Weights...")
test_data_dir = "../test_data"
os.makedirs(test_data_dir, exist_ok=True)

model = GalaxyClassifierS4D(colored=False) 
weights_path = "../model_params/galaxys4-30EPOCH-STANDARD.pth"
model.load_state_dict(torch.load(weights_path, map_location='cpu'))
model.eval()

# 1. Sync the binary weights for RISC-V
out_path = "../model_params/model_weights.bin"
keys_in_order = [
    'hilbert_scan.indices', 'uproject.weight', 'uproject.bias',
    's4_1.log_dt', 's4_1.log_A_real', 's4_1.A_imag', 's4_1.C', 's4_1.D',
    's4_2.log_dt', 's4_2.log_A_real', 's4_2.A_imag', 's4_2.C', 's4_2.D',
    'fc.weight', 'fc.bias'
]
with open(out_path, 'wb') as f:
    for k in keys_in_order:
        tensor = model.state_dict()[k].cpu().detach()
        if tensor.is_complex(): tensor = torch.view_as_real(tensor)
        if tensor.dtype == torch.int64: tensor = tensor.to(torch.int32)
        elif tensor.dtype == torch.float64: tensor = tensor.to(torch.float32)
        f.write(tensor.numpy().flatten().tobytes())

activations = {}
def get_activation(name):
    def hook(model, input, output):
        if isinstance(output, tuple): activations[name] = output[0].detach()
        else: activations[name] = output.detach()
    return hook

model.hilbert_scan.register_forward_hook(get_activation('hilbert'))
model.take_last.register_forward_hook(get_activation('takelast'))

# 2. Generate 10 test samples
for i in range(10):
    torch.manual_seed(i)
    image = torch.randn(1, 1, 64, 64) 
    prefix = f"{test_data_dir}/sample_{i}"

    with torch.no_grad():
        probs = model(image, return_logits=False)
        activations['softmax'] = probs.detach()

    export_tensor_to_bin(image, f"{prefix}_img.bin")
    export_tensor_to_bin(activations['softmax'], f"{prefix}_softmax.bin")
    print(f"Generated data for Sample {i}")

print("Sync and Generation Complete!")
