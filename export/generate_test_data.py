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

def main():
    print("Generating Per-Layer Reference Data for C Validation...")
    test_data_dir = "../test_data"
    os.makedirs(test_data_dir, exist_ok=True)

    model = GalaxyClassifierS4D(colored=False) 
    weights_path = "../model_params/galaxys4-30EPOCH-STANDARD.pth"
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()

    # Dictionary to hold the output of every layer
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            # S4D returns a tuple (output, state), we only want the output tensor
            if isinstance(output, tuple):
                activations[name] = output[0].detach()
            else:
                activations[name] = output.detach()
        return hook

    # Attach hooks to intercept data passing through every layer
    model.hilbert_scan.register_forward_hook(get_activation('hilbert'))
    model.uproject.register_forward_hook(get_activation('uproject'))
    model.s4_1.register_forward_hook(get_activation('s4_1'))
    model.act1.register_forward_hook(get_activation('gelu_1'))
    model.s4_2.register_forward_hook(get_activation('s4_2'))
    model.act2.register_forward_hook(get_activation('gelu_2'))
    model.take_last.register_forward_hook(get_activation('takelast'))
    model.fc.register_forward_hook(get_activation('fc'))

    num_samples = 5 
    for i in range(num_samples):
        # Use a seed so the random data is identical if you run it twice
        torch.manual_seed(i)
        image = torch.randn(1, 1, 64, 64) 
        
        prefix = f"{test_data_dir}/sample_{i}"

        with torch.no_grad():
            probs = model(image, return_logits=False)
            activations['softmax'] = probs.detach()

        # Export the main image
        export_tensor_to_bin(image, f"{prefix}_img.bin")
        
        # Export every single layer's output
        for layer_name, tensor in activations.items():
            export_tensor_to_bin(tensor, f"{prefix}_{layer_name}.bin")
            
        print(f" Generated full layer data for Sample {i}")

if __name__ == "__main__":
    main()