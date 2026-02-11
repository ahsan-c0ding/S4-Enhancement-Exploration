import torch
import time

from model.s4_recurrent import S4Recurrent
from model.s4_conv import S4Convolutional

def numerical_equivalence_test(device):
    print("Running equivalnce test....")
    torch.manual_seed(0)
    
    B = 2  #batch size
    L = 100  #sequence length
    H = 16  #no of features
    
    u = torch.randn(B,L,H).to(device)
    
    model_rec = S4Recurrent(d_model=H).to(device)
    model_conv = S4Convolutional(d_model=H).to(device)
    
    model_conv.load_state_dict(model_rec.state_dict()) #copying weights from reccurent to convalutional model
    
    y_rec = model_rec(u)
    y_conv = model_conv(u)
    
    max_diff = torch.max(torch.abs(y_rec - y_conv)).item()
    print(f"Maximum absolute difference: {max_diff:.6e}")
    
    assert max_diff < 1e-5, "CAUTION: Outputs are not nummerically equivalent!"
    
    print("Numerical Equivalence verified \n")
    
    return max_diff

def benchmark(device):
    print("Running Benchmark.... \n")
    torch.manual_seed(0)
    
    B = 2
    H = 16
    lengths = [64, 256, 1024, 4096]
    results = []
    
    for L in lengths:
        u = torch.randn(B, L, H).to(device)
        model_rec = S4Recurrent(d_model=H).to(device)
        model_conv = S4Convolutional(d_model=H).to(device)
    
        model_conv.load_state_dict(model_rec.state_dict())
        _ = model_rec(u)
        _ = model_conv(u)
        
        if device.type == "cuda": #my (ahsan) machine won't use this but ik Abdul Rahim does have a GPU
            torch.cuda.synchronize()
        
        #------------- timings --------------------
        #Recurrent Timing
        start = time.time()
        _ = model_rec(u)
        if device.type == "cuda":
            torch.cuda.synchronize()
        rec_time = time.time() - start
        
        #Convolutional Timing    
        start = time.time()
        _ = model_conv(u)
        if device.type == "cuda":
            torch.cuda.synchronize()
        conv_time = time.time() - start            
        
        results.append((L,rec_time,conv_time))
        
        print(f"L={L:4d} | Recurrent: {rec_time:.6f}s | Convolutional: {conv_time:.6f}s")
        
    return results
    
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: ", device, "\n")
    
    max_diff = numerical_equivalence_test(device)
    results = benchmark(device)
    
    print("\nBenchmark Summary:")
    print("Length | Recurrent (s) | Convolutional (s)")
    print("--------------------------------")
    
    for L,rec_time,conv_time in results:
        print(f"{L:6d} | {rec_time:14.6f} | {conv_time:17.6f}")
    

if __name__ == "__main__":
    main()
        