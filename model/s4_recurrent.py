import torch
import torch.nn as nn

class RecurrentS4(nn.Module):
    def __init__(self, A, B, C, D):
        super().__init__()
        self.A = nn.Parameter(A)    # A ∈ ℝ^{N×N}: state transition matrix
        self.B = nn.Parameter(B)    # B ∈ ℝ^{N×1}: input-to-state matrix
        self.C = nn.Parameter(C)    # C ∈ ℝ^{1×N}: state-to-output matrix
        self.D = nn.Parameter(D)    # D ∈ ℝ^{1×1}: direct input-to-output term
        
    def forward(self, u, x):
        
        x_next = x @ self.A.T + u.unsqueeze(-1) @ self.B.T  #x_{t+1} = A x_t + B u_t
        
        y = x_next @ self.C.T + u.unsqueeze(-1) @ self.D.T  #y_t = C x_{t+1} + D u_t
        
        return y,x_next

def main():
    batch_size = 2
    T = 5
    N = 4
    
    A = torch.randn(N,N)
    B = torch.randn(N,1)
    C = torch.randn(1,N)
    D = torch.randn(1,1)
    
    model = RecurrentS4(A,B,C,D)
    
    u = torch.randn(batch_size,T)
    x = torch.zeros(batch_size, N)
    
    outputs = []
    
    for t in range(T):
        y, x = model (u[:, t], x)
        outputs.append(y)
    
    outputs = torch.stack(outputs, dim=1)
    print(outputs.shape)
    assert x.shape == (batch_size, N)
    assert outputs.shape == (batch_size, T, 1)

if __name__ == "__main__":
    main()