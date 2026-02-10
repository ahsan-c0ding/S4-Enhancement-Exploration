import torch
import torch.nn as nn

class RecurrentS4(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model  #number of features
        self.d_state = d_state  #state dimension per feature
        A = -torch.eye(d_state)  #initilizes A as a -ve identity matrix (all zeros except diagonals are -1)
        
        self.A = nn.Parameter(A)# A ∈ ℝ^{N×N}: state transition matrix
        self.B = nn.Parameter(torch.randn(d_state, 1))    # B ∈ ℝ^{N×1}: input-to-state matrix
        self.C = nn.Parameter(torch.randn(1,d_state))    # C ∈ ℝ^{1×N}: state-to-output matrix
        self.D = nn.Parameter(torch.randn(1,1))    # D ∈ ℝ^{1×1}: direct input-to-output term
        
        #log-step size
        log_dt = torch.rand(1) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) + torch.log(torch.tensor(dt_min))
        self.log_dt = nn.Parameter(log_dt)
        
        
    def discretize(self):
        #discretize continuous time (A,B) into discrete steps
        dt = torch.exp(self.log_dt)
        
        A_bar = torch.matrix_exp(dt * self.A)    #A' = e^(delta A)
       
        I = torch.eye(self.d_state, device=self.A.device)
        B_bar = torch.linalg.solve(self.A, (A_bar - I)) @ self.B   #B' = A^(-1) * (A' - I) * B
        
        return A_bar, B_bar
        
    def forward(self, u):
        B,L,H = u.shape
        assert H == self.d_model
        
        A_bar, B_bar = self.discretize()
        
        #initial state
        x = torch.zeros(B, H, self.d_state, device = u.device)
        outputs = []        
        
        for k in range(L):
            u_k = u[:,k,:]
            
            u_k = u_k.unsqueeze(-1)  #converts u_k into column vector for matrix operations
            
            x = torch.einsum("ij, bhj -> bhi", A_bar, x) + torch.einsum("ij, bhj -> bhi", B_bar, u_k) #for each batch 'b', and feature 'h', multiply A' or B' or C or D with appropraite state vector

            y = torch.einsum("ij, bhj -> bhi", self.C, x) + torch.einsum("ij, bhj -> bhi", self.D, u_k)
            
            outputs.append(y.squeeze(-1)) #removes the "dummy" dimension added in previous comment for math usage
        return torch.stack(outputs, dim=1)
             

def main():
    model = RecurrentS4(d_model=3)
    u = torch.randn(2, 5, 3)
    y = model(u)

    print(y.shape)  #should output torch.Size([2, 5, 3])


if __name__ == "__main__":
    main()