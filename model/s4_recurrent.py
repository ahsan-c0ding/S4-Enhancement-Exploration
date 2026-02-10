import torch
import torch.nn as nn

class RecurrentS4(nn.Module):
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1): 
        super().__init__()
        self.d_model = d_model  #number of features
        self.d_state = d_state  #state dimension per feature
        A = -torch.eye(d_state)  #initilizes A as a -ve identity matrix (all zeros except diagonals are -1)
        
        self.A = nn.Parameter(A)# A ∈ R^{N×N}: state transition matrix
        self.B = nn.Parameter(torch.randn(d_state, 1))    # B ∈ R^{N×1}: input-to-state matrix
        self.C = nn.Parameter(torch.randn(1,d_state))    # C ∈ R^{1×N}: state-to-output matrix
        self.D = nn.Parameter(torch.randn(1,1))    # D ∈ R^{1×1}: direct input-to-output term
        
        #log-step size
        log_dt = torch.rand(1) * (torch.log(torch.tensor(dt_max)) - torch.log(torch.tensor(dt_min))) + torch.log(torch.tensor(dt_min))
        self.log_dt = nn.Parameter(log_dt)
        
        
    def discretize(self):
        #discretize continuous time (A,B) into discrete steps
        # dx/dt = A x + B u  --> x_{k+1} = A_bar x_k + B_bar u_k
        
        #step-size is time, so cannot be -ve
        dt = torch.exp(self.log_dt)
        
        # This corresponds to the homogeneous solution of the ODE
        A_bar = torch.matrix_exp(dt * self.A)    #A' = e^(delta A)
       
        # Identity matrix I ∈ R^{N×N}
        I = torch.eye(self.d_state, device=self.A.device)
        
        B_bar = torch.linalg.solve(self.A, (A_bar - I)) @ self.B   #computes A^{-1}X without expensive matrix inversion
        
        return A_bar, B_bar
        
    def forward(self, u): #reccurence step cost is limited by matrix multiplication of A_bar(xk-1), on N(d_state) dimensional matrix = N * N = N^2
        B,L,H = u.shape   #step preformed for at each time step for sequence length L 
        assert H == self.d_model #final Big-O = O(L * (N^2)) per sequence
        
        A_bar, B_bar = self.discretize()
        
        #initial state x_0 = 0
        x = torch.zeros(B, H, self.d_state, device = u.device)  
        outputs = []        
        
        for k in range(L):
            u_k = u[:,k,:]  # u_k ∈ R^{B×H}
            
            u_k = u_k.unsqueeze(-1)  #converts u_k into column vector for matrix operations
            
            #for each batch 'b', and feature 'h', multiply A' or B' or C or D with appropriate state vector
            x = (torch.einsum("ij, bhj -> bhi", A_bar, x) + #A_bar @ x_{k-1}
            torch.einsum("ij, bhj -> bhi", B_bar, u_k)) #B_bar @ u_k

            y = (torch.einsum("ij, bhj -> bhi", self.C, x) + # C @ x_k
            torch.einsum("ij, bhj -> bhi", self.D, u_k)) # D @ u_k
            
            outputs.append(y.squeeze(-1)) #removes the "dummy" dimension added in previous comment for math usage
        return torch.stack(outputs, dim=1)
             

def main():
    model = RecurrentS4(d_model=3)
    u = torch.randn(2, 5, 3)
    y = model(u)

    print(y.shape)  #should output torch.Size([2, 5, 3])


if __name__ == "__main__":
    main()