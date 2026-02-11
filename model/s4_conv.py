import torch
import torch.nn as nn
import torch.nn.functional as F
from .s4_recurrent import S4Recurrent
    
class S4Convolutional(S4Recurrent):
    def compute_kernel(self, L):
        """
        From Recurrence:
        x_1 = A*x_0 + B*u_0
        x_2 = A*x_1 + B*u_1 = A * (A*x_0 + B*u_0) + B*u_1
        x_2 = A^2 * x_0 + A*B*u_0 + B*u_1
        
        so the following can be induced:
        x_k = A^k*x_0 + Sum_of_k(A^k-1-i * B*u_i)
        x_k = A'x_k-1 + B'u_k
        
        We assume:
        x_0 = 0
        Thus:
        x_k = Sum_of_k(A'^k-i * B'u_i)
        
        Subbed into:
        y_k = Sum_of_k(C * A'^k-i * B'u_i + Du_k)
        
        We define the convolution kernel as:
            K_l = C A'^l * B'
        for l = 0 .... L-1
        
        Finally for discrete Convolution:
        y_k = Sum_of_k(K_l * u_k-l + D * u_k)
        """
        A_bar, B_bar = self.discretize() 
        K = []
        
        # v = A_bar^0 B_bar
        v = B_bar
        
        # K_l = C * A'^l * B'
        for l in range(L):
            K_l = self.C @ v    # K_l = C @ v
            K.append(K_l.squeeze())
            
            # Updating v = A_bar v
            v = A_bar @ v
        return torch.stack(K)
    
    def forward(self,u):
        """
        u: (B, L, H)
        """
        B, L, H = u.shape
        assert H == self.d_model
        
        K = self.compute_kernel(L)  # (L,)
        outputs  = []
        
        for h in range (H):
            u_h = u[:, :, h]    # (B, L)
            u_h = u_h.unsqueeze(1)  # reshape for conv1d
            
            kernel = K.flip(0).view(1, 1, L)    # conv kernel must be reversed
            y_h = F.conv1d(u_h, kernel, padding=L-1)
            
            y_h = y_h[:, :, :L] # Trim to length L
            
            outputs.append(y_h.squeeze(1))
            
        y = torch.stack(outputs, dim = -1)
            
        # Add skip connection Du
        y = y + self.D * u
        return y
    
def main():
    model_rec = S4Recurrent(d_model=3)
    model_conv = S4Convolutional(d_model=3)

    model_conv.load_state_dict(model_rec.state_dict())

    u = torch.randn(2, 10, 3)

    y_rec = model_rec(u)
    y_conv = model_conv(u)

    print(torch.norm(y_rec - y_conv)) #tensor(8.1747e-07, grad_fn=<LinalgVectorNormBackward0>)
                                      #i.e mathematically equivelent due to very small difference between both models 
    #print("Input Shape: ",u.shape)
    #print("Output Shape: ",y.shape)
    #print("Norm difference:", torch.norm(y - u)) #checking if anything is actually happening between input and output

    #assert y.shape == (B, L, H)
    #print("Forward Pass Successful!")
    
if __name__ == "__main__":
    main()