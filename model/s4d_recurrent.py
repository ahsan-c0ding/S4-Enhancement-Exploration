import math
import torch
import torch.nn as nn
from einops import repeat

class S4D(nn.Module):
    """
    Diagonal Structured State Space (S4D) layer -- recurrent version.

    Same parameterization (log_dt, log_A_real, A_imag, C, D) as the FFT-based
    S4D layer this replaced, so the existing trained checkpoint
    (model_params/galaxys4-30609.pth) loads here with zero remapping -- verified
    against it directly, see images/recurrent_vs_causal_conv_verification.png.
    The only thing that changed is forward(): instead of building a kernel and
    convolving with it, we just step through the sequence one timestep at a
    time, the way the report's Section III.A recurrence does it.

    Why bother, if the FFT version already worked? Two reasons:
    1. It's the form we actually need once we leave PyTorch -- C and RISC-V
       don't get FFT for free, but a per-step multiply-accumulate is just a loop.
    2. Because A is diagonal here (unlike the dense S4Recurrent from Milestone 1),
       each step is O(N) instead of O(N^2) -- no matrix_exp, no linalg.solve,
       just elementwise complex arithmetic on (h, n//2) tensors.

    Parameters
    ----------
    d_model : int
        Input/output feature dimension (number of independent SSM copies).
    d_state : int, optional
        Latent state dimension (must be even). Default 64.
    dt_min, dt_max : float, optional
        Range for the (log-uniform) initial discretization timestep.
    transposed : bool, optional
        If True, expects (B, H, L); if False, expects (B, L, H). Default True.
    lr : float, optional
        Custom LR for the SSM core params, passed straight to register().

    Input
    -----
    u : torch.Tensor, shape (B, H, L) if transposed else (B, L, H)

    Returns
    -------
    y : torch.Tensor, same shape as u
    None
        Kept only so the call signature matches what gclassifier.py expects (s4_out, _ = self.s4_1(x)).
    """
    def __init__(self, d_model, d_state=64, dt_min=0.001, dt_max=0.1, transposed=True, lr=None):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.transposed = transposed

        # --- Initial Parameter Tensors (identical init to s4d.py) ---
        log_dt = torch.rand(self.h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        log_A_real = torch.log(0.5 * torch.ones(self.h, self.n // 2))
        A_imag = math.pi * repeat(torch.arange(self.n // 2), 'n -> h n', h=self.h)
        C_init = torch.randn(self.h, self.n // 2, dtype=torch.cfloat)

        self.register("log_dt", log_dt, lr)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

        self.C = nn.Parameter(torch.view_as_real(C_init))
        self.D = nn.Parameter(torch.randn(self.h))

    def register(self, name, tensor, lr=None):
        # this optim-metadata logic is copied straight from the old FFT layer on
        # purpose -- keeping it identical is what makes the two state_dicts compatible.
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)

    def discretize(self):
        """
        Turn the continuous-time (A, dt) into the discrete step matrices.

        Because A is diagonal we get to skip the whole matrix_exp / linalg.solve
        dance from S4Recurrent -- exp() and division are elementwise here.
        B is fixed at 1 (it's not a learned param in this parameterization,
        wasn't in the old FFT layer either), so B_bar reduces to (A_bar - 1) / A.
        """
        dt = torch.exp(self.log_dt)                          # (h,)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag    # (h, n//2)

        dtA = A * dt.unsqueeze(-1)                            # (h, n//2)
        A_bar = torch.exp(dtA)                                # e^(dt*A), the per-step decay
        B_bar = (A_bar - 1.) / A                              # ZOH input matrix, B=1 folded in

        return A_bar, B_bar

    def forward(self, u):
        if not self.transposed:
            u = u.transpose(-1, -2)   # work in (B, H, L) internally either way
        B, H, L = u.shape
        assert H == self.h

        A_bar, B_bar = self.discretize()
        C = torch.view_as_complex(self.C)   # (h, n//2)

        # state starts at zero, same as every other SSM in this repo
        x = torch.zeros(B, H, self.n // 2, dtype=torch.cfloat, device=u.device)
        outputs = []

        for t in range(L):
            u_t = u[:, :, t]                              # (B, H), real
            x = A_bar * x + B_bar * u_t.unsqueeze(-1)     # x_t = A_bar*x_{t-1} + B_bar*u_t

            # conjugate pairs are implicit (we only ever stored n//2 of them),
            # so doubling the real part here is what reconstructs the full sum
            y_t = 2 * (C * x).sum(-1).real + self.D * u_t
            outputs.append(y_t)

        y = torch.stack(outputs, dim=-1)   # (B, H, L)

        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


def main():
    # quick gut-check now that there's no FFT layer left to compare against --
    # just confirm forward+backward actually run and the shapes/values look sane.
    torch.manual_seed(0)
    B, H, L = 2, 8, 64

    u = torch.randn(B, H, L, requires_grad=True)
    model = S4D(d_model=H)

    y, _ = model(u)
    assert y.shape == u.shape, f"shape mismatch: {y.shape} vs {u.shape}"
    assert torch.isfinite(y).all(), "got NaN/inf in output, something's wrong with discretize()"

    y.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all(), "backward pass broke somewhere"

    print(f"forward OK: {u.shape} -> {y.shape}, backward OK, no NaNs")


if __name__ == "__main__":
    main()
