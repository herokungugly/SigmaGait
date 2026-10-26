import math
import torch
from .base import BaseLoss


class BarlowTwinsLoss(BaseLoss):
    """
    Barlow-Twins loss based on the https://arxiv.org/abs/2103.03230 paper.
    Barlow-Twins loss that is numerically safe for fp16 *and* keeps its
    magnitude independent of the embedding dimension.

    * Accepts inputs of shape  [N, S, H, W]  or  [N, 1, H, W]
    * Works with batch-size 1 (although >1 is preferred for better stats)
    """
    def __init__( self, lambda_offdiag: float = 5e-3, eps: float = 1e-6, loss_term_weight: float = 1.0):
        super().__init__(loss_term_weight)
        self.lambda_offdiag = lambda_offdiag
        self.eps = eps

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor, weight):
        """
        x_a, x_b : tensors from two encoders, same shape [N, S, H, W]
        returns  : (scalar loss, info-dict)
        """
        N = x_a.size(0)  # batch
        z_a = x_a.view(N, -1).float()  # → [N, D]  in fp32
        z_b = x_b.view(N, -1).float()

        # ---------- 1. per-channel batch-norm ---------------------------
        mean_a, mean_b = z_a.mean(0, keepdim=True), z_b.mean(0, keepdim=True)
        var_a  = z_a.var(0, unbiased=False, keepdim=True)
        var_b  = z_b.var(0, unbiased=False, keepdim=True)

        # guard against zero variance (important for tiny batches)
        z_a = (z_a - mean_a) * torch.rsqrt(var_a + self.eps)
        z_b = (z_b - mean_b) * torch.rsqrt(var_b + self.eps)

        # ---------- 2. scale by 1/√D to cap magnitudes ------------------
        inv_sqrt_D = 1.0 / math.sqrt(x_a.shape[1])
        z_a.mul_(inv_sqrt_D)
        z_b.mul_(inv_sqrt_D)

        # ---------- 3. cross-correlation matrix -------------------------
        c = (z_a.T @ z_b) / float(N)  # [D, D]

        # on-diagonal (invariance) term
        on_diag  = torch.diagonal(c) - 1.0  # target is 1
        on_loss  = on_diag.pow_(2).sum()  # **mean**, not sum

        # off-diagonal (decorrelation) term
        off_diag = c - torch.diag_embed(torch.diagonal(c))
        off_loss = off_diag.pow_(2).sum()  # **mean**, not sum

        loss = (on_loss + self.lambda_offdiag * off_loss) * self.loss_term_weight * weight

        self.info.update({
            "loss": loss.detach(),
            "on_max": on_diag.abs().max().detach(),
            "off_max": off_diag.abs().max().detach(),
        })
        return loss, self.info


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a = torch.randn(1, 1, 128, 64, dtype=torch.float16, device=device)
    b = torch.randn_like(a)
    loss_fn = BarlowTwinsLoss()
    loss, info = loss_fn(a, b)
    print("loss:", loss.item(), info)
