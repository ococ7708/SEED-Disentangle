"""P0/P1/P2: existing independent S/Q maps, conditional identity adversary.

P3 addition: conditional identity adversary on the RELATIVE logits t = A^T ell
(the actual fused decision), plus a frozen-teacher task-preservation KL is
handled by the training script, not here.
"""
import torch
from torch import nn
import torch.nn.functional as F


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class ConditionalDisentangler(nn.Module):
    def __init__(self, dim, classes, subjects):
        super().__init__()
        def mapping():
            return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.LayerNorm(dim))
        self.shared, self.specific = mapping(), mapping()
        self.identity = nn.ModuleList([nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                                   nn.Linear(64, subjects)) for _ in range(classes)])
        self.specific_id = nn.Linear(dim, subjects)
        # P3: class-conditional identity heads on the relative logits t in R^{K-1}.
        t_dim = classes - 1
        self.t_identity = nn.ModuleList([nn.Sequential(nn.Linear(t_dim, 32), nn.ReLU(),
                                                      nn.Linear(32, subjects)) for _ in range(classes)])

    def forward(self, h):
        return self.shared(h), self.specific(h.detach())

    def identity_loss(self, s, y, r, strength=0.):
        losses = [F.cross_entropy(head(Reverse.apply(s[y == c], strength)), r[y == c])
                  for c, head in enumerate(self.identity) if (y == c).any()]
        return torch.stack(losses).mean()

    def t_identity_loss(self, t, y, r, strength=0.):
        """Class-conditional identity adversary on the relative logits t [B,K-1]."""
        losses = [F.cross_entropy(head(Reverse.apply(t[y == c], strength)), r[y == c])
                  for c, head in enumerate(self.t_identity) if (y == c).any()]
        return torch.stack(losses).mean()

    def t_parameters(self):
        return [p for module in self.t_identity for p in module.parameters()]

    def decorrelation(self, s, q, y):
        losses = []
        for c in y.unique():
            a, b = s[y == c], q[y == c].detach()
            if len(a) < 2:
                continue
            a = (a - a.mean(0)) / a.std(0, unbiased=False).clamp_min(1e-3)
            b = (b - b.mean(0)) / b.std(0, unbiased=False).clamp_min(1e-3)
            losses.append((a.T @ b / len(a)).square().mean())
        return torch.stack(losses).mean() if losses else s.sum() * 0
