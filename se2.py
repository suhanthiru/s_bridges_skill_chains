"""SE(2) Lie-group operations and the manifold interface used by the bridge solver.

Two manifolds implement the same interface, so every downstream component
(reference SDEs, bridge matching, IPF, rollout) is written once:

  Flat : (x, y, theta) treated as R^3.  World-frame noise, linear interpolation,
         Gaussian uncertainty in R^3.  This is the naive baseline.
  SE2  : group elements.  Body-frame (left-invariant) noise, geodesic
         interpolation, Gaussian uncertainty in the tangent space se(2).

A manifold supplies:
  interp(g0, g1, tau)  point on the path from g0 to g1
  logmap(g, h)         tangent vector at g pointing to h, in the manifold's
                       tangent coordinates (world for Flat, body for SE2)
  retract(g, xi)       move from g along xi
  frame(g)             3x3 matrix taking BODY-frame tangent vectors into the
                       manifold's tangent coordinates (so a body-frame slip
                       covariance S becomes F S F^T)
  feats(g, goal, tau)  network input features

Evaluation metrics (W2, Mahalanobis) always use the SE(2)-correct metric
regardless of which manifold a method was built on, so the comparison is fair.
"""
import math
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HEADING_W = 0.3  # metres per radian, for turning SE(2) tangent vectors into a single distance


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(theta):
    c, s = theta.cos(), theta.sin()
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def rot3(theta):
    """Body -> world tangent map: blockdiag(R(theta), 1)."""
    n = theta.shape[0]
    F = torch.zeros(n, 3, 3, device=theta.device, dtype=theta.dtype)
    F[:, :2, :2] = rot(theta)
    F[:, 2, 2] = 1.0
    return F


def compose(g1, g2):
    t = g1[:, :2] + torch.einsum("nij,nj->ni", rot(g1[:, 2]), g2[:, :2])
    return torch.cat([t, wrap(g1[:, 2:3] + g2[:, 2:3])], 1)


def inv(g):
    t = -torch.einsum("nij,nj->ni", rot(-g[:, 2]), g[:, :2])
    return torch.cat([t, -g[:, 2:3]], 1)


def _ab(phi):
    """a = sin(phi)/phi, b = (1-cos phi)/phi, with Taylor fallback near 0."""
    small = phi.abs() < 1e-4
    p = torch.where(small, torch.ones_like(phi), phi)
    a = torch.where(small, 1 - phi ** 2 / 6, p.sin() / p)
    b = torch.where(small, phi / 2 - phi ** 3 / 24, (1 - p.cos()) / p)
    return a, b


def exp_se2(xi):
    """xi = (rho_x, rho_y, phi) in se(2) -> group element."""
    phi = xi[:, 2]
    a, b = _ab(phi)
    V = torch.stack([torch.stack([a, -b], -1), torch.stack([b, a], -1)], -2)
    t = torch.einsum("nij,nj->ni", V, xi[:, :2])
    return torch.cat([t, wrap(phi).unsqueeze(1)], 1)


def log_se2(g):
    phi = wrap(g[:, 2])
    a, b = _ab(phi)
    det = (a ** 2 + b ** 2).clamp_min(1e-12)
    Vi = torch.stack([torch.stack([a, b], -1), torch.stack([-b, a], -1)], -2) / det[:, None, None]
    rho = torch.einsum("nij,nj->ni", Vi, g[:, :2])
    return torch.cat([rho, phi.unsqueeze(1)], 1)


def between(g, h):
    """Body-frame tangent vector at g pointing to h: log(g^-1 h)."""
    return log_se2(compose(inv(g), h))


class Flat:
    name = "flat"
    dim = 3

    @staticmethod
    def interp(g0, g1, tau):
        d = g1 - g0
        d = torch.cat([d[:, :2], wrap(d[:, 2:3])], 1)
        return torch.cat([(g0 + tau.unsqueeze(1) * d)[:, :2], wrap(g0[:, 2:3] + tau.unsqueeze(1) * d[:, 2:3])], 1)

    @staticmethod
    def interp_from(g0, vnom, tau):
        """interp(g0, g1, tau) with vnom = logmap(g0, g1) precomputed."""
        p = g0 + tau.unsqueeze(1) * vnom
        return torch.cat([p[:, :2], wrap(p[:, 2:3])], 1)

    @staticmethod
    def logmap(g, h):
        d = h - g
        return torch.cat([d[:, :2], wrap(d[:, 2:3])], 1)

    @staticmethod
    def retract(g, xi):
        return torch.cat([g[:, :2] + xi[:, :2], wrap(g[:, 2:3] + xi[:, 2:3])], 1)

    @staticmethod
    def frame(g):
        return rot3(g[:, 2])

    @staticmethod
    def feats(g, goal, tau):
        d = Flat.logmap(g, goal)
        return torch.cat([g[:, :2], g[:, 2:3].cos(), g[:, 2:3].sin(), d, tau.unsqueeze(1)], 1)

    n_feat = 8


class SE2:
    name = "se2"
    dim = 3

    @staticmethod
    def interp(g0, g1, tau):
        return compose(g0, exp_se2(tau.unsqueeze(1) * between(g0, g1)))

    @staticmethod
    def interp_from(g0, vnom, tau):
        return compose(g0, exp_se2(tau.unsqueeze(1) * vnom))

    @staticmethod
    def logmap(g, h):
        return between(g, h)

    @staticmethod
    def retract(g, xi):
        return compose(g, exp_se2(xi))

    @staticmethod
    def frame(g):
        return torch.eye(3, device=g.device, dtype=g.dtype).expand(g.shape[0], 3, 3)

    @staticmethod
    def feats(g, goal, tau):
        """Left-invariant features: everything expressed in the body frame."""
        d = between(g, goal)
        return torch.cat([d, d[:, :2].norm(dim=1, keepdim=True), tau.unsqueeze(1)], 1)

    n_feat = 5


MANIFOLDS = {"flat": Flat, "se2": SE2}


# --------------------------------------------------------------- metrics
def se2_dist2(a, b):
    """Squared SE(2) distance matrix between two clouds, tangent metric."""
    n, m = a.shape[0], b.shape[0]
    aa = a[:, None, :].expand(n, m, 3).reshape(-1, 3)
    bb = b[None, :, :].expand(n, m, 3).reshape(-1, 3)
    xi = between(aa, bb)
    d2 = (xi[:, :2] ** 2).sum(1) + (HEADING_W * xi[:, 2]) ** 2
    return d2.reshape(n, m)


def w2_se2(a, b, max_n=400, seed=0):
    """Exact empirical 2-Wasserstein on SE(2) by assignment (sub-sampled to max_n)."""
    a, b = torch.as_tensor(a).float().cpu(), torch.as_tensor(b).float().cpu()
    g = torch.Generator().manual_seed(seed)
    if a.shape[0] > max_n:
        a = a[torch.randperm(a.shape[0], generator=g)[:max_n]]
    if b.shape[0] > max_n:
        b = b[torch.randperm(b.shape[0], generator=g)[:max_n]]
    n = min(a.shape[0], b.shape[0])
    if n < 2:
        return float("nan")
    C = se2_dist2(a[:n], b[:n]).numpy().astype(np.float64)
    r, c = linear_sum_assignment(C)
    return float(np.sqrt(C[r, c].mean()))


def mahalanobis(g, mean, cov):
    """SE(2)-correct Mahalanobis distance of g to a tangent-space Gaussian at `mean`."""
    xi = between(mean.expand_as(g), g)
    Si = torch.linalg.inv(cov)
    return torch.einsum("ni,ij,nj->n", xi, Si, xi).clamp_min(0).sqrt()


def sample_marginal(mean, cov, n, device, gen=None):
    """Sample a tangent-space Gaussian marginal on SE(2): g = mean * exp(xi)."""
    L = torch.linalg.cholesky(cov)
    xi = torch.randn(n, 3, generator=gen, device=device) @ L.T
    return compose(mean.expand(n, 3), exp_se2(xi))
