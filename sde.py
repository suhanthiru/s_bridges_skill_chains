"""Reference SDEs and their Gaussian bridges.

Every reference is an Ornstein-Uhlenbeck deviation from the nominal path
between the two endpoints:

    d(xi) = -kappa xi dt + L(x) dW,     xi = deviation from the interpolant

with kappa = 0 recovering Brownian motion.  Writing the reference this way
buys three things: the bridge (Doob h-transform) is available in closed form,
the position-dependent slip covariance enters through a quadrature rather than
a simulation, and the linear-Gaussian case has a known Schrodinger-bridge
solution that the solver can be tested against.

Bridge quantities, for tau in [0,1], Phi(t,s) = exp(-kappa (t-s)):

    Q(t,s)     = \\int_s^t Phi(t,r)^2 Sigma(xbar_r) dr          (Gauss-Legendre)
    cov(tau)   = Q(tau,0) - Q(tau,0) Phi(1,tau) Q(1,0)^-1 Phi(1,tau) Q(tau,0)
    drift(g)   = vnom(tau) - kappa xi + Sigma(g) Phi(1,tau) Q(1,tau)^-1 (-Phi(1,tau) xi)

where xi = logmap(xbar_tau, g).  For kappa = 0 and constant Sigma this reduces
exactly to the Brownian-bridge drift logmap(g, g1)/(1-tau); `tests.py` checks it.

Approximations, stated plainly: deviations are carried in the tangent space at
the interpolant (first-order parallel transport), and the position-dependent
slip covariance is frozen along the interpolant rather than along the realised
path.  Both are standard and both are documented in the findings.
"""
import math
import numpy as np
import torch
import terrain as TR

GL8_X, GL8_W = np.polynomial.legendre.leggauss(8)
TAU_CLIP = 0.02


class Reference:
    """kind in {brownian, killed, unicycle, slip}."""

    def __init__(self, kind, sigma=0.05, kappa=2.0, slip_scale=1.0, fields=None, push_sigma=0.0):
        self.kind, self.sigma, self.slip_scale, self.push_sigma = kind, sigma, slip_scale, push_sigma
        self.kappa = 0.0 if kind in ("brownian", "killed") else kappa
        self.fields = fields
        self.state_dependent = kind == "slip"

    def cov(self, g, mf):
        """Diffusion covariance in the manifold's tangent coordinates, (n,3,3)."""
        n = g.shape[0]
        if not self.state_dependent:
            return (self.sigma ** 2) * torch.eye(3, device=g.device).expand(n, 3, 3)
        S = TR.slip_cov_body(self.fields, g[:, :2], self.slip_scale)
        S = S + (self.push_sigma ** 2) * torch.eye(3, device=g.device)
        F = mf.frame(g)
        return F @ S @ F.transpose(1, 2)

    # ---------------------------------------------------------------- bridge
    def _Q(self, g0, g1, s, t, mf):
        """int_s^t Phi(t,r)^2 Sigma(xbar_r) dr, s and t are (n,) tensors."""
        n = g0.shape[0]
        half, mid = (t - s) / 2, (t + s) / 2
        nodes = mid[:, None] + half[:, None] * torch.tensor(GL8_X, device=g0.device, dtype=g0.dtype)
        w = half[:, None] * torch.tensor(GL8_W, device=g0.device, dtype=g0.dtype)
        w = w * torch.exp(-2 * self.kappa * (t[:, None] - nodes))
        if not self.state_dependent:
            q = (w.sum(1) * self.sigma ** 2)
            return q[:, None, None] * torch.eye(3, device=g0.device)
        gg0 = g0.repeat_interleave(8, 0); gg1 = g1.repeat_interleave(8, 0)
        xbar = mf.interp(gg0, gg1, nodes.reshape(-1))
        S = self.cov(xbar, mf).reshape(n, 8, 3, 3)
        return (w[:, :, None, None] * S).sum(1)

    def bridge_sample(self, g0, g1, tau, mf, gen=None):
        """Sample x_tau from the reference bridge pinned at g0, g1."""
        n = g0.shape[0]
        zero, one = torch.zeros_like(tau), torch.ones_like(tau)
        Q_t0 = self._Q(g0, g1, zero, tau, mf)
        Q_10 = self._Q(g0, g1, zero, one, mf)
        phi = torch.exp(-self.kappa * (1 - tau))[:, None, None]
        M = Q_t0 @ (phi * torch.linalg.inv(Q_10))
        P = Q_t0 - M @ (phi * Q_t0)
        P = 0.5 * (P + P.transpose(1, 2)) + 1e-10 * torch.eye(3, device=g0.device)
        L = torch.linalg.cholesky(P)
        xi = torch.einsum("nij,nj->ni", L, torch.randn(n, 3, generator=gen, device=g0.device))
        return mf.retract(mf.interp(g0, g1, tau), xi)

    def bridge_drift(self, g, g0, g1, tau, mf):
        """Reference-bridge drift at g (target for regression), tangent coords at g."""
        xbar = mf.interp(g0, g1, tau)
        xi = mf.logmap(xbar, g)
        vnom = mf.logmap(g0, g1)          # constant nominal velocity along the path
        one = torch.ones_like(tau)
        rem = (1 - tau).clamp_min(TAU_CLIP)
        Q_1t = self._Q(g0, g1, one - rem, one, mf)
        phi = torch.exp(-self.kappa * rem)[:, None, None]
        K = self.cov(g, mf) @ (phi * torch.linalg.inv(Q_1t))
        return vnom - self.kappa * xi - torch.einsum("nij,nj->ni", K, (phi[:, :, 0] * xi))


# ------------------------------------------------- linear-Gaussian closed form
def gaussian_sb_coupling(S0, S1, eps):
    """Cross-covariance of the Schrodinger bridge between N(m0,S0) and N(m1,S1)
    under the reference dX = sqrt(eps) dW on [0,1] (entropic-OT Gaussian solution).

        C = 1/2 ( S0^1/2 (4 S0^1/2 S1 S0^1/2 + eps^2 I)^1/2 S0^-1/2 - eps I )
    """
    S0, S1 = np.asarray(S0, float), np.asarray(S1, float)
    R = _sqrtm(S0); Ri = np.linalg.inv(R)
    return 0.5 * (R @ _sqrtm(4 * R @ S1 @ R + eps ** 2 * np.eye(len(S0))) @ Ri - eps * np.eye(len(S0)))


def gaussian_sb_marginal(S0, S1, C, eps, t):
    """Marginal covariance of the Gaussian SB at time t."""
    return ((1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + eps * t * (1 - t) * np.eye(len(S0)))


def _sqrtm(A):
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    return V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
