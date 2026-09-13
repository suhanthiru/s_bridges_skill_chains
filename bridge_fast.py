"""Fast path for bridge sampling + drift regression targets.

`sde.Reference` holds the mathematics and is what `tests.py` checks.  This
module is the same mathematics specialised to the three shapes the covariance
can actually take, because the training loop runs it tens of millions of times:

  scalar : Sigma = sigma^2 I               (brownian, killed, unicycle)
  diag   : Sigma diagonal                  (slip on SE(2): body frame, frame = I)
  full   : Sigma dense 3x3                 (slip on flat: world frame, R Sigma R^T)

It also caches the per-pair quantities that do not depend on tau (the nominal
velocity and the full-interval Q), and returns the sample and its regression
target together so the interpolant and the deviation xi are computed once.
`tests.py::test_fast_path_matches_reference` checks it against `sde` directly.
"""
import numpy as np
import torch
import terrain as TR

GL_N = 8
_GLX, _GLW = np.polynomial.legendre.leggauss(GL_N)
_CACHE = {}


def _gl(device, dtype):
    key = (str(device), dtype)
    if key not in _CACHE:
        _CACHE[key] = (torch.tensor(_GLX, device=device, dtype=dtype),
                       torch.tensor(_GLW, device=device, dtype=dtype))
    return _CACHE[key]


def mode_of(ref, mf):
    if not ref.state_dependent:
        return "scalar"
    return "diag" if mf.name in ("se2", "se2x") else "full"


def cov_at(ref, g, mf, mode):
    """Diffusion covariance in the shape given by `mode`."""
    if mode == "scalar":
        return torch.full((g.shape[0],), ref.sigma ** 2, device=g.device)
    if ref.body_cov is not None:
        d = ref.body_cov.to(g.device).expand(g.shape[0], 3) + ref.push_sigma ** 2
    else:
        p = TR.lookup(ref.fields, g[:, :2])[:, :3] * ref.slip_scale
        d = p ** 2 + ref.push_sigma ** 2
    if mode == "diag":
        return d
    F = mf.frame(g)
    return F @ torch.diag_embed(d) @ F.transpose(1, 2)


def quad_Q(ref, g0, vnom, s, t, mf, mode):
    """int_s^t exp(-2 kappa (t-r)) Sigma(xbar_r) dr by Gauss-Legendre."""
    X, W = _gl(g0.device, g0.dtype)
    half, mid = (t - s) / 2, (t + s) / 2
    nodes = mid[:, None] + half[:, None] * X
    w = half[:, None] * W * torch.exp(-2 * ref.kappa * (t[:, None] - nodes))
    if mode == "scalar":
        return w.sum(1) * ref.sigma ** 2
    n = g0.shape[0]
    gg = g0.repeat_interleave(GL_N, 0)
    vv = vnom.repeat_interleave(GL_N, 0)
    xbar = mf.interp_from(gg, vv, nodes.reshape(-1))
    S = cov_at(ref, xbar, mf, mode)
    if mode == "diag":
        return (w[:, :, None] * S.reshape(n, GL_N, 3)).sum(1)
    return (w[:, :, None, None] * S.reshape(n, GL_N, 3, 3)).sum(1)


def prepare(ref, g0, g1, mf):
    """Per-pair, tau-independent quantities.  Computed once per fit() call."""
    mode = mode_of(ref, mf)
    vnom = mf.logmap(g0, g1)
    n = g0.shape[0]
    z, o = torch.zeros(n, device=g0.device), torch.ones(n, device=g0.device)
    Q10 = quad_Q(ref, g0, vnom, z, o, mf, mode)
    if mode == "full":
        Q10i = torch.linalg.inv(Q10)
    else:
        Q10i = 1.0 / Q10.clamp_min(1e-12)
    return dict(mode=mode, vnom=vnom, Q10i=Q10i)


def sample_and_target(ref, g0, prep, tau, mf, gen=None, backward=False):
    """Draw x_tau from the reference bridge and return its drift regression target.

    With `backward=True` the roles of the endpoints are already swapped by the
    caller (prep built on the reversed pair), so this is direction-agnostic.
    """
    mode, vnom, Q10i = prep["mode"], prep["vnom"], prep["Q10i"]
    n = g0.shape[0]
    z, o = torch.zeros(n, device=g0.device), torch.ones(n, device=g0.device)
    Qt0 = quad_Q(ref, g0, vnom, z, tau, mf, mode)
    phi = torch.exp(-ref.kappa * (1 - tau))
    xbar = mf.interp_from(g0, vnom, tau)
    eps = torch.randn(n, 3, generator=gen, device=g0.device)
    if mode == "full":
        ph = phi[:, None, None]
        M = Qt0 @ (ph * Q10i)
        P = Qt0 - M @ (ph * Qt0)
        from sde import psd_sqrt
        xi = torch.einsum("nij,nj->ni", psd_sqrt(P), eps)
    else:
        ph = phi[:, None] if mode == "diag" else phi
        P = (Qt0 - Qt0 * ph * Q10i * ph * Qt0).clamp_min(0)
        xi = (P.sqrt() if mode == "diag" else P.sqrt()[:, None]) * eps
    g = mf.retract(xbar, xi)

    rem = (1 - tau).clamp_min(0.02)
    Q1t = quad_Q(ref, g0, vnom, 1 - rem, o, mf, mode)
    phi2 = torch.exp(-ref.kappa * rem)
    Sg = cov_at(ref, g, mf, mode)
    if mode == "full":
        K = Sg @ torch.linalg.inv(Q1t)
        pull = torch.einsum("nij,nj->ni", K, (phi2[:, None] ** 2) * xi)
    elif mode == "diag":
        K = Sg / Q1t.clamp_min(1e-12)
        pull = K * (phi2[:, None] ** 2) * xi
    else:
        K = Sg / Q1t.clamp_min(1e-12)
        pull = (K * phi2 ** 2)[:, None] * xi
    target = vnom - ref.kappa * xi - pull
    if mf.name == "se2x":
        from se2 import adjoint, exp_se2
        target = adjoint(exp_se2(-xi), target)    # xbar-frame velocity -> g-frame velocity
    return g, target
