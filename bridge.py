"""eps-conditioned Schrodinger bridge skills trained with Diffusion Schrodinger
Bridge Matching (Shi et al. 2023) and executed as an SDE over tau in [0,1].

Reference process: Brownian motion killed on the wall.  We implement the
killing in three places: (1) endpoint pairs whose straight line crosses the
wall are rejected, (2) trajectories that hit the wall during IPF simulation
are dropped from the coupling, (3) a penalty on drift pointing into the wall
for bridge samples within 0.02 of it.
"""
import math
import numpy as np
import torch
import torch.nn as nn
from env import DT, STEPS_PER_SKILL, w2

EPS_MIN, EPS_MAX = 1e-3, 1e-1
LOG_EPS_MIN, LOG_EPS_MAX = math.log(EPS_MIN), math.log(EPS_MAX)


def sinusoidal(v, n_freq=16):
    """v: (n,) -> (n, 2*n_freq)."""
    f = 2.0 ** torch.arange(n_freq, device=v.device, dtype=v.dtype) * math.pi
    a = v.unsqueeze(1) * f
    return torch.cat([a.sin(), a.cos()], 1)


class DriftNet(nn.Module):
    """f(x, tau, eps) -> u in R^2.  4 x 256, SiLU."""

    def __init__(self, hidden=256, n_freq=16):
        super().__init__()
        self.n_freq = n_freq
        d_in = 2 + 4 * n_freq
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2))

    def forward(self, x, tau, log_eps):
        h = torch.cat([x, sinusoidal(tau, self.n_freq),
                       sinusoidal((log_eps - LOG_EPS_MIN) / (LOG_EPS_MAX - LOG_EPS_MIN), self.n_freq)], 1)
        return self.net(h)


def drift(net, x, tau, eps):
    """Shared execution call.  tau, eps: scalars or (n,) tensors."""
    n = x.shape[0]
    tau = torch.as_tensor(tau, device=x.device, dtype=x.dtype).expand(n) if not torch.is_tensor(tau) or tau.dim() == 0 \
        else tau
    eps = torch.as_tensor(eps, device=x.device, dtype=x.dtype).expand(n) if not torch.is_tensor(eps) or eps.dim() == 0 \
        else eps
    return net(x, tau, eps.log())


def sample_log_eps(n, device, gen=None):
    return LOG_EPS_MIN + (LOG_EPS_MAX - LOG_EPS_MIN) * torch.rand(n, generator=gen, device=device)


def sample_pairs(rho_a, rho_b, wall, n, device, gen=None):
    """Independent coupling with wall-crossing pairs rejected (killed reference)."""
    xs0, xs1 = [], []
    got = 0
    while got < n:
        x0 = rho_a.sample(n, wall, device, gen)
        x1 = rho_b.sample(n, wall, device, gen)
        ok = ~wall.crosses(x0, x1)
        xs0.append(x0[ok]); xs1.append(x1[ok]); got += int(ok.sum())
    return torch.cat(xs0)[:n], torch.cat(xs1)[:n]


def bridge_loss(net, x0, x1, log_eps, wall, forward=True, penalty=10.0, tau_clip=0.02):
    """Brownian-bridge regression.  x0,x1: (n,2); log_eps: (n,)."""
    n = x0.shape[0]
    tau = torch.rand(n, device=x0.device)
    eps = log_eps.exp()
    xt = (1 - tau).unsqueeze(1) * x0 + tau.unsqueeze(1) * x1 \
        + (eps * tau * (1 - tau)).sqrt().unsqueeze(1) * torch.randn_like(x0)
    if forward:
        target = (x1 - xt) / (1 - tau).clamp_min(tau_clip).unsqueeze(1)
    else:
        target = (x0 - xt) / tau.clamp_min(tau_clip).unsqueeze(1)
    pred = net(xt, tau, log_eps)
    mse = ((pred - target) ** 2).sum(1).mean()
    near = wall.near(xt, 0.02)
    # drift component pointing toward the wall plane, penalised near the wall
    toward = pred[:, 0] * torch.sign(0.5 - xt[:, 0])
    pen = (torch.relu(toward) ** 2 * near.float()).mean()
    return mse + penalty * pen, mse.item(), pen.item()


@torch.no_grad()
def simulate(net, x0, eps, wall, forward=True, sigma_d=0.0, gen=None, keep_killed=False):
    """Euler-Maruyama over 100 steps of dtau=DT.  eps: (n,) or scalar.
    Returns endpoints and an alive mask (trajectories that hit the wall are killed)."""
    n = x0.shape[0]
    x = x0.clone()
    eps = torch.as_tensor(eps, device=x.device, dtype=x.dtype).expand(n)
    alive = torch.ones(n, dtype=torch.bool, device=x.device)
    for i in range(STEPS_PER_SKILL):
        tau = torch.full((n,), i * DT if forward else 1 - i * DT, device=x.device)
        u = net(x, tau, eps.log())
        u = u / u.norm(dim=1, keepdim=True).clamp_min(1.0)
        xn = x + u * DT + (eps * DT).sqrt().unsqueeze(1) * torch.randn(n, 2, generator=gen, device=x.device)
        if sigma_d > 0:
            xn = xn + 0.2 * sigma_d * DT ** 0.5 * torch.randn(n, 2, generator=gen, device=x.device)
        xn = xn.clamp(0, 1)
        alive &= ~wall.crosses(x, xn)
        x = xn
    return x, alive


def sinkhorn(a, b, reg=0.01, iters=100):
    """Entropic OT cost between equal-weight point clouds (log-domain)."""
    c = torch.cdist(a, b) ** 2
    n, m = c.shape
    la, lb = torch.full((n,), -math.log(n), device=a.device), torch.full((m,), -math.log(m), device=a.device)
    f, g = torch.zeros(n, device=a.device), torch.zeros(m, device=a.device)
    for _ in range(iters):
        f = -reg * torch.logsumexp((g[None, :] - c) / reg + lb[None, :], 1)
        g = -reg * torch.logsumexp((f[:, None] - c) / reg + la[:, None], 0)
    p = torch.exp((f[:, None] + g[None, :] - c) / reg + la[:, None] + lb[None, :])
    return float((p * c).sum())


def fit(net, x0, x1, log_eps, wall, forward, steps, batch, lr, penalty, gen=None):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = x0.shape[0]
    hist = []
    for s in range(steps):
        idx = torch.randint(n, (batch,), generator=gen, device=x0.device)
        loss, mse, pen = bridge_loss(net, x0[idx], x1[idx], log_eps[idx], wall, forward, penalty)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 100 == 0:
            hist.append(mse)
    return hist


def train_bridge(rho_a, rho_b, wall, seed, device, ipf_iters=5, n_pairs=20000, steps0=4000,
                 steps_ipf=1500, batch=1024, lr=1e-3, penalty=10.0, log=print):
    """DSBM.  Returns forward net and per-iteration convergence records."""
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    fwd, bwd = DriftNet().to(device), DriftNet().to(device)
    x0, x1 = sample_pairs(rho_a, rho_b, wall, n_pairs, device, cpu_gen)
    log_eps = sample_log_eps(n_pairs, device, gen)
    records = []

    def evaluate(it):
        xa = rho_a.sample(500, wall, device, cpu_gen)
        xb = rho_b.sample(500, wall, device, cpu_gen)
        xe, alive = simulate(fwd, xa, 0.01, wall, True, gen=gen)
        rec = dict(iter=it, coupling_cost=float(((x1 - x0) ** 2).sum(1).mean()),
                   sinkhorn=sinkhorn(xe[alive], xb), w2=w2(xe[alive].cpu(), xb.cpu()),
                   killed=float(1 - alive.float().mean()))
        records.append(rec)
        log(f"  ipf {it}: cost={rec['coupling_cost']:.4f} sinkhorn={rec['sinkhorn']:.4f} "
            f"w2={rec['w2']:.4f} killed={rec['killed']:.3f}")

    # iteration 0: independent coupling, forward net
    fit(fwd, x0, x1, log_eps, wall, True, steps0, batch, lr, penalty, gen)
    evaluate(0)
    for it in range(1, ipf_iters + 1):
        # backward fit on coupling induced by forward simulation from rho_a
        xa = rho_a.sample(n_pairs, wall, device, cpu_gen)
        log_eps = sample_log_eps(n_pairs, device, gen)
        xe, alive = simulate(fwd, xa, log_eps.exp(), wall, True, gen=gen)
        x0b, x1b, le_b = xa[alive], xe[alive], log_eps[alive]
        fit(bwd, x0b, x1b, le_b, wall, False, steps_ipf, batch, lr, penalty, gen)
        # forward fit on coupling induced by backward simulation from rho_b
        xb = rho_b.sample(n_pairs, wall, device, cpu_gen)
        log_eps = sample_log_eps(n_pairs, device, gen)
        xs, alive = simulate(bwd, xb, log_eps.exp(), wall, False, gen=gen)
        x0, x1, log_eps = xs[alive], xb[alive], log_eps[alive]
        fit(fwd, x0, x1, log_eps, wall, True, steps_ipf, batch, lr, penalty, gen)
        evaluate(it)
    return fwd, records


@torch.no_grad()
def verify_bridge(net, rho_a, rho_b, wall, device, eps_list=(0.003, 0.01, 0.03), n=500, seed=0):
    """Endpoint distribution vs rho_b with no disturbance.  Returns rows and sample clouds."""
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    rows, clouds = [], {}
    for eps in eps_list:
        xa = rho_a.sample(n, wall, device, cpu_gen)
        xb = rho_b.sample(n, wall, device, cpu_gen)
        xe, alive = simulate(net, xa, eps, wall, True, gen=gen)
        rows.append(dict(eps=eps, w2=w2(xe[alive].cpu(), xb[alive].cpu()), killed=float(1 - alive.float().mean())))
        clouds[eps] = (xe.cpu().numpy(), alive.cpu().numpy(), xb.cpu().numpy())
    return rows, clouds


def save_net(net, path):
    torch.save(net.state_dict(), path)


def load_net(path, device):
    net = DriftNet().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net
