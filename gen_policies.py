"""Downstream policy classes behind one interface: train(tk, G, U, seed, device, steps) -> policy,
and a controller wrapper for Task.rollout.  All predict an 8-step chunk of body commands from
obs = x ⊕ one-hot k ⊕ tau ⊕ terrain probes (phase2.obs_of) and execute 4 steps.

  diffusion  Phase 2/5's DDPM-50 / DDIM-10 denoiser (beta_max 0.2)
  flow       conditional flow matching on the chunk, 10 Euler steps at test time
  bc         plain MLP regression on the chunk
  bc_gmm     MLP with a 5-component Gaussian-mixture head, NLL, sample the mode-weighted mean
"""
import math
import torch
import torch.nn as nn
import task as TK
import phase2 as P2
from bridge import sinusoidal

CHUNK, EXEC = P2.CHUNK, P2.EXEC
SCALE = torch.tensor([TK.UMAX, TK.UMAX, TK.WMAX])


def _windows(tk, G, U, batch, gen, device):
    n = G.shape[0]
    i = torch.randint(n, (batch,), generator=gen, device=device)
    t0 = torch.randint(300 - CHUNK + 1, (batch,), generator=gen, device=device)
    obs = P2.obs_of(tk, G[i, t0], t0 // TK.T_SKILL, (t0 % TK.T_SKILL).float() / TK.T_SKILL)
    idx = t0[:, None] + torch.arange(CHUNK, device=device)[None]
    a0 = (U[i[:, None], idx] / SCALE.to(device)).clamp(-1, 1).reshape(batch, -1)
    return obs, a0


class FlowPolicy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(CHUNK * 3 + P2.OBS_DIM + 32, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, CHUNK * 3))

    def forward(self, a, obs, t):
        return self.net(torch.cat([a, obs, sinusoidal(t, 16)], 1))

    @torch.no_grad()
    def sample(self, obs, n_step=10):
        n = obs.shape[0]; a = torch.randn(n, CHUNK * 3, device=obs.device)
        for i in range(n_step):
            t = torch.full((n,), i / n_step, device=obs.device)
            a = a + self(a, obs, t) / n_step
        return a.clamp(-1, 1).reshape(n, CHUNK, 3) * SCALE.to(obs.device)


class BCPolicy(nn.Module):
    def __init__(self, hidden=256, gmm=0):
        super().__init__()
        self.gmm = gmm
        out = CHUNK * 3 if gmm == 0 else gmm * (2 * CHUNK * 3 + 1)
        self.net = nn.Sequential(nn.Linear(P2.OBS_DIM, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, out))

    def forward(self, obs):
        return self.net(obs)

    def nll(self, obs, a0):
        h = self(obs); n, D = a0.shape
        K = self.gmm
        logit = h[:, :K]; mu = h[:, K:K + K * D].reshape(n, K, D); ls = h[:, K + K * D:].reshape(n, K, D).clamp(-5, 2)
        lp = -0.5 * (((a0[:, None] - mu) / ls.exp()) ** 2 + 2 * ls + math.log(2 * math.pi)).sum(2)
        return -(torch.logsumexp(torch.log_softmax(logit, 1) + lp, 1)).mean()

    @torch.no_grad()
    def sample(self, obs):
        h = self(obs); n = obs.shape[0]
        if self.gmm == 0:
            a = h
        else:
            K, D = self.gmm, CHUNK * 3
            logit = h[:, :K]; mu = h[:, K:K + K * D].reshape(n, K, D)
            a = mu[torch.arange(n), logit.argmax(1)]
        return a.clamp(-1, 1).reshape(n, CHUNK, 3) * SCALE.to(obs.device)


def train(kind, tk, G, U, seed, device, steps, log=lambda m: None):
    """Same optimiser, batch and step budget for every policy class."""
    if kind == "diffusion":
        return P2.train_diffusion(tk, G, U, seed, device, steps, log=log)
    torch.manual_seed(seed); gen = torch.Generator(device=device).manual_seed(seed)
    pol = (FlowPolicy() if kind == "flow" else BCPolicy(gmm=5 if kind == "bc_gmm" else 0)).to(device)
    opt = torch.optim.Adam(pol.parameters(), lr=3e-4)
    for s in range(steps):
        obs, a0 = _windows(tk, G, U, 1024, gen, device)
        if kind == "flow":
            t = torch.rand(1024, generator=gen, device=device); e = torch.randn_like(a0)
            at = (1 - t)[:, None] * e + t[:, None] * a0
            loss = ((pol(at, obs, t) - (a0 - e)) ** 2).mean()
        elif kind == "bc":
            loss = ((pol(obs) - a0) ** 2).mean()
        else:
            loss = pol.nll(obs, a0)
        opt.zero_grad(); loss.backward(); opt.step()
    log(f"  {kind} loss {loss.item():.4f}")
    pol.eval()
    return pol


class ChunkCtl:
    """Receding-horizon execution of any chunk policy (4 of 8), as P2.DiffCtl."""

    def __init__(self, pol, tk):
        self.pol, self.tk, self.buf, self.ptr = pol, tk, None, EXEC

    def __call__(self, g, k, tau, step):
        if self.buf is None or self.ptr >= EXEC or step % TK.T_SKILL == 0:
            self.buf, self.ptr = self.pol.sample(P2.obs_of(self.tk, g, k, tau)), 0
        u = self.buf[:, self.ptr]; self.ptr += 1
        return u, None


@torch.no_grad()
def evaluate(pol, tk_eval, n=200):
    """Success, collision, handoff Mahalanobis and steps under tk_eval's disturbance."""
    out = tk_eval.rollout(ChunkCtl(pol, tk_eval), n=n)
    m = tk_eval.metrics(out)
    return dict(success=m["success"], collision=m["collision"], handoff1_md=m["handoff1_md"],
                handoff2_md=m["handoff2_md"], w2_goal=m["w2_goal"], energy=m["energy"])
