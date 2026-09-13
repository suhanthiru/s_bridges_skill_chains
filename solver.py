"""Bridge-matching drift nets and the DSBM / IPF solver.

Generic over (reference SDE, manifold).  One training run produces four nets per
skill: forward and backward drifts at IPF iteration 0 and after K iterations, so
`bridge_iter0` and `bridge_ipf` come from the same run and the same data.

Forward-backward disagreement.  Both drifts are fitted to reference-bridge
drifts, so near the bridge's support b_fwd ~ vnom - xi/(1-tau) and
b_bwd ~ -vnom - xi/tau, where xi is the deviation from the nominal path.  Their
sum is therefore -xi / (tau (1-tau)): a normalised "how far off the bridge's
support am I" signal, available at run time without knowing the endpoint.  That
is what Phase 3 tests as a replan trigger.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn

import se2 as S
import terrain as TR
import task as TK
from sde import Reference, psd_sqrt
import bridge_fast as BF

TAU_EMB = 4


def tau_emb(tau):
    f = 2.0 ** torch.arange(TAU_EMB, device=tau.device, dtype=tau.dtype) * math.pi
    a = tau.unsqueeze(1) * f
    return torch.cat([a.sin(), a.cos()], 1)


class DriftNet(nn.Module):
    def __init__(self, mf, hidden=256):
        super().__init__()
        self.mf = mf
        d_in = mf.n_feat + TR.N_TFEAT + 2 * TAU_EMB
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3))

    def forward(self, g, tau, goal, fields):
        h = torch.cat([self.mf.feats(g, goal.expand_as(g), tau),
                       TR.terrain_feats(fields, g, self.mf.name != "flat"), tau_emb(tau)], 1)
        return self.net(h)


def make_ref(kind, sigma, fields, slip_scale=1.0, kappa=2.0):
    return Reference(kind, sigma=sigma, kappa=kappa, slip_scale=slip_scale, fields=fields)


def sample_pairs(tk, k, n, ref, mf):
    """Endpoint coupling.  The killed reference rejects pairs whose straight line
    crosses an obstacle; every other reference uses the independent coupling."""
    x0, x1 = tk.sample(k, n), tk.sample(k + 1, n)
    if ref.kind != "killed":
        return x0, x1
    keep0, keep1 = [], []
    got = 0
    while got < n:
        a, b = tk.sample(k, n), tk.sample(k + 1, n)
        ok = ~tk.layout.collides(a[:, :2], b[:, :2])
        keep0.append(a[ok]); keep1.append(b[ok]); got += int(ok.sum())
    return torch.cat(keep0)[:n], torch.cat(keep1)[:n]


def fit(net, x0, x1, ref, mf, goal, fields, steps, batch, lr, backward=False, gen=None):
    """Bridge matching.  The backward net is fitted on the reversed pair, at
    progress s along it, and evaluated at forward time tau = 1 - s."""
    a_all, b_all = (x1, x0) if backward else (x0, x1)
    prep = BF.prepare(ref, a_all, b_all, mf)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = a_all.shape[0]
    last = 0.0
    for it in range(steps):
        i = torch.randint(n, (batch,), generator=gen, device=a_all.device)
        s = torch.rand(batch, generator=gen, device=a_all.device).clamp(1e-3, 1 - 1e-3)
        p = dict(mode=prep["mode"], vnom=prep["vnom"][i], Q10i=prep["Q10i"][i])
        g, target = BF.sample_and_target(ref, a_all[i], p, s, mf, gen)
        tau = (1 - s) if backward else s
        loss = ((net(g, tau, goal, fields) - target) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
    return last


@torch.no_grad()
def simulate(net, g0, ref, mf, goal, fields, layout, n_step=50, backward=False, gen=None, kill=False):
    """Integrate the learned SDE across the skill, returning endpoints and a keep mask."""
    g = g0.clone()
    n = g.shape[0]
    mode = BF.mode_of(ref, mf)
    dt = 1.0 / n_step
    keep = torch.ones(n, dtype=torch.bool, device=g.device)
    for i in range(n_step):
        tau = torch.full((n,), (1 - i * dt) if backward else i * dt, device=g.device)
        b = net(g, tau, goal, fields)
        S = BF.cov_at(ref, g, mf, mode)
        z = torch.randn(n, 3, generator=gen, device=g.device)
        if mode == "full":
            z = torch.einsum("nij,nj->ni", psd_sqrt(S), z)
        elif mode == "diag":
            z = S.sqrt() * z
        else:
            z = S.sqrt()[:, None] * z
        xi = b * dt + math.sqrt(dt) * z
        g_new = mf.retract(g, xi)
        if kill:
            keep &= ~layout.collides(g[:, :2], g_new[:, :2])
        g = g_new
    return g, keep


def train_skill(tk, k, ref, mf, seed, device, K=5, n_pair=8000, steps0=2500, steps_ipf=1000,
                batch=512, lr=1e-3, n_sim=50, log=print, frozen_coupling=False, with_bwd=True):
    """DSBM.  Returns {(iter, direction): net} for iter in {0, K}.

    `frozen_coupling` is the control for the Phase-1 IPF result: it spends the
    identical extra optimisation budget re-fitting on the *original* independent
    coupling instead of on the IPF-refined one.  Any change it produces is
    optimiser churn, not the refined coupling."""
    gen = torch.Generator(device=device).manual_seed(seed)
    goal, fields = tk.means[k + 1].unsqueeze(0), tk.obs_fields
    fwd, bwd = DriftNet(mf).to(device), DriftNet(mf).to(device)
    x0, x1 = sample_pairs(tk, k, n_pair, ref, mf)
    kill = ref.kind == "killed"

    diag = []

    def diagnose(it):
        """Marginal fit and transport cost of the current forward coupling."""
        a = tk.sample(k, 600)
        e, keep = simulate(fwd, a, ref, mf, goal, fields, tk.layout, n_sim, False, gen, kill)
        tgt = tk.sample(k + 1, 600)
        cost = float((mf.logmap(a[keep], e[keep]) ** 2).sum(1).mean()) if int(keep.sum()) > 10 else float("nan")
        diag.append(dict(skill=k, iter=it, kept=float(keep.float().mean()),
                         w2_endpoint=S.w2_se2(e[keep].cpu(), tgt.cpu()) if int(keep.sum()) > 10 else float("nan"),
                         transport_cost=cost))

    l1 = fit(fwd, x0, x1, ref, mf, goal, fields, steps0, batch, lr, False, gen)
    l2 = fit(bwd, x0, x1, ref, mf, goal, fields, steps0, batch, lr, True, gen) if (with_bwd or K > 0) else float("nan")
    out = {(0, "fwd"): {kk: v.detach().clone() for kk, v in fwd.state_dict().items()},
           (0, "bwd"): {kk: v.detach().clone() for kk, v in bwd.state_dict().items()}}
    diagnose(0)
    log(f"    iter0 loss fwd {l1:.4f} bwd {l2:.4f}")
    for it in range(1, K + 1):
        if frozen_coupling:
            fit(bwd, x0, x1, ref, mf, goal, fields, steps_ipf, batch, lr, True, gen)
            fit(fwd, x0, x1, ref, mf, goal, fields, steps_ipf, batch, lr, False, gen)
            diagnose(it)
            continue
        a = tk.sample(k, n_pair)
        e, keep = simulate(fwd, a, ref, mf, goal, fields, tk.layout, n_sim, False, gen, kill)
        fit(bwd, a[keep], e[keep], ref, mf, goal, fields, steps_ipf, batch, lr, True, gen)
        b = tk.sample(k + 1, n_pair)
        s, keep = simulate(bwd, b, ref, mf, goal, fields, tk.layout, n_sim, True, gen, kill)
        lf = fit(fwd, s[keep], b[keep], ref, mf, goal, fields, steps_ipf, batch, lr, False, gen)
        diagnose(it)
        if it == K:
            log(f"    iter{it} loss fwd {lf:.4f} kept {float(keep.float().mean()):.2f}")
    out[(K, "fwd")] = {kk: v.detach().clone() for kk, v in fwd.state_dict().items()}
    out[(K, "bwd")] = {kk: v.detach().clone() for kk, v in bwd.state_dict().items()}
    return out, diag


def train_all(tk, ref_kind, mf, seed, device, sigma=0.05, K=5, log=print, **kw):
    t0 = time.time()
    ref = make_ref(ref_kind, sigma, tk.obs_fields)
    nets, diag = {}, []
    for k in range(TK.N_SKILL):
        nets[k], d = train_skill(tk, k, ref, mf, seed * 17 + k, device, K=K, log=log, **kw)
        diag += d
    log(f"    trained 3 skills in {time.time() - t0:.0f}s")
    return nets, time.time() - t0, diag


class BridgeController:
    """Executes the chained skills; optionally logs forward-backward disagreement."""

    def __init__(self, nets, mf, tk, it, device, with_D=True):
        self.mf, self.tk, self.with_D = mf, tk, with_D
        self.fwd, self.bwd = [], []
        for k in range(TK.N_SKILL):
            f, b = DriftNet(mf).to(device), DriftNet(mf).to(device)
            f.load_state_dict(nets[k][(it, "fwd")]); f.eval(); self.fwd.append(f)
            if with_D:
                b.load_state_dict(nets[k][(it, "bwd")]); b.eval()
            self.bwd.append(b)

    def __call__(self, g, k, tau, step):
        goal, fields = self.tk.means[k + 1].unsqueeze(0), self.tk.obs_fields
        b = self.fwd[k](g, tau, goal, fields)
        extra = None
        if self.with_D:
            bb = self.bwd[k](g, tau, goal, fields)
            D = (b + bb).norm(dim=1)
            extra = torch.stack([D, D * (tau * (1 - tau))], 1)
        # tangent coords -> body-frame command
        F = self.mf.frame(g)
        u = torch.einsum("nij,nj->ni", F.transpose(1, 2), b) if self.mf.name == "flat" else b
        return u, extra
