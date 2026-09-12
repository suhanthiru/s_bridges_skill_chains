"""The chained-skill task: three SE(2) skills, terrain, obstacles, disturbances.

The *physics* is always SE(2) (body-frame velocity integrated on the group) no
matter which manifold a method models with, and every evaluation metric uses the
SE(2)-correct distance.  Only the model's internal geometry changes between the
`flat` and `se2` conditions, so the comparison is fair.
"""
import math
import numpy as np
import torch
import se2 as S
import terrain as TR

DT = 0.01
T_SKILL = 100
N_SKILL = 3
UMAX, WMAX = 1.0, 3.0
SIGMA_BASE = 0.02          # body-frame process noise present in every condition
DISTURBANCES = ("none", "slip", "rain", "push")


def clip_u(u):
    """Clip body-frame twist: translation norm <= UMAX, |omega| <= WMAX."""
    v = u[:, :2]
    v = v / (v.norm(dim=1, keepdim=True) / UMAX).clamp_min(1.0)
    return torch.cat([v, u[:, 2:3].clamp(-WMAX, WMAX)], 1)


class PushField:
    """Smooth, unmodelled velocity field (not represented in the terrain map)."""

    def __init__(self, layout_id, mag, device):
        rng = np.random.RandomState(4242 + layout_id)
        self.k = torch.tensor(rng.uniform(2, 5, (3, 2)), dtype=torch.float32, device=device)
        self.p = torch.tensor(rng.uniform(0, 2 * math.pi, (3, 2)), dtype=torch.float32, device=device)
        self.mag = mag

    def __call__(self, xy):
        a = xy @ self.k.T
        return self.mag * torch.stack([torch.sin(a + self.p[:, 0]).mean(1),
                                       torch.cos(a + self.p[:, 1]).mean(1)], 1)


class Task:
    """One (layout, disturbance) instance, batched over n robots."""

    def __init__(self, layout, disturbance, n, seed, device, route="lower", gap_lat=0.040,
                 slip_scale=0.7, push_mag=0.25, map_noise=False, layout_id=0):
        self.layout, self.dist, self.n, self.device = layout, disturbance, n, device
        kw = dict(flip_rate=0.10, boundary_jitter=0.02) if map_noise else {}
        self.cmap = TR.class_map(layout_id)
        self.true_fields = torch.tensor(TR.property_fields(self.cmap), device=device)
        self.rain_fields = torch.tensor(TR.property_fields(self.cmap, rain=True), device=device)
        # the map a method is allowed to see (identical to the truth unless map_noise)
        self.obs_fields = torch.tensor(TR.property_fields(TR.class_map(layout_id, **kw)), device=device) \
            if map_noise else self.true_fields
        self.means, self.covs = TR.marginals(layout, route, gap_lat)
        self.means = [m.to(device) for m in self.means]
        self.covs = [c.to(device) for c in self.covs]
        self.slip_scale = slip_scale
        self.push = PushField(layout_id, push_mag, device) if disturbance == "push" else None
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.rain_t = int(torch.randint(60, 240, (1,), generator=self.gen, device=device)) \
            if disturbance == "rain" else 10 ** 9

    def fields_at(self, step):
        return self.rain_fields if step >= self.rain_t else self.true_fields

    def sample(self, k, n):
        return S.sample_marginal(self.means[k], self.covs[k], n, self.device, self.gen)

    def dynamics(self, g, u, step):
        """One physics step.  Returns new pose and a collision flag."""
        n = g.shape[0]
        u = clip_u(u)
        if self.dist == "none":
            mu = torch.ones(n, device=self.device)
            Sig = (SIGMA_BASE ** 2) * torch.eye(3, device=self.device).expand(n, 3, 3)
            rate = torch.zeros(n, device=self.device)
        else:
            p = TR.lookup(self.fields_at(step), g[:, :2])
            mu, rate = p[:, 3], p[:, 4]
            Sig = torch.diag_embed((p[:, :3] * self.slip_scale) ** 2 + SIGMA_BASE ** 2)
        xi = mu.unsqueeze(1) * u * DT
        L = torch.linalg.cholesky(Sig)
        xi = xi + math.sqrt(DT) * torch.einsum("nij,nj->ni", L, torch.randn(n, 3, generator=self.gen, device=self.device))
        if self.push is not None:
            f = self.push(g[:, :2])                                   # world-frame, rotate into body
            R = S.rot(-g[:, 2])
            xi[:, :2] = xi[:, :2] + torch.einsum("nij,nj->ni", R, f) * DT
            kick = (torch.rand(n, generator=self.gen, device=self.device) < rate * 3).float()
            xi = xi + kick.unsqueeze(1) * 0.05 * torch.randn(n, 3, generator=self.gen, device=self.device)
        elif self.dist != "none":
            kick = (torch.rand(n, generator=self.gen, device=self.device) < rate).float()
            xi = xi + kick.unsqueeze(1) * 0.03 * torch.randn(n, 3, generator=self.gen, device=self.device)
        g_new = S.compose(g, S.exp_se2(xi))
        hit = self.layout.collides(g[:, :2], g_new[:, :2])
        g_new = torch.where(hit.unsqueeze(1), g, g_new)
        return g_new, hit

    @torch.no_grad()
    def rollout(self, controller, n=None, record=False):
        """Chain three skills.  `controller(g, k, tau, step)` returns a body-frame twist."""
        n = n or self.n
        g = self.sample(0, n)
        alive = torch.ones(n, dtype=torch.bool, device=self.device)
        energy = torch.zeros(n, device=self.device)
        handoff = []
        traj = [g.clone()] if record else None
        dlog = []
        for k in range(N_SKILL):
            for t in range(T_SKILL):
                tau = torch.full((n,), t / T_SKILL, device=self.device)
                u, extra = controller(g, k, tau, k * T_SKILL + t)
                energy = energy + (u ** 2).sum(1) * DT * alive.float()
                g_new, hit = self.dynamics(g, u, k * T_SKILL + t)
                g = torch.where(alive.unsqueeze(1), g_new, g)
                alive = alive & ~hit
                if record:
                    traj.append(g.clone())
                if extra is not None:
                    dlog.append(extra)
            handoff.append(g.clone())
        goal_md = S.mahalanobis(g, self.means[3], self.covs[3])
        succ = alive & (goal_md <= 2.0)
        out = dict(success=succ, alive=alive, energy=energy, goal_md=goal_md, final=g, handoff=handoff)
        if record:
            out["traj"] = torch.stack(traj, 1)
        if dlog:
            out["D"] = torch.stack(dlog, 1)
        return out

    def metrics(self, out, n_ref=400):
        """Success, collision, energy, handoff Mahalanobis, terminal W2 to rho_3."""
        m = dict(success=float(out["success"].float().mean()),
                 collision=float(1 - out["alive"].float().mean()),
                 energy=float(out["energy"].mean()),
                 goal_md=float(out["goal_md"].median()))
        for j in (1, 2):
            md = S.mahalanobis(out["handoff"][j - 1], self.means[j], self.covs[j])
            m[f"handoff{j}_md"] = float(md.median())
        ref = S.sample_marginal(self.means[3], self.covs[3], n_ref, self.device, self.gen)
        m["w2_goal"] = S.w2_se2(out["final"].cpu(), ref.cpu())
        alive = out["alive"]
        m["w2_goal_alive"] = S.w2_se2(out["final"][alive].cpu(), ref.cpu()) if int(alive.sum()) > 10 else float("nan")
        return m
