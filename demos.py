"""Oracle sampling MPC (MPPI-style, knows the true terrain) and demo collection.

The MPC is the demo generator for every learned method and, run with pushes,
the ORACLE-MPC upper reference.  Demos are collected without pushes.
"""
import os
import numpy as np
import torch
from env import DT, STEPS_PER_SKILL, N_SKILLS, clip_u
import terrain_env as T


class MPC:
    """Per step: K sampled action sequences over horizon H, rolled out with the
    mean terrain dynamics (no noise), MPPI-weighted; warm-started nominal plan."""

    def __init__(self, fields, lids, rough, K=200, H=30, noise=0.5, lam=0.002, seed=0):
        self.fields, self.lids, self.K, self.H, self.noise, self.lam = fields, lids, K, H, noise, lam
        self.s, self.theta, _ = T.ROUGH[rough] if isinstance(rough, str) else rough
        self.gen = torch.Generator(device=fields.device).manual_seed(seed)
        self.plan = None

    @torch.no_grad()
    def act(self, x, goal):
        n, d = x.shape[0], x.device
        if self.plan is None or self.plan.shape[0] != n:
            self.plan = torch.zeros(n, self.H, 2, device=d)
        to_goal = goal - x
        base = self.plan.clone()
        base[:, -1] = clip_u(to_goal)
        U = base[:, None] + self.noise * torch.randn(n, self.K, self.H, 2, generator=self.gen, device=d)
        U[:, 0] = base  # keep the warm-started plan as one candidate
        U = U / U.norm(dim=-1, keepdim=True).clamp_min(1.0)
        xs = x[:, None].expand(n, self.K, 2).reshape(-1, 2)
        lid = self.lids.repeat_interleave(self.K)
        cost = torch.zeros(n * self.K, device=d)
        g = goal[:, None].expand(n, self.K, 2).reshape(-1, 2)
        for h in range(self.H):
            u = U[:, :, h].reshape(-1, 2)
            v, _ = T.slip(self.fields, xs, lid, u, self.s, self.theta)
            xs = xs + v * DT
            cost += 0.3 * ((xs - g) ** 2).sum(1) + 0.0005 * (u ** 2).sum(1) * DT
        cost += 10.0 * ((xs - g) ** 2).sum(1)
        cost = cost.reshape(n, self.K)
        wgt = torch.softmax(-(cost - cost.min(1, keepdim=True).values) / self.lam, 1)
        plan = (wgt[:, :, None, None] * U).sum(1)
        self.plan = torch.cat([plan[:, 1:], plan[:, -1:]], 1)
        return plan[:, 0]


@torch.no_grad()
def rollout_skill(fields, lids, rough, k, x0, goals, seed, sigma_p_noise=True):
    """Run the MPC for one skill (100 steps, process noise, no pushes).  Returns states (n,101,2), actions (n,100,2)."""
    n, d = x0.shape[0], x0.device
    env = T.TerrainEnv(fields, lids, rough, 0.0, 0.0, seed, d, disturb=False)
    if not sigma_p_noise:
        env.sigma_p = 0.0
    env.reset(); env.x = x0.clone(); env.k[:] = k
    mpc = MPC(fields, lids, rough, seed=seed)
    S, A = [x0.clone()], []
    for _ in range(STEPS_PER_SKILL):
        u = mpc.act(env.x, goals)
        A.append(u.clone())
        env.step(u)
        S.append(env.x.clone())
    return torch.stack(S, 1), torch.stack(A, 1)


def demo_path(root, layout, k, rough):
    return os.path.join(root, f"demos_layout{layout}_k{k + 1}_{rough}.npz")


def collect(root, fields_all, layouts, rough, n=500, device="cpu", log=print):
    """500 demos per skill per layout: x0 ~ rho_{k-1}, goal ~ rho_k."""
    os.makedirs(root, exist_ok=True)
    for L in layouts:
        lids = torch.full((n,), L, dtype=torch.long, device=device)
        gen = torch.Generator(device=device).manual_seed(500 + L)
        for k in range(N_SKILLS):
            x0 = T.waypoint_sample(k, n, device, gen); goals = T.goal_sample(k + 1, n, device, gen)
            S, A = rollout_skill(fields_all, lids, rough, k, x0, goals, seed=700 + 10 * L + k)
            reach = float((((S[:, -1] - goals) ** 2).sum(1).sqrt() < 2 * T.WP_STD).float().mean())
            np.savez_compressed(demo_path(root, L, k, rough), states=S.cpu().numpy().astype(np.float16),
                                actions=A.cpu().numpy().astype(np.float16), goals=goals.cpu().numpy().astype(np.float16),
                                times=(np.arange(STEPS_PER_SKILL + 1) * DT).astype(np.float32))
            log(f"[demos {rough} L={L} k={k + 1}] reach={reach:.3f}")


def load_demos(root, layouts, rough, k):
    S, A, G, Lid = [], [], [], []
    for L in layouts:
        z = np.load(demo_path(root, L, k, rough))
        S.append(z["states"].astype(np.float32)); A.append(z["actions"].astype(np.float32))
        G.append(z["goals"].astype(np.float32)); Lid.append(np.full(len(z["states"]), L))
    return np.concatenate(S), np.concatenate(A), np.concatenate(G), np.concatenate(Lid)
