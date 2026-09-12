"""Seam-suite unit tests (run before training): `python tests_seam.py`.

  1. PD reference following: on a pure integrator (no slip, no noise) the shared PD
     tracker reproduces a known spline to 1e-2
  2. Mahalanobis in the (R^2) tangent metric matches a hand computation
  3. Phase-D shifted marginals land where intended (mean, covariance, rotation no-op)
"""
import math
import sys
import torch
import terrain_env as T
import seam as SM

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def test_pd_following():
    print("1. PD reference following on a known spline")
    d = torch.device("cpu"); n = 64
    F = torch.zeros(1, T.GRID, T.GRID)                               # flat terrain
    lids = torch.zeros(n, dtype=torch.long)
    env = T.TerrainEnv(F, lids, (0.0, 0.0, 0.0), 0.0, 0.0, 0, d, disturb=False)  # pure integrator
    env.reset()
    # spline: cubic bend in y between consecutive route means
    R = T.ROUTE
    tt = torch.arange(T.STEPS_PER_SKILL + 1).float() / T.STEPS_PER_SKILL

    def gen(k, x, l):
        m = x.shape[0]
        X = R[k] + tt[:, None] * (R[k + 1] - R[k]); X = X.clone()
        X[:, 1] += 0.06 * torch.sin(math.pi * tt)
        U = (X[1:] - X[:-1]) / E_DT
        return X.expand(m, -1, -1), U.expand(m, -1, -1)
    ctl = SM.PDRef(gen, d)
    env.x[:] = R[0]                                                   # start exactly on the spline
    worst = 0.0
    for s in range(3 * T.STEPS_PER_SKILL):
        k, t = s // 100, s % 100
        X, _ = gen(k, env.x, lids)
        env.step(ctl.act(env, None))
        worst = max(worst, float((env.x - X[:, t + 1]).norm(dim=1).max()))
    check("max tracking error < 1e-2", worst < 1e-2, f"max err {worst:.2e}")


E_DT = T.DT


def test_mahalanobis():
    print("2. Mahalanobis in the tangent metric")
    x = torch.tensor([[0.40, 0.55]]); mean = torch.tensor([0.37, 0.5])
    cov = torch.diag(torch.tensor([0.05, 0.05])) ** 2
    hand = math.sqrt((0.03 / 0.05) ** 2 + (0.05 / 0.05) ** 2)
    md = float(SM.mahal(x, mean, cov)); md2 = float(SM.md_iso(x, 1))
    check("matches hand computation", abs(md - hand) < 1e-5 and abs(md2 - hand) < 1e-5, f"{md:.5f} vs {hand:.5f}")
    aniso = torch.tensor([[0.05 ** 2, 0.0], [0.0, 0.01 ** 2]])
    md_a = float(SM.mahal(x, mean, aniso)); hand_a = math.sqrt((0.03 / 0.05) ** 2 + (0.05 / 0.01) ** 2)
    check("anisotropic case", abs(md_a - hand_a) < 1e-4, f"{md_a:.4f} vs {hand_a:.4f}")


def test_shifts():
    print("3. Phase-D shifted marginals land where intended")
    d = torch.device("cpu"); g = torch.Generator().manual_seed(0); n = 40000
    R = T.ROUTE
    x, mean, cov = SM.shifted_entry("shift_normal", 2.0, None, 1.0, n, d, g)
    e = float((x.mean(0) - (R[1] + torch.tensor([0.0, 2 * SM.STD]))).abs().max())
    check("2 sigma shift along the corridor normal", e < 2e-3, f"mean err {e:.2e}")
    x, mean, cov = SM.shifted_entry("shift_heading", 3.0, None, 1.0, n, d, g)
    e = float((x.mean(0) - (R[1] + torch.tensor([3 * SM.STD, 0.0]))).abs().max())
    check("3 sigma shift along the heading axis", e < 2e-3, f"mean err {e:.2e}")
    x, mean, cov = SM.shifted_entry("shrink", 0.0, None, 0.25, n, d, g)
    e = float((x.std(0) - SM.STD * 0.5).abs().max())
    check("0.25x covariance -> 0.5x std", e < 1e-3, f"std err {e:.2e}")
    x, mean, cov = SM.shifted_entry("grow", 0.0, None, 4.0, n, d, g)
    e = float((x.std(0) - SM.STD * 2).abs().max())
    check("4x covariance -> 2x std", e < 2e-3, f"std err {e:.2e}")
    _, _, c45 = SM.shifted_entry("rot45", 0.0, 45.0, 1.0, 10, d, g)
    check("rotation of the isotropic nominal cov is a no-op (documented)",
          float((c45 - SM.STD ** 2 * torch.eye(2)).abs().max()) < 1e-8)


if __name__ == "__main__":
    test_pd_following(); test_mahalanobis(); test_shifts()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}"); sys.exit(1)
    print("all seam unit tests passed")
