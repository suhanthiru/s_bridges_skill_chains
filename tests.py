"""Unit tests that gate everything else.  `python tests.py` must pass before training.

  1. linear-Gaussian covariance steering: IPF must reproduce the closed-form
     Schrodinger bridge between two Gaussians to 1e-3 (fixed point, convergence
     from the independent coupling, and the time-marginal covariances)
  2. SE(2) exp/log round-trip, composition, geodesic endpoints, wrap-around
  3. terrain lookup (bilinear values, spatial patch structure)
  4. W2 on a Gaussian pair with a known answer
  5. reference-bridge reduction: kappa=0 with constant Sigma must reproduce the
     Brownian bridge exactly (drift and covariance)

Test 1 runs IPF on propagated moments rather than samples, so the tolerance
measures the solver's *logic* (half-bridge projections, drift formula, time
discretisation) and not Monte-Carlo or neural approximation error.  That is the
point: it separates "IPF is implemented wrong" from "the net fitted it badly".
"""
import math
import sys
import numpy as np
import torch

import se2 as S
import terrain as TR
import sde as SD

torch.manual_seed(0)
np.random.seed(0)
FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)


# ------------------------------------------------- 1. covariance steering
def imf_step(S0, S1, C, eps, n_t=4000, t_end=1 - 1e-5):
    """One Iterative-Markovian-Fitting step on Gaussian moments.

    Given the coupling Cov(X0,X1)=C, build the reciprocal process (Brownian
    bridges between coupled endpoints), take its Markovian projection
    dX = A_t X dt + sqrt(eps) dW with A_t = (K_t - I)/(1-t),
    K_t = Cov(X1,X_t) S_t^-1, simulate it from N(0,S0) and return the new
    coupling Cov(X0,X1).  The Schrodinger bridge is the fixed point.
    """
    d = len(S0)
    I = np.eye(d)

    def A(t):
        St = (1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + eps * t * (1 - t) * I
        K = ((1 - t) * C.T + t * S1) @ np.linalg.inv(St)
        return (K - I) / (1 - t)

    J = S0.copy()
    ts = np.linspace(0, t_end, n_t + 1)
    for i in range(n_t):
        t, h = ts[i], ts[i + 1] - ts[i]
        k1 = J @ A(t).T
        k2 = (J + h / 2 * k1) @ A(t + h / 2).T
        k3 = (J + h / 2 * k2) @ A(t + h / 2).T
        k4 = (J + h * k3) @ A(t + h).T
        J = J + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return J


def test_covariance_steering():
    print("1. linear-Gaussian covariance steering")
    S0 = np.array([[0.040, 0.012], [0.012, 0.015]])
    S1 = np.array([[0.008, -0.004], [-0.004, 0.030]])
    eps = 0.05
    Cstar = SD.gaussian_sb_coupling(S0, S1, eps)

    # the closed form must itself be a valid coupling (symmetric PSD joint)
    Jt = np.block([[S0, Cstar], [Cstar.T, S1]])
    check("closed form is a valid joint covariance", np.linalg.eigvalsh(Jt).min() > -1e-10,
          f"min eig {np.linalg.eigvalsh(Jt).min():.2e}")

    # (a) fixed point: the SB coupling must be invariant under an IMF step
    C1 = imf_step(S0, S1, Cstar, eps)
    e_fix = np.abs(C1 - Cstar).max()
    check("IMF fixed point at the closed-form coupling (<1e-3)", e_fix < 1e-3, f"max abs err {e_fix:.2e}")

    # (b) convergence from the independent coupling
    C = np.zeros_like(S0)
    errs = []
    for _ in range(12):
        C = imf_step(S0, S1, C, eps)
        errs.append(np.abs(C - Cstar).max())
    check("IMF converges to the closed form from pi_0 = rho_0 x rho_1 (<1e-3)", errs[-1] < 1e-3,
          f"err after 12 iters {errs[-1]:.2e} (iter1 {errs[0]:.2e})")

    # (c) time-marginal covariances
    worst = 0.0
    for t in (0.1, 0.25, 0.5, 0.75, 0.9):
        St_ipf = (1 - t) ** 2 * S0 + t ** 2 * S1 + t * (1 - t) * (C + C.T) + eps * t * (1 - t) * np.eye(2)
        St_cf = SD.gaussian_sb_marginal(S0, S1, Cstar, eps, t)
        worst = max(worst, np.abs(St_ipf - St_cf).max())
    check("marginal covariances Sigma_t match closed form (<1e-3)", worst < 1e-3, f"max abs err {worst:.2e}")

    # (d) eps -> 0 recovers the Bures (Monge) coupling
    C0 = SD.gaussian_sb_coupling(S0, S1, 1e-9)
    R = SD._sqrtm(S0)
    bures = R @ SD._sqrtm(R @ S1 @ R) @ np.linalg.inv(R)
    check("eps -> 0 recovers the deterministic Bures coupling", np.abs(C0 - bures).max() < 1e-5,
          f"max abs err {np.abs(C0 - bures).max():.2e}")


# ------------------------------------------------------------- 2. SE(2)
def test_se2():
    print("2. SE(2) group operations")
    n = 4000
    xi = torch.randn(n, 3) * torch.tensor([0.5, 0.5, 1.5])
    xi[:, 2] = S.wrap(xi[:, 2]).clamp(-math.pi + 1e-6, math.pi - 1e-6)  # exp is injective only here
    xi[:1000, 2] *= 1e-5                                    # near-identity rotation
    xi[1000:2000, 2] = math.pi - 1e-4                       # near the wrap
    e = (S.log_se2(S.exp_se2(xi)) - xi).abs().max().item()
    check("exp/log round-trip on |phi| < pi (incl. phi~0 and phi~pi)", e < 1e-4, f"max abs err {e:.2e}")

    g = S.exp_se2(torch.randn(n, 3))
    h = S.exp_se2(torch.randn(n, 3))
    e = (S.compose(g, S.inv(g)) - torch.tensor([0.0, 0.0, 0.0])).abs().max().item()
    check("g . g^-1 = identity", e < 1e-5, f"max abs err {e:.2e}")

    e = (S.compose(g, S.exp_se2(S.between(g, h))) - h).abs().max().item()
    check("g . exp(log(g^-1 h)) = h", e < 1e-5, f"max abs err {e:.2e}")

    tau0, tau1 = torch.zeros(n), torch.ones(n)
    e0 = (S.SE2.interp(g, h, tau0) - g).abs().max().item()
    e1 = (S.SE2.interp(g, h, tau1) - h).abs().max().item()
    check("geodesic interpolation hits both endpoints", max(e0, e1) < 1e-5, f"max abs err {max(e0, e1):.2e}")

    # left-equivariance of the geodesic: interp(a.g, a.h) = a.interp(g, h)
    a = S.exp_se2(torch.randn(1, 3)).expand(n, 3).contiguous()
    lhs = S.SE2.interp(S.compose(a, g), S.compose(a, h), torch.full((n,), 0.37))
    rhs = S.compose(a, S.SE2.interp(g, h, torch.full((n,), 0.37)))
    e = (lhs - rhs).abs().max().item()
    check("geodesic is left-equivariant", e < 1e-5, f"max abs err {e:.2e}")

    # heading wrap: a marginal at theta = pi must not blow up the SE(2) metric
    m = torch.tensor([[0.5, 0.5, math.pi]])
    gs = torch.cat([m.repeat(2, 1)], 0); gs[0, 2] = math.pi - 0.05; gs[1, 2] = -math.pi + 0.05
    cov = torch.diag(torch.tensor([0.05, 0.05, 0.1])) ** 2
    md = S.mahalanobis(gs, m, cov)
    check("Mahalanobis is finite across the theta = pi cut", float(md.max()) < 1.5,
          f"max {float(md.max()):.3f} (flat would give ~{2 * math.pi / 0.1:.0f})")


# ----------------------------------------------------------- 3. terrain
def test_terrain():
    print("3. terrain map and lookup")
    cm = TR.class_map(0)
    F = torch.tensor(TR.property_fields(cm))
    g = (torch.arange(TR.GRID) + 0.5) / TR.GRID
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    pts = torch.stack([xx.reshape(-1), yy.reshape(-1)], 1)
    got = TR.lookup(F, pts)
    want = F.reshape(5, -1).T
    e = (got - want).abs().max().item()
    check("bilinear lookup is exact at cell centres", e < 1e-5, f"max abs err {e:.2e}")

    same = (cm[:, :-1] == cm[:, 1:]).mean()
    check("classes form spatial patches, not iid cells", same > 0.9,
          f"horizontal neighbour agreement {same:.3f} (iid would be ~{1 / len(TR.NAMES):.2f})")

    cov = TR.slip_cov_body(F, pts[:64])
    check("slip covariance is PD and anisotropic", float(torch.linalg.eigvalsh(cov).min()) > 0
          and float((cov[:, 1, 1] / cov[:, 0, 0]).max()) > 3,
          f"max lat/long ratio {float((cov[:, 1, 1] / cov[:, 0, 0]).max()):.1f}")

    rain = torch.tensor(TR.property_fields(cm, rain=True))
    drop = (rain[3] / F[3]).min().item()
    check("rain drops wet-class friction by 40%", abs(drop - 0.6) < 1e-5, f"min friction ratio {drop:.3f}")


# ---------------------------------------------------------------- 4. W2
def test_w2():
    print("4. W2 on SE(2)")
    n, s, d = 400, 0.05, 0.50
    gen = torch.Generator().manual_seed(0)
    a = torch.zeros(n, 3); a[:, :2] = torch.randn(n, 2, generator=gen) * s
    b = a.clone(); b[:, 0] += d
    w = S.w2_se2(a, b)
    check("W2 of a pure translation equals the shift", abs(w - d) < 1e-4, f"W2 {w:.5f} vs {d}")

    b2 = torch.zeros(n, 3); b2[:, :2] = torch.randn(n, 2, generator=gen) * s; b2[:, 0] += d
    w2 = S.w2_se2(a, b2)
    # the assignment is a minimum over a finite sample, so it may sit just below d
    check("W2 of equal-covariance Gaussians ~ mean shift", abs(w2 - d) < 0.05 * d, f"W2 {w2:.4f} vs {d}")

    w0 = S.w2_se2(a, a.clone())
    check("W2 of a cloud with itself is 0", w0 < 1e-6, f"W2 {w0:.2e}")

    # heading enters through the tangent metric with weight HEADING_W
    c = a.clone(); c[:, 2] += 0.4
    check("heading difference is metrised", abs(S.w2_se2(a, c) - S.HEADING_W * 0.4) < 1e-4,
          f"W2 {S.w2_se2(a, c):.5f} vs {S.HEADING_W * 0.4:.5f}")


# --------------------------------------------- 5. reference-bridge sanity
def test_reference_bridge():
    print("5. reference bridge reduces to the Brownian bridge")
    n = 20000
    ref = SD.Reference("brownian", sigma=0.08)
    mf = S.Flat
    gen = torch.Generator().manual_seed(0)
    g0 = torch.randn(n, 3, generator=gen) * 0.1
    g1 = torch.randn(n, 3, generator=gen) * 0.1 + torch.tensor([0.6, 0.0, 0.0])
    for tau_v in (0.15, 0.5, 0.85):
        tau = torch.full((n,), tau_v)
        g = ref.bridge_sample(g0, g1, tau, mf, gen)
        emp = (g - mf.interp(g0, g1, tau)).var(0).mean().item()
        want = ref.sigma ** 2 * tau_v * (1 - tau_v)
        check(f"bridge covariance at tau={tau_v} is sigma^2 tau(1-tau)", abs(emp / want - 1) < 0.05,
              f"empirical {emp:.5f} vs {want:.5f}")
        b = ref.bridge_drift(g, g0, g1, tau, mf)
        exact = mf.logmap(g, g1) / (1 - tau_v)
        e = (b - exact).abs().max().item()
        check(f"bridge drift at tau={tau_v} equals (x1-x)/(1-tau)", e < 1e-3, f"max abs err {e:.2e}")

    # an OU reference must pull the bridge mean off the straight line
    ou = SD.Reference("unicycle", sigma=0.08, kappa=4.0)
    tau = torch.full((n,), 0.5)
    dev_b = (ref.bridge_sample(g0, g1, tau, mf, gen) - mf.interp(g0, g1, tau)).var(0).mean().item()
    dev_o = (ou.bridge_sample(g0, g1, tau, mf, gen) - mf.interp(g0, g1, tau)).var(0).mean().item()
    check("OU reference concentrates the bridge relative to Brownian", dev_o < dev_b * 0.9,
          f"OU var {dev_o:.5f} < Brownian var {dev_b:.5f}")


if __name__ == "__main__":
    test_covariance_steering(); test_se2(); test_terrain(); test_w2(); test_reference_bridge()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}")
        sys.exit(1)
    print("all unit tests passed")
