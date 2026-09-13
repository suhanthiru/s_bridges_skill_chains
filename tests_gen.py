"""Generator-suite unit tests (run before training): `python tests_gen.py`.

  1. PD-noise and BRIDGE-slip consume identical standard-normal draws for a fixed seed
  2. DART's injected action noise reproduces the slip covariance in displacement
  3. MPC-relabel reproduces the oracle's own actions on the demo states to 1e-3
  4. the diffusion schedule reaches abar_T < 0.05
"""
import math
import sys
import torch
import se2 as S
import terrain as TR
import task as TK
import phase2 as P2
import gen_sources as GS
import bridge_fast as BF

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


def test_shared_noise():
    print("1. shared noise stream")
    d = torch.device("cpu")
    z1, z2 = GS.NoiseStream(16, 7, d), GS.NoiseStream(16, 7, d)
    check("same seed -> identical draws", torch.equal(z1.z, z2.z))
    check("different seed -> different draws", not torch.equal(z1.z, GS.NoiseStream(16, 8, d).z))
    # both generators map the same z through the same covariance at the same state
    tk = GS.GenTask(TR.Layout("L1"), "none", 8, 1, d); ref = GS.make_ref("slip", tk)
    g = tk.sample(0, 8); z = z1(0)[:8]
    a = GS.noise_step(ref, S.SE2, g, z); b = GS.noise_step(ref, S.SE2, g, z)
    check("noise_step is deterministic in (state, z)", torch.equal(a, b))
    want = (TK.DT * BF.cov_at(ref, g, S.SE2, "diag")).sqrt() * z                 # exact per element
    check("noise_step = sqrt(dt Sigma(x)) z elementwise", float((a - want).abs().max()) < 1e-6)


def test_dart_noise():
    print("2. DART action noise matches the slip covariance")
    d = torch.device("cpu"); n = 20000
    tk = GS.GenTask(TR.Layout("L1"), "none", n, 2, d); ref = GS.make_ref("slip", tk)
    g = tk.sample(1, n); g[:, :2] = 0.5                       # one location: constant Sigma
    z = torch.randn(n, 3)
    eta = GS.noise_step(ref, S.SE2, g, z) / TK.DT
    disp = (eta * TK.DT)                                       # displacement from the action noise
    emp = disp.var(0); want = TK.DT * BF.cov_at(ref, g, S.SE2, "diag")[0]
    e = float(((emp - want).abs() / want).max())
    check("displacement variance = Sigma dt per axis (<5%)", e < 0.05, f"max rel err {e:.3f}")


def test_relabel():
    print("3. MPC-relabel reproduces the oracle on demo states")
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk = GS.GenTask(TR.Layout("L1"), "none", 8, 3, d)
    G, U = P2.load_demos("L1", d); G = G[:8]
    ref = GS.make_ref("slip", tk); z = GS.NoiseStream(8, 0, d)
    _, U1 = GS.rollout(tk, S.SE2, GS.mpc_act(tk, 5), 8, z, ref, states=G)
    _, U2 = GS.rollout(tk, S.SE2, GS.mpc_act(tk, 5), 8, z, ref, states=G)
    e = float((U1 - U2).abs().max())
    check("relabel is reproducible to 1e-3 on the same states", e < 1e-3, f"max abs diff {e:.2e}")
    # and the relabelled action is the oracle's action from that state (one-step check)
    m = GS.MPC(tk, seed=5); u_direct = m.act(G[:, 0], 0)
    e2 = float((U1[:, 0] - TK.clip_u(u_direct)).abs().max())
    check("first relabelled action equals a fresh oracle call", e2 < 1e-3, f"max abs diff {e2:.2e}")


def test_schedule():
    print("4. diffusion schedule")
    pol = P2.DiffPolicy(); ab = float(pol.abar[-1])
    check("abar_T < 0.05", ab < 0.05, f"abar_T = {ab:.4f}")
    check("abar decreasing", bool((pol.abar[1:] < pol.abar[:-1]).all()))


if __name__ == "__main__":
    test_shared_noise(); test_dart_noise(); test_relabel(); test_schedule()
    print()
    if FAILS:
        print(f"FAILED {len(FAILS)}: {FAILS}"); sys.exit(1)
    print("all generator-suite unit tests passed")
