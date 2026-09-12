"""Run the whole experiment: bridges -> baseline nets -> PPO -> eval -> plots.

    python run_all.py            # everything (~30-40 min on one GPU)
    python run_all.py --quick    # smoke test with tiny budgets (~3 min)
    python run_all.py --stage eval --stage plots
    python run_all.py --tune     # small hyper-parameter check at w=0.20, sigma_d=0.02 only
"""
import argparse
import json
import multiprocessing as mp
import os
import time
import numpy as np
import pandas as pd
import torch

import env as E
import bridge as B
import rl as R
import baselines as BL

RES, MODELS, FIGS = "results", os.path.join("results", "models"), "figures"
WIDTHS = (0.10, 0.20, 0.40)
SIGMAS = R.SIGMA_D_SWEEP
SEEDS = (0, 1, 2, 3, 4)
FIXED_EPS = (0.001, 0.003, 0.01, 0.03, 0.1)
ORACLE_EPS = (0.03, 0.003, 0.03)
VERIFY_EPS = (0.003, 0.01, 0.03)

# Frozen after a small check at the tuning cell (w=0.20, sigma_d=0.02); see results/tuning.csv.
CONFIG = dict(
    bridge=dict(ipf_iters=5, n_pairs=20000, steps0=4000, steps_ipf=1500, batch=1024, lr=1e-3, penalty=10.0),
    ppo=dict(lr=3e-4, rollout=32, epochs=4, minibatch=4096, gamma=0.99, lam=0.95, clip=0.2),
    ppo_steps=2_000_000, ppo_envs=512, episodes=200, verify_n=500,
)
QUICK = dict(
    bridge=dict(ipf_iters=1, n_pairs=4000, steps0=300, steps_ipf=100, batch=512, lr=1e-3, penalty=10.0),
    ppo=CONFIG["ppo"], ppo_steps=40_000, ppo_envs=256, episodes=50, verify_n=200,
)


def dev():
    return E.get_device(os.environ.get("SB_DEVICE") or None)


def net_path(kind, w, seed, k):
    return os.path.join(MODELS, f"{kind}_w{w:.2f}_s{seed}_k{k}.pt")


def pol_path(cond, w, seed):
    return os.path.join(MODELS, f"policy_{cond}_w{w:.2f}_s{seed}.pt")


# ----------------------------------------------------------------- workers
def job_bridges(args):
    w, seed, cfg, kind = args
    d = dev(); E.seed_all(seed)
    wall, rho = E.Wall(w), E.regions(w)
    ipf_rows, ver_rows, clouds = [], [], {}
    bcfg = dict(cfg["bridge"])
    if kind == "flow0":
        bcfg["ipf_iters"] = 0
    for k in range(3):
        t0 = time.time()
        log = (lambda m: print(f"[{kind} w={w} s={seed} k={k + 1}] {m}", flush=True))
        if kind == "fm":
            net = BL.train_flow(rho[k], rho[k + 1], wall, seed, d, steps=bcfg["steps0"], penalty=bcfg["penalty"])
            recs = []
        else:
            net, recs = B.train_bridge(rho[k], rho[k + 1], wall, seed, d, log=log, **bcfg)
        B.save_net(net, net_path(kind, w, seed, k))
        ipf_rows += [dict(kind=kind, w=w, seed=seed, skill=k + 1, **r) for r in recs]
        rows, cl = B.verify_bridge(net, rho[k], rho[k + 1], wall, d, VERIFY_EPS, cfg["verify_n"], seed)
        ver_rows += [dict(kind=kind, w=w, seed=seed, skill=k + 1, **r) for r in rows]
        for eps, v in cl.items():
            clouds[f"{kind}_w{w:.2f}_s{seed}_k{k + 1}_eps{eps}"] = v
        log(f"done in {time.time() - t0:.0f}s")
    return ipf_rows, ver_rows, clouds


def job_ppo(args):
    cond, w, seed, cfg = args
    d = dev(); E.seed_all(seed)
    log = (lambda m: print(f"[ppo {cond} w={w} s={seed}] {m}", flush=True))
    if cond == "E":
        envr = BL.PlainEnv(w, cfg["ppo_envs"], seed, d)
        pol, crit = BL.GaussianPolicy().to(d), BL.PlainCritic().to(d)
    else:
        kind = "bridge" if cond == "C" else ("flow0" if cond == "D" else "fm")
        nets = [B.load_net(net_path(kind, w, seed, k), d) for k in range(3)]
        envr = R.BridgeEnv(nets, w, cfg["ppo_envs"], seed, d)
        pol, crit = R.EpsPolicy().to(d), R.Critic().to(d)
    t0 = time.time()
    curve = R.ppo_train(envr, pol, crit, cfg["ppo_steps"], seed, d, log=log, **cfg["ppo"])
    torch.save(pol.state_dict(), pol_path(cond, w, seed))
    log(f"done in {time.time() - t0:.0f}s")
    hm = R.eps_heatmap(pol, d) if cond != "E" else None
    return [dict(cond=cond, w=w, seed=seed, **c) for c in curve], (cond, w, seed, hm)


@torch.no_grad()
def run_episodes(cond, w, sigma_d, seed, cfg, d, cache):
    """200 episodes in one batch.  Returns per-episode rows and handoff W2 rows."""
    n = cfg["episodes"]
    envb = E.BatchEnv(w, sigma_d, n, 10_000 + seed, d)
    rho = envb.rho
    if cond == "E":
        pol = cache.setdefault(("E", w, seed), _load_pol(BL.GaussianPolicy, pol_path("E", w, seed), d))
        stepper = lambda obs, k, t: (torch.tanh(pol.mu(pol.features(obs))), torch.zeros(n, device=d))
    else:
        if cond.startswith("A"):
            kind, eps_fn = "bridge", (lambda obs, k: torch.full((n,), float(cond[1:]), device=d))
        elif cond == "B":
            kind, eps_fn = "bridge", (lambda obs, k: torch.tensor(ORACLE_EPS, device=d)[k])
        else:
            kind = {"C": "bridge", "D": "flow0", "Dprime": "fm"}[cond]
            pol = cache.setdefault((cond, w, seed), _load_pol(R.EpsPolicy, pol_path(cond, w, seed), d))
            eps_fn = lambda obs, k: pol.mean_log_eps(obs).exp()
        nets = cache.setdefault((kind, w, seed), [B.load_net(net_path(kind, w, seed, k), d) for k in range(3)])

        def stepper(obs, k, t):
            eps = eps_fn(obs, k)
            u = torch.zeros(n, 2, device=d)
            for j, net in enumerate(nets):
                sel = k == j
                if sel.any():
                    u[sel] = net(obs[sel, :2], t[sel].float() * E.DT, eps[sel].log())
            return u, eps
    obs = envb.reset()
    done = torch.zeros(n, dtype=torch.bool, device=d)
    succ, coll = torch.zeros(n, dtype=torch.bool, device=d), torch.zeros(n, dtype=torch.bool, device=d)
    effort, length = torch.zeros(n, device=d), torch.zeros(n, dtype=torch.long, device=d)
    for _ in range(E.STEPS_PER_SKILL * E.N_SKILLS):
        u, eps = stepper(obs, envb.k, envb.t)
        obs, r, term, _, info = envb.step(u, eps)
        new = term & ~done
        succ |= info["success"] & new; coll |= info["collision"] & new
        effort = torch.where(new, info["effort"], effort); length = torch.where(new, info["length"], length)
        done |= term
        if done.all():
            break
    rows = [dict(condition=cond, w=w, sigma_d=sigma_d, seed=seed, episode=i, success=int(succ[i]),
                 collision=int(coll[i]), effort=float(effort[i]), length=int(length[i])) for i in range(n)]
    hrows = []
    for j in range(2):
        h = envb.handoff[j]; ok = ~torch.isnan(h[:, 0])
        m = int(ok.sum())
        w2 = E.w2(h[ok].cpu(), rho[j + 1].sample(m, envb.wall, "cpu").numpy()) if m >= 5 else float("nan")
        hrows.append(dict(condition=cond, w=w, sigma_d=sigma_d, seed=seed, handoff=j + 1, n_reached=m, w2=w2))
    return rows, hrows


def _load_pol(cls, path, d):
    p = cls().to(d); p.load_state_dict(torch.load(path, map_location=d)); p.eval(); return p


# ------------------------------------------------------------------ stages
def pool_map(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as p:
        return p.map(fn, jobs, chunksize=1)


def stage_nets(kind, cfg, seeds, workers):
    out = pool_map(job_bridges, [(w, s, cfg, kind) for w in WIDTHS for s in seeds], workers)
    ipf = sum((o[0] for o in out), []); ver = sum((o[1] for o in out), [])
    clouds = {k: v for o in out for k, v in o[2].items()}
    mode = "a" if kind != "bridge" and os.path.exists(os.path.join(RES, "ipf.csv")) else "w"
    pd.DataFrame(ipf).to_csv(os.path.join(RES, "ipf.csv"), mode=mode, header=mode == "w", index=False)
    pd.DataFrame(ver).to_csv(os.path.join(RES, "verify.csv"), mode=mode, header=mode == "w", index=False)
    np.savez_compressed(os.path.join(RES, f"verify_samples_{kind}.npz"),
                        **{k + "_end": v[0] for k, v in clouds.items()},
                        **{k + "_alive": v[1] for k, v in clouds.items()},
                        **{k + "_target": v[2] for k, v in clouds.items()})


def stage_ppo(cond, cfg, seeds, workers):
    out = pool_map(job_ppo, [(cond, w, s, cfg) for w in WIDTHS for s in seeds], workers)
    curves = sum((o[0] for o in out), [])
    path = os.path.join(RES, "ppo_curves.csv")
    first = not os.path.exists(path) or cond == "C"
    pd.DataFrame(curves).to_csv(path, mode="w" if first else "a", header=first, index=False)
    if cond != "E":
        hp = os.path.join(RES, "heatmaps.npz")
        hm = dict(np.load(hp)) if os.path.exists(hp) and cond != "C" else {}
        for c, w, s, h in (o[1] for o in out):
            hm[f"{c}_w{w:.2f}_s{s}"] = h
        np.savez_compressed(hp, **hm)


def stage_eval(cfg, seeds, conds):
    d = dev(); cache = {}
    rows, hrows = [], []
    for cond in conds:
        t0 = time.time()
        for w in WIDTHS:
            for sd in SIGMAS:
                for s in seeds:
                    r, h = run_episodes(cond, w, sd, s, cfg, d, cache)
                    rows += r; hrows += h
        df = pd.DataFrame([x for x in rows if x["condition"] == cond])
        print(f"[eval {cond}] success={df.success.mean():.3f} collision={df.collision.mean():.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    ep = pd.DataFrame(rows); ep.to_csv(os.path.join(RES, "episodes.csv"), index=False)
    pd.DataFrame(hrows).to_csv(os.path.join(RES, "handoff.csv"), index=False)
    per_seed = ep.groupby(["condition", "w", "sigma_d", "seed"])[["success", "collision", "effort", "length"]].mean()
    summ = per_seed.groupby(["condition", "w", "sigma_d"]).agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(os.path.join(RES, "summary.csv"), index=False)


def stage_tune(cfg, workers):
    """Hyper-parameter check at the tuning cell only (w=0.20, sigma_d=0.02, seed 0)."""
    d = dev(); rows = []
    w, sd, seed = 0.20, 0.02, 0
    for pen in (1.0, 10.0):
        bcfg = dict(cfg["bridge"], penalty=pen)
        nets = [B.train_bridge(E.regions(w)[k], E.regions(w)[k + 1], E.Wall(w), seed, d, log=lambda m: None, **bcfg)[0]
                for k in range(3)]
        for lr in (1e-4, 3e-4):
            E.seed_all(seed)
            envr = R.BridgeEnv(nets, w, cfg["ppo_envs"], seed, d)
            pol, crit = R.EpsPolicy().to(d), R.Critic().to(d)
            curve = R.ppo_train(envr, pol, crit, cfg["ppo_steps"] // 4, seed, d, log=lambda m: None,
                                **dict(cfg["ppo"], lr=lr))
            # evaluate at the tuning cell only
            envb = E.BatchEnv(w, sd, 200, 10_000, d); obs = envb.reset()
            done = torch.zeros(200, dtype=torch.bool, device=d); succ = done.clone()
            with torch.no_grad():
                for _ in range(300):
                    eps = pol.mean_log_eps(obs).exp(); u = torch.zeros(200, 2, device=d)
                    for k, net in enumerate(nets):
                        sel = envb.k == k
                        if sel.any():
                            u[sel] = net(obs[sel, :2], envb.t[sel].float() * E.DT, eps[sel].log())
                    obs, r, term, _, info = envb.step(u, eps)
                    succ |= info["success"] & term & ~done; done |= term
            rows.append(dict(penalty=pen, lr=lr, w=w, sigma_d=sd, success=float(succ.float().mean()),
                             train_ret=curve[-1]["ret"]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuning.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--stage", action="append", help="bridges|flows|ppo_c|ppo_d|ppo_e|eval|plots")
    ap.add_argument("--with-dprime", action="store_true")
    ap.add_argument("--workers", type=int, default=3 if torch.cuda.is_available() else 5)
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True); os.makedirs(FIGS, exist_ok=True)
    cfg = QUICK if a.quick else CONFIG
    seeds = SEEDS[:2] if a.quick else SEEDS
    stages = a.stage or ["bridges", "flows", "ppo_c", "ppo_d", "ppo_e", "eval", "plots"]
    json.dump(dict(seeds=list(seeds), quick=a.quick, config=cfg, torch=torch.__version__,
                   device=str(dev())), open(os.path.join(RES, "seeds.json"), "w"), indent=1)
    timing = {}
    if a.tune:
        stages = ["tune"]
    for st in stages:
        t0 = time.time(); print(f"=== stage {st} ===", flush=True)
        if st == "tune":
            stage_tune(cfg, a.workers)
        elif st == "bridges":
            stage_nets("bridge", cfg, seeds, a.workers)
        elif st == "flows":
            stage_nets("flow0", cfg, seeds, a.workers)
            if a.with_dprime:
                stage_nets("fm", cfg, seeds, a.workers)
        elif st.startswith("ppo_"):
            stage_ppo(st[-1].upper(), cfg, seeds, a.workers)
            if st == "ppo_d" and a.with_dprime:
                stage_ppo("Dprime", cfg, seeds, a.workers)
        elif st == "eval":
            conds = [f"A{e}" for e in FIXED_EPS] + ["B", "C", "D", "E"] + (["Dprime"] if a.with_dprime else [])
            stage_eval(cfg, seeds, conds)
        elif st == "plots":
            import plots; plots.main()
        timing[st] = round(time.time() - t0, 1)
        print(f"=== {st} took {timing[st]}s ===", flush=True)
        tp = os.path.join(RES, "timing.json")
        old = json.load(open(tp)) if os.path.exists(tp) else {}
        old.update(timing); json.dump(old, open(tp, "w"), indent=1)


if __name__ == "__main__":
    main()
