"""Terrain experiment runner: layouts -> demos -> train -> eval -> plots.

    python run_terrain.py                  # everything
    python run_terrain.py --quick          # smoke test
    python run_terrain.py --stage layouts --stage demos
    python run_terrain.py --until bridges  # stop after BRIDGE conditions (checkpoint report)
    python run_terrain.py --tune           # PD gains / learning rates at the tuning cell
"""
import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch

import env as E
import terrain_env as T
import demos as D
import conditions as C
import rl as R

RES = os.environ.get("SB_RES", "results_terrain")
MODELS, FIGS = os.path.join(RES, "models"), os.environ.get("SB_FIGS", "figures")
DATA = os.environ.get("SB_DATA", "data")
TRAIN_L, TUNE_L, TEST_L = list(range(20)), list(range(20, 25)), list(range(20, 30))
SIGMA_K = (0.0, 0.03, 0.06, 0.10, 0.15)
P_PUSH = 0.02
SEEDS = (0, 1, 2, 3, 4)
SETTINGS = ("mild", "rough")
CONDS = ("ORACLE", "TRACK", "BRIDGE-plain", "BRIDGE-obs", "DIFF", "PPO")

# Frozen after tuning on cell (sigma_k=0.06, mild), layouts 20-24; see results_terrain/tuning.csv.
CONFIG = dict(track=dict(kp=10.0, kd=0.0), bridge=dict(steps=4000, lr=3e-4), diff=dict(steps=20000, lr=1e-4),
              ppo=dict(lr=3e-4, rollout=32, epochs=4, minibatch=4096, gamma=0.99, lam=0.95, clip=0.2),
              ppo_steps=2_000_000, ppo_envs=512, episodes=200, n_demos=500)
QUICK = dict(CONFIG, bridge=dict(steps=300, lr=1e-3), diff=dict(steps=500, lr=3e-4), ppo_steps=40_000,
             ppo_envs=256, episodes=40, n_demos=50)


def dev():
    return E.get_device(os.environ.get("SB_DEVICE") or None)


def fields_all(d):
    p = os.path.join(RES, "layouts.npz")
    if not os.path.exists(p):
        t0 = time.time(); F = T.load_fields(range(30), "cpu")
        np.savez_compressed(p, fields=F.numpy()); print(f"generated 30 layouts in {time.time() - t0:.0f}s")
    return torch.tensor(np.load(p)["fields"], device=d)


def mpath(name, rough, seed):
    return os.path.join(MODELS, f"{name}_{rough}_s{seed}.pt")


def eps_ref(rough):
    """Reference Brownian noise level for the bridges = env process noise at mean roughness (0.5)."""
    return T.ROUGH[rough][2] ** 2 * 0.5


# ------------------------------------------------------------------- stages
def stage_layouts(cfg):
    d = dev(); F = fields_all(d); n = 300
    lids = torch.arange(n, device=d) % 30
    rows = []
    for rough in SETTINGS:
        envr = T.TerrainEnv(F, lids, rough, 0.0, 0.0, 1, d, disturb=False); o = envr.reset()
        ctl = C.OracleCtl(F, lids, rough, 1); succ = torch.zeros(n, dtype=torch.bool, device=d)
        for _ in range(300):
            o, r, term, _, info = envr.step(ctl.act(envr, o)); succ |= info["success"]
        per = succ.float().reshape(-1, 30).mean(0).cpu().numpy()
        rows += [dict(rough=rough, layout=i, oracle_success_no_push=float(per[i])) for i in range(30)]
        print(f"[layouts {rough}] oracle success without pushes: mean={per.mean():.3f} min={per.min():.2f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "layout_check.csv"), index=False)


def stage_demos(cfg):
    d = dev(); F = fields_all(d)
    for rough in SETTINGS:
        D.collect(DATA, F, TRAIN_L, rough, n=cfg["n_demos"], device=d)


def train_bridges(F, rough, seed, cfg, use_obs, d, log=print):
    nets = []
    for k in range(3):
        S, A, G, Lid = D.load_demos(DATA, TRAIN_L, rough, k)
        x0, x1 = torch.tensor(S[:, 0], device=d), torch.tensor(S[:, -1], device=d)
        lid = torch.tensor(Lid, device=d)
        log(f"[bridge-{'obs' if use_obs else 'plain'} {rough} s={seed} k={k + 1}] {len(x0)} pairs")
        net = C.train_terrain_bridge(F, x0, x1, lid, eps_ref(rough), use_obs, seed * 10 + k, d, log=log, **cfg["bridge"])
        nets.append(net)
    return nets


def stage_train(cfg, seeds, which):
    d = dev(); F = fields_all(d)
    for rough in SETTINGS:
        for s in seeds:
            E.seed_all(s)
            if "bridges" in which:
                for use_obs in (False, True):
                    nets = train_bridges(F, rough, s, cfg, use_obs, d)
                    torch.save([n.state_dict() for n in nets], mpath("bridge_obs" if use_obs else "bridge_plain", rough, s))
            if "diff" in which:
                obs, act = zip(*[C.demo_windows(*D.load_demos(DATA, TRAIN_L, rough, k)[:2], D.load_demos(DATA, TRAIN_L, rough, k)[3], F, k, d)
                                 for k in range(3)])
                obs, act = torch.cat(obs), torch.cat(act)
                t0 = time.time()
                pol = C.train_diffusion(F, obs, act, s, d, log=lambda m: print(f"[diff {rough} s={s}] {m}", flush=True), **cfg["diff"])
                torch.save(pol.state_dict(), mpath("diff", rough, s)); print(f"[diff {rough} s={s}] {time.time() - t0:.0f}s")
            if "ppo" in which:
                lids = torch.tensor(TRAIN_L, device=d).repeat(cfg["ppo_envs"] // len(TRAIN_L) + 1)[:cfg["ppo_envs"]]
                envr = C.PPOEnvWrap(T.TerrainEnv(F, lids, rough, 0.10, P_PUSH, s, d))  # trained under the mid push level
                pol, crit = C.TerrainPolicy().to(d), C.TerrainCritic().to(d)
                t0 = time.time()
                curve = R.ppo_train(envr, pol, crit, cfg["ppo_steps"], s, d, log=lambda m: print(f"[ppo {rough} s={s}] {m}", flush=True), **cfg["ppo"])
                torch.save(pol.state_dict(), mpath("ppo", rough, s)); print(f"[ppo {rough} s={s}] {time.time() - t0:.0f}s")
                pd.DataFrame(curve).assign(rough=rough, seed=s).to_csv(os.path.join(RES, f"ppo_curve_{rough}_s{s}.csv"), index=False)


def make_ctl(cond, F, lids, rough, seed, cfg, d, cache):
    key = (cond, rough, seed)
    if cond == "ORACLE":
        return C.OracleCtl(F, lids, rough, 10_000 + seed)
    if cond == "TRACK":
        if key not in cache:
            dm = [D.load_demos(DATA, TRAIN_L, rough, k) for k in range(3)]
            cache[key] = ([x[0] for x in dm], [x[1] for x in dm])
        S, A = cache[key]
        return C.TrackCtl(S, A, cfg["track"]["kp"], cfg["track"]["kd"], d)
    if cond.startswith("BRIDGE"):
        use_obs = cond.endswith("obs")
        if key not in cache:
            nets = [C.TerrainDrift(use_obs).to(d) for _ in range(3)]
            for n, sd in zip(nets, torch.load(mpath("bridge_obs" if use_obs else "bridge_plain", rough, seed), map_location=d)):
                n.load_state_dict(sd); n.eval()
            cache[key] = nets
        return C.BridgeCtl(cache[key], use_obs)
    if cond == "DIFF":
        if key not in cache:
            pol = C.DiffusionPolicy().to(d); pol.load_state_dict(torch.load(mpath("diff", rough, seed), map_location=d)); pol.eval()
            cache[key] = pol
        return C.DiffCtl(cache[key])
    if cond == "PPO":
        if key not in cache:
            pol = C.TerrainPolicy().to(d); pol.load_state_dict(torch.load(mpath("ppo", rough, seed), map_location=d)); pol.eval()
            cache[key] = pol
        return C.PPOCtl(cache[key])


@torch.no_grad()
def run_cell(cond, rough, sigma_k, seed, cfg, F, d, cache, layouts=TEST_L, record=False):
    n = cfg["episodes"]
    lids = torch.tensor(layouts, device=d).repeat(n // len(layouts) + 1)[:n]
    envr = T.TerrainEnv(F, lids, rough, sigma_k, P_PUSH, 20_000 + seed, d)
    ctl = make_ctl(cond, F, lids, rough, seed, cfg, d, cache)
    obs = envr.reset(); succ = torch.zeros(n, dtype=torch.bool, device=d)
    traj, pushes = [envr.x.clone()], []
    for _ in range(E.STEPS_PER_SKILL * E.N_SKILLS):
        obs, r, term, _, info = envr.step(ctl.act(envr, obs)); succ |= info["success"]
        if record:
            traj.append(envr.x.clone()); pushes.append(info["pushed"].clone())
    base = dict(condition=cond, rough=rough, sigma_k=sigma_k, seed=seed)
    ep = [dict(base, episode=i, layout=int(lids[i]), success=int(succ[i]), effort=float(info["effort"][i]),
               corridor_dev=float(info["corridor_dev"][i])) for i in range(n)]
    pl = [dict(base, episode=i, push_step=t, recovery=rec) for i, t, rec in envr.push_log]
    ho = [dict(base, handoff=j + 1, w2=E.w2(envr.handoff[j].cpu(), T.waypoint_sample(j + 1, n, "cpu").numpy())) for j in range(2)]
    extra = (torch.stack(traj, 1).cpu().numpy(), torch.stack(pushes, 1).cpu().numpy()) if record else None
    return ep, pl, ho, extra


def stage_eval(cfg, seeds, conds):
    d = dev(); F = fields_all(d); cache = {}
    ep, pl, ho = [], [], []
    for cond in conds:
        t0 = time.time()
        for rough in SETTINGS:
            for sk in SIGMA_K:
                for s in seeds:
                    e, p, h, _ = run_cell(cond, rough, sk, s, cfg, F, d, cache)
                    ep += e; pl += p; ho += h
        df = pd.DataFrame([x for x in ep if x["condition"] == cond])
        print(f"[eval {cond}] success={df.success.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)
    # example rollouts for fig 3: sigma_k=0.10, rough, layout 25, seed 0
    roll = {}
    for cond in conds:
        _, _, _, (traj, pushes) = run_cell(cond, "rough", 0.10, 0, dict(cfg, episodes=20), F, d, cache, layouts=[25], record=True)
        roll[f"{cond}_traj"], roll[f"{cond}_push"] = traj, pushes
    np.savez_compressed(os.path.join(RES, "rollouts.npz"), field=F[25].cpu().numpy(), **roll)
    write_results(ep, pl, ho)


def write_results(ep, pl, ho, tag=""):
    ep, pl, ho = pd.DataFrame(ep), pd.DataFrame(pl), pd.DataFrame(ho)
    ep.to_csv(os.path.join(RES, f"episodes{tag}.csv"), index=False)
    pl.to_csv(os.path.join(RES, f"pushes{tag}.csv"), index=False)
    ho.to_csv(os.path.join(RES, f"handoff{tag}.csv"), index=False)
    keys = ["condition", "rough", "sigma_k"]
    per_seed = ep.groupby(keys + ["seed"])[["success", "effort", "corridor_dev"]].mean()
    summ = per_seed.groupby(keys).agg(["mean", "std"]); summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    if len(pl):
        rec = pl.assign(never=pl.recovery < 0, rec=pl.recovery.where(pl.recovery >= 0))
        g = rec.groupby(keys + ["seed"]).agg(rec_median=("rec", "median"), never=("never", "mean"), n_push=("rec", "size"))
        g2 = g.groupby(keys).agg(rec_median_mean=("rec_median", "mean"), rec_median_std=("rec_median", "std"),
                                 never_mean=("never", "mean"), never_std=("never", "std"), n_push=("n_push", "sum"))
        summ = summ.join(g2)
    hw = ho.groupby(keys + ["handoff"]).w2.agg(["mean", "std"]).unstack("handoff")
    hw.columns = [f"handoff{h}_w2_{a}" for a, h in hw.columns]
    summ = summ.join(hw)
    summ.reset_index().to_csv(os.path.join(RES, f"summary{tag}.csv"), index=False)


def stage_tune(cfg):
    """Cell (sigma_k=0.06, mild), layouts 20-24, seed 0.  Writes tuning.csv; CONFIG is frozen by hand afterwards."""
    d = dev(); F = fields_all(d); rows = []
    ccfg = dict(cfg, episodes=200)

    def score(cond, c, cache=None):
        e, _, _, _ = run_cell(cond, "mild", 0.06, 0, c, F, d, cache if cache is not None else {}, layouts=TUNE_L)
        return float(np.mean([x["success"] for x in e]))
    for kp in (2.0, 5.0, 10.0):
        for kd in (0.0, 0.05):
            rows.append(dict(what="track", kp=kp, kd=kd, success=score("TRACK", dict(ccfg, track=dict(kp=kp, kd=kd)))))
            print(rows[-1], flush=True)
    for lr in (3e-4, 1e-3):
        for use_obs in (False, True):
            nets = train_bridges(F, "mild", 0, dict(ccfg, bridge=dict(cfg["bridge"], lr=lr)), use_obs, d, log=lambda m: None)
            name = "BRIDGE-obs" if use_obs else "BRIDGE-plain"
            rows.append(dict(what=name, lr=lr, success=score(name, ccfg, {(name, "mild", 0): nets})))
            print(rows[-1], flush=True)
    obs, act = zip(*[C.demo_windows(*D.load_demos(DATA, TRAIN_L, "mild", k)[:2], D.load_demos(DATA, TRAIN_L, "mild", k)[3], F, k, d) for k in range(3)])
    obs, act = torch.cat(obs), torch.cat(act)
    for lr in (1e-4, 3e-4):
        pol = C.train_diffusion(F, obs, act, 0, d, log=lambda m: None, **dict(cfg["diff"], lr=lr))
        rows.append(dict(what="diff", lr=lr, success=score("DIFF", ccfg, {("DIFF", "mild", 0): pol}))); print(rows[-1], flush=True)
    lids = torch.tensor(TRAIN_L, device=d).repeat(cfg["ppo_envs"] // 20 + 1)[:cfg["ppo_envs"]]
    for lr in (1e-4, 3e-4):
        pol, crit = C.TerrainPolicy().to(d), C.TerrainCritic().to(d)
        R.ppo_train(C.PPOEnvWrap(T.TerrainEnv(F, lids, "mild", 0.10, P_PUSH, 0, d)), pol, crit, cfg["ppo_steps"] // 4, 0, d,
                    log=lambda m: None, **dict(cfg["ppo"], lr=lr))
        rows.append(dict(what="ppo", lr=lr, success=score("PPO", ccfg, {("PPO", "mild", 0): pol}))); print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "tuning.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true"); ap.add_argument("--tune", action="store_true")
    ap.add_argument("--stage", action="append", help="layouts|demos|train_bridges|train_diff|train_ppo|eval|plots")
    ap.add_argument("--until", default=None, help="stop after this stage (e.g. bridges)")
    ap.add_argument("--conds", default=",".join(CONDS))
    a = ap.parse_args()
    os.makedirs(MODELS, exist_ok=True); os.makedirs(FIGS, exist_ok=True); os.makedirs(DATA, exist_ok=True)
    cfg = QUICK if a.quick else CONFIG
    seeds = SEEDS[:2] if a.quick else SEEDS
    conds = a.conds.split(",")
    json.dump(dict(seeds=list(seeds), quick=a.quick, config=cfg, torch=torch.__version__, device=str(dev()),
                   train_layouts=TRAIN_L, test_layouts=TEST_L, tune_layouts=TUNE_L),
              open(os.path.join(RES, "seeds.json"), "w"), indent=1)
    stages = ["tune"] if a.tune else (a.stage or ["layouts", "demos", "train_bridges", "train_diff", "train_ppo", "eval", "plots"])
    for st in stages:
        t0 = time.time(); print(f"=== stage {st} ===", flush=True)
        if st == "tune": stage_tune(cfg)
        elif st == "layouts": stage_layouts(cfg)
        elif st == "demos": stage_demos(cfg)
        elif st.startswith("train_"): stage_train(cfg, seeds, [st[6:]])
        elif st == "eval": stage_eval(cfg, seeds, conds)
        elif st == "plots":
            import plots_terrain; plots_terrain.main()
        el = round(time.time() - t0, 1); print(f"=== {st} took {el}s ===", flush=True)
        tp = os.path.join(RES, "timing.json"); old = json.load(open(tp)) if os.path.exists(tp) else {}
        old[st] = el; json.dump(old, open(tp, "w"), indent=1)
        if a.until and st == a.until:
            break


if __name__ == "__main__":
    main()
