"""Bridge-as-generator suite: phases g0-g6.

Fixed setup: SE(2) terrain env, Phase 5 demo set (clean demonstrator commands), N_demo = 20,
generated set = 4 x N_demo, diffusion policy (beta_max 0.2) with the same training budget
for every data source, 5 seeds, 200 evaluation episodes per cell.  Every generator runs on
the reference kinematics with a noise stream shared across sources for a given seed
(gen_sources.NoiseStream), so PD-noise is a control for BRIDGE-slip and not a different
experiment.
"""
import json
import math
import os
import time
import numpy as np
import pandas as pd
import torch

import se2 as S
import terrain as TR
import task as TK
import solver as SV
import phase2 as P2
import phase1b as P1B
import gen_sources as GS
import gen_policies as GP

RES, MODELS = "results_generator", os.path.join("results_generator", "models")
N_DEMO, GEN_MULT, N_EVAL = 20, 4, 200
STEPS, STEPS_FAST = 8000, 2500
DISTS = ("none", "slip", "rain", "push")
SOURCES_0 = ("DEMO", "NOISED", "BRIDGE-slip", "PD-noise", "DART", "MPC-rollout")
SOURCES_1 = ("BRIDGE-slip", "BRIDGE-brownian", "BRIDGE-unicycle", "PD-noise", "PD-iso", "MPC-relabel")
BRIDGE_CFG = dict(n_pair=6000, steps0=1200, steps_ipf=300, batch=512, lr=1e-3, n_sim=50)
os.makedirs(MODELS, exist_ok=True)


def dev():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------- demos
def demos_for(tk, layout, n, seed, device, demo_noise=0.0, standard=True):
    """Phase 5 demo set for the standard L1/L2 tasks; otherwise fresh demonstrator rollouts
    on `tk` (Nominal, kp = 6) with optional action noise recorded in the labels."""
    if standard and demo_noise == 0.0:
        G, U = P2.load_demos(layout, device); return G[:n], U[:n]
    g0 = torch.Generator(device=device).manual_seed(4242 + seed)
    tk0 = type(tk).__new__(type(tk)); tk0.__dict__.update(tk.__dict__); tk0.gen = g0; tk0.dist = "none"
    nom = P2.Nominal(tk0); g = tk0.sample(0, n); G, U = [g.clone()], []
    for k in range(TK.N_SKILL):
        for t in range(TK.T_SKILL):
            tau = torch.full((n,), t / TK.T_SKILL, device=device)
            u = TK.clip_u(nom(g, k, tau, 0)[0] + demo_noise * torch.randn(n, 3, generator=g0, device=device))
            g = S.compose(g, S.exp_se2(u * TK.DT + math.sqrt(TK.DT) * TK.SIGMA_BASE * torch.randn(n, 3, generator=g0, device=device)))
            G.append(g.clone()); U.append(u)
    return torch.stack(G, 1), torch.stack(U, 1)


# -------------------------------------------------------------- bridges
def get_bridges(kind, tk, layout, seed, device, tag="", mf=S.SE2, steps0=None):
    """Iteration-0 bridge nets for reference `kind` on task `tk` (cached by tag)."""
    if kind == "slip" and tag == "" and layout in ("L1", "L2") and mf.name == "se2" and os.path.exists(P2.mpath("bridge", layout, seed)):
        return torch.load(P2.mpath("bridge", layout, seed), map_location=device)    # Phase 2's nets
    p = os.path.join(MODELS, f"bridge_{kind}_{layout}{tag}_{mf.name}_s{seed}.pt")
    if os.path.exists(p):
        return torch.load(p, map_location=device)
    torch.manual_seed(seed); cpu = torch.device("cpu")
    tkc = type(tk).__new__(type(tk)); tkc.__dict__.update({k: (v.to(cpu) if torch.is_tensor(v) else v) for k, v in tk.__dict__.items()})
    tkc.means = [m.to(cpu) for m in tk.means]; tkc.covs = [c.to(cpu) for c in tk.covs]; tkc.device = cpu
    tkc.gen = torch.Generator(device=cpu).manual_seed(1000 + seed)
    if hasattr(tkc, "body_std"):
        tkc.body_std = tk.body_std.to(cpu)
    cfg = dict(BRIDGE_CFG); cfg["steps0"] = steps0 or cfg["steps0"]
    ref_body = tk.body_std ** 2 if hasattr(tk, "body_std") else None
    from sde import Reference
    ref = Reference(kind, sigma=0.05, kappa=2.0, slip_scale=tk.slip_scale if hasattr(tk, "slip_scale") else 1.0,
                    fields=tkc.obs_fields, body_cov=ref_body.to(cpu) if ref_body is not None else None)
    nets = {}
    for k in range(TK.N_SKILL):
        nets[k], _ = SV.train_skill(tkc, k, ref, mf, seed * 17 + k, cpu, K=0, log=lambda m: None, with_bwd=False, **cfg)
    nets = {k: {key: {pn: t.to(device) for pn, t in sd.items()} for key, sd in n_.items()} for k, n_ in nets.items()}
    torch.save(nets, p)
    return nets


# ------------------------------------------------------------- datasets
def build(source, tk, layout, seed, device, demos, n_demo=N_DEMO, mult=GEN_MULT, ref_kind="slip", mf=S.SE2,
          tag="", world_heading=None, cache=None, bridge_steps=None):
    """Return (G, U, meta).  meta has coverage of the generated part."""
    cache = cache if cache is not None else {}
    Gd, Ud = demos[0][:n_demo], demos[1][:n_demo]
    m = mult * n_demo
    z = GS.NoiseStream(m, seed, device)
    if source == "DEMO":
        return Gd, Ud, dict(n_gen=0)
    if source == "NOISED":
        Gn, Un = GS.noised_copies(Gd, Ud, m, seed, device)
        return torch.cat([Gd, Gn]), torch.cat([Ud, Un]), dict(n_gen=m, **GS.coverage(Gn, demos[0]))
    ref = GS.make_ref(ref_kind, tk)
    if hasattr(tk, "body_std"):
        ref.body_cov = tk.body_std ** 2
    if source.startswith("BRIDGE"):
        kind = source.split("-")[1]
        r = GS.make_ref(kind, tk)
        if hasattr(tk, "body_std"):
            r.body_cov = tk.body_std ** 2
        nets = get_bridges(kind, tk, layout, seed, device, tag, mf, steps0=bridge_steps)
        Gg, Ug = GS.rollout(tk, mf, GS.bridge_act(nets, mf, tk), m, z, r)
    elif source == "PD-noise":
        Gg, Ug = GS.rollout(tk, mf, GS.pd_act(Gd, Ud), m, z, ref)
    elif source == "PD-world":
        Gg, Ug = GS.rollout(tk, mf, GS.pd_act(Gd, Ud), m, z, ref, world_heading=world_heading)
    elif source == "PD-iso":
        key = ("bridge_states", layout, seed, tag)
        if key not in cache:
            nets = get_bridges("slip", tk, layout, seed, device, tag, mf)
            cache[key] = GS.rollout(tk, mf, GS.bridge_act(nets, mf, tk), m, z, ref)[0]
        v = GS.iso_variance(ref, mf, cache[key])
        Gg, Ug = GS.rollout(tk, mf, GS.pd_act(Gd, Ud), m, z, ref, iso_var=v)
    elif source == "DART":
        Gg, Ug = GS.rollout(tk, mf, GS.nominal_act(tk), m, z, ref, action_noise_ref=ref)
    elif source == "MPC-rollout":
        Gg, Ug = GS.rollout(tk, mf, GS.mpc_act(tk, seed), m, z, ref)
    elif source == "MPC-relabel":
        nets = get_bridges("slip", tk, layout, seed, device, tag, mf)
        Gb, _ = GS.rollout(tk, mf, GS.bridge_act(nets, mf, tk), m, z, ref)
        Gg, Ug = GS.rollout(tk, mf, GS.mpc_act(tk, seed), m, z, ref, states=Gb)
    else:
        raise ValueError(source)
    return torch.cat([Gd, Gg]), torch.cat([Ud, Ug]), dict(n_gen=m, **GS.coverage(Gg, demos[0]))


def train_eval(kind, tk_train, G, U, seed, device, steps, evals, log=lambda m: None):
    """Train one policy and evaluate on each (name, tk_eval) in evals."""
    t0 = time.time(); pol = GP.train(kind, tk_train, G, U, seed, device, steps, log)
    rows = []
    for name, tke in evals:
        m = GP.evaluate(pol, tke, N_EVAL)
        rows.append(dict(disturbance=name, train_s=time.time() - t0, **m))
    return pol, rows


def std_task(layout, dist, n, seed, device, **kw):
    return GS.GenTask(TR.Layout(layout), dist, n, seed, device, **kw)


def eval_tasks(layout, seed, device, dists=DISTS, **kw):
    return [(d, std_task(layout, d, N_EVAL, 50_000 + seed, device, **kw)) for d in dists]


def pol_path(phase, source, layout, seed, extra=""):
    return os.path.join(MODELS, f"pol_{phase}_{source}_{layout}{extra}_s{seed}.pt")


def write(rows, phase, seed):
    p = os.path.join(RES, f"phase{phase}_seed{seed}.parquet")
    pd.DataFrame(rows).to_parquet(p, index=False); print(f"wrote {len(rows)} rows -> {p}", flush=True)


def base_row(**kw):
    r = dict(phase=None, source=None, layout="L1", disturbance=None, seed=None, policy="diffusion", n_demo=N_DEMO,
             gen_mult=GEN_MULT, slip_scale_eval=0.7, slip_scale_gen=0.7, aniso=1.0, n_seed=14, push_mult=1.0,
             heading_std=None, demo_noise=0.0, map_flip=0.0, manifold="se2", route="L", n_gen=0,
             cov_cells=np.nan, off_frac=np.nan, mean_disp=np.nan)
    r.update(kw); return r


# ---------------------------------------------------------------- g0
def phase0(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []
    for layout in ("L1", "L2"):
        tk = std_task(layout, "none", 64, 1000 + seed, device)
        demos = demos_for(tk, layout, 500, seed, device)
        evals = eval_tasks(layout, seed, device)
        for source in SOURCES_0:
            t0 = time.time()
            G, U, meta = build(source, tk, layout, seed, device, demos)
            pol, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
            torch.save(pol.state_dict(), pol_path("g0", source, layout, seed))
            for r in rs:
                rows.append(base_row(phase="g0", source=source, layout=layout, seed=seed, **meta, **r))
            print(f"[g0 s{seed} {layout} {source}] " + " ".join(f"{r['disturbance']}:{r['success']:.2f}" for r in rs) + f" ({time.time() - t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------- g1
def phase1(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []; layout = "L1"; cache = {}
    tk = std_task(layout, "none", 64, 1000 + seed, device)
    demos = demos_for(tk, layout, 500, seed, device); evals = eval_tasks(layout, seed, device)
    for source in SOURCES_1:
        t0 = time.time()
        G, U, meta = build(source, tk, layout, seed, device, demos, cache=cache)
        p0 = pol_path("g0", source, layout, seed)
        if os.path.exists(p0) and not quick:            # identical data + seed as g0
            pol = P2.DiffPolicy().to(device); pol.load_state_dict(torch.load(p0, map_location=device)); pol.eval()
            rs = [dict(disturbance=n, train_s=0.0, **GP.evaluate(pol, tke, N_EVAL)) for n, tke in evals]
        else:
            pol, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
        for r in rs:
            rows.append(base_row(phase="g1", source=source, layout=layout, seed=seed, **meta, **r))
        print(f"[g1 s{seed} {source}] " + " ".join(f"{r['disturbance']}:{r['success']:.2f}" for r in rs) + f" cov {meta.get('cov_cells')} ({time.time() - t0:.0f}s)", flush=True)
    return rows


# ---------------------------------------------------------------- g2
def dart_in_range():
    import glob
    fs = glob.glob(os.path.join(RES, "phaseg0_seed*.parquet"))
    if not fs:
        return False
    d = pd.concat([pd.read_parquet(f) for f in fs]); d = d[(d.layout == "L1") & (d.disturbance == "slip")]
    m = d.groupby("source").success.mean()
    return abs(m.get("DART", 0) - max(m.get("BRIDGE-slip", 0), m.get("PD-noise", 0))) <= 0.05


def phase2(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []; layout = "L1"
    sources = ("BRIDGE-slip", "PD-noise") + (("DART",) if dart_in_range() else ())
    tk = std_task(layout, "none", 64, 1000 + seed, device)
    demos = demos_for(tk, layout, 500, seed, device); evals = eval_tasks(layout, seed, device)
    mults = (1, 4, 16, 64) if not quick else (1, 4)
    for source in sources:                                        # 2a generated size
        for mult in mults:
            G, U, meta = build(source, tk, layout, seed, device, demos, mult=mult)
            _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
            for r in rs:
                rows.append(base_row(phase="g2a", source=source, layout=layout, seed=seed, gen_mult=mult, **meta, **r))
            print(f"[g2a s{seed} {source} x{mult}] slip {rs[1]['success']:.2f}", flush=True)
        for nd in ((1, 5, 20, 100) if not quick else (1, 5)):     # 2b demo count
            G, U, meta = build(source, tk, layout, seed, device, demos, n_demo=nd)
            _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
            for r in rs:
                rows.append(base_row(phase="g2b", source=source, layout=layout, seed=seed, n_demo=nd, **meta, **r))
            print(f"[g2b s{seed} {source} N={nd}] slip {rs[1]['success']:.2f}", flush=True)
        p0 = pol_path("g0", source, layout, seed)                  # 2c severity, g0 policy (1x data)
        if os.path.exists(p0):
            pol = P2.DiffPolicy().to(device); pol.load_state_dict(torch.load(p0, map_location=device)); pol.eval()
            for sc in (0.5, 1.0, 2.0, 4.0):
                tke = std_task(layout, "slip", N_EVAL, 50_000 + seed, device, slip_scale=0.7 * sc)
                rows.append(base_row(phase="g2c", source=source, layout=layout, seed=seed, disturbance="slip",
                                     slip_scale_eval=0.7 * sc, train_s=0.0, **GP.evaluate(pol, tke, N_EVAL)))
    return rows


# ---------------------------------------------------------------- g3
def phase3(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []
    for layout in ("L1", "L2"):
        tk = std_task(layout, "none", 64, 1000 + seed, device)
        demos = demos_for(tk, layout, 500, seed, device); evals = eval_tasks(layout, seed, device, dists=("none", "slip"))
        for source in ("BRIDGE-slip", "PD-noise"):
            G, U, meta = build(source, tk, layout, seed, device, demos)
            for kind in ("diffusion", "flow", "bc", "bc_gmm"):
                p0 = pol_path("g0", source, layout, seed)
                if kind == "diffusion" and os.path.exists(p0) and not quick:
                    pol = P2.DiffPolicy().to(device); pol.load_state_dict(torch.load(p0, map_location=device)); pol.eval()
                    rs = [dict(disturbance=n, train_s=0.0, **GP.evaluate(pol, tke, N_EVAL)) for n, tke in evals]
                else:
                    _, rs = train_eval(kind, tk, G, U, seed, device, steps, evals)
                for r in rs:
                    rows.append(base_row(phase="g3", source=source, layout=layout, seed=seed, policy=kind, **meta, **r))
                print(f"[g3 s{seed} {layout} {source} {kind}] none {rs[0]['success']:.2f} slip {rs[1]['success']:.2f}", flush=True)
    return rows


# ---------------------------------------------------------------- g4
def curved_task(h, a, n, seed, device):
    means, covs = P1B.marginals_1b("scurve", h)
    tk = P1B.UniformTask(means, covs, P1B.body_std(h, a), n, seed, device)
    tk.slip_scale = 1.0
    return tk


def phase4(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []
    for h in ((0.2, 0.5) if not quick else (0.2,)):
        for a in ((1, 3) if not quick else (1,)):
            tk = curved_task(h, a, 64, 1000 + seed, device)
            demos = demos_for(tk, "scurve", 100, seed, device, standard=False)
            tke = [("body", curved_task(h, a, N_EVAL, 50_000 + seed, device))]
            theta_bar = [float(S.between(tk.means[k][None], tk.means[k + 1][None])[0, 2]) for k in range(3)]
            for source, mf, wh in (("BRIDGE-slip", S.SE2, None), ("BRIDGE-slip", S.Flat, None),
                                   ("PD-noise", S.SE2, None), ("PD-world", S.SE2, theta_bar)):
                tag = f"_scurve_h{h}_a{a}"
                G, U, meta = build(source, tk, "scurve", seed, device, demos, mf=mf, tag=tag, world_heading=wh)
                _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, tke)
                gen_name = "PD-noise-tangent" if source == "PD-noise" else ("PD-noise-world" if source == "PD-world" else f"BRIDGE-slip-{mf.name}")
                for r in rs:
                    rows.append(base_row(phase="g4", source=gen_name, layout="scurve", seed=seed, heading_std=h, aniso=a,
                                         manifold=mf.name, route="scurve", **meta, **r))
                print(f"[g4 s{seed} h={h} a={a} {gen_name}] success {rs[0]['success']:.2f}", flush=True)
    return rows


# ---------------------------------------------------------------- g5
AXES = ("slip_mag", "aniso", "corr_len", "push_rate", "heading", "n_demo", "demo_noise")


def decode(u):
    """[0,1]^7 -> theta.  Log scale where the prompt's range spans a decade or more."""
    u = np.clip(np.asarray(u, float), 0, 1)
    return dict(slip_mag=float(0.5 * (8.0 ** u[0])), aniso=float(10.0 ** u[1]),
                corr_len=float(0.25 * (32.0 ** u[2])), push_rate=float(0.2 * u[3]),
                heading=float(0.05 * (20.0 ** u[4])), n_demo=int([1, 5, 20][min(int(u[5] * 3), 2)]),
                demo_noise=float(0.3 * u[6]))


def theta_task(th, dist, n, seed, device):
    """corr_len (0.25-8, box units x 4) -> Voronoi seed count; push_rate -> class push-rate multiplier
    relative to the nominal mean rate (0.014)."""
    n_seed = int(np.clip(round(64.0 / (th["corr_len"] * 4) ** 2 * 4), 2, 64))
    return std_task("L1", dist, n, seed, device, slip_scale=0.7 * th["slip_mag"], aniso=th["aniso"], n_seed=n_seed,
                    push_mult=th["push_rate"] / 0.014, heading_std=th["heading"])


def f_theta(th, seeds, device, steps=STEPS_FAST, n_eval=100, log_rows=None, tag=""):
    """success(BRIDGE-slip data) - success(PD-noise data) at env theta, mean over seeds."""
    diffs = []
    for seed in seeds:
        tk = theta_task(th, "none", 64, 1000 + seed, device)
        demos = demos_for(tk, "L1", max(th["n_demo"], 1), seed, device, demo_noise=th["demo_noise"], standard=False)
        tke = [("slip", theta_task(th, "slip", n_eval, 50_000 + seed, device))]
        succ = {}
        ttag = f"_th{tag}_{abs(hash(tuple(round(v, 4) if isinstance(v, float) else v for v in th.values()))) % 10 ** 8}"
        for source in ("BRIDGE-slip", "PD-noise"):
            G, U, meta = build(source, tk, "L1", seed, device, demos, n_demo=th["n_demo"], tag=ttag,
                               bridge_steps=600 if not tag.startswith("validate") else None)   # search: cheaper bridges
            _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, tke)
            succ[source] = rs[0]["success"]
            if log_rows is not None:
                log_rows.append(base_row(phase="g5", source=source, layout="L1", seed=seed, n_demo=th["n_demo"],
                                         slip_scale_gen=0.7 * th["slip_mag"], slip_scale_eval=0.7 * th["slip_mag"], aniso=th["aniso"],
                                         n_seed=tk.cmap.max() + 1, push_mult=th["push_rate"] / 0.014, heading_std=th["heading"],
                                         demo_noise=th["demo_noise"], corr_len=th["corr_len"], push_rate=th["push_rate"],
                                         search=tag, **meta, **rs[0]))
        diffs.append(succ["BRIDGE-slip"] - succ["PD-noise"])
    return float(np.mean(diffs))


def l27():
    """3-level fractional factorial for 7 factors (3^(7-4), resolution III)."""
    runs = []
    for a in range(3):
        for b in range(3):
            for c in range(3):
                runs.append([a, b, c, (a + b) % 3, (a + 2 * b) % 3, (a + c) % 3, (b + c) % 3])
    return np.array(runs) / 2.0


def phase5(seed, quick=False):
    import cma
    device = dev(); seeds = (0, 1, 2); steps = 300 if quick else STEPS_FAST
    n_evals = 6 if quick else 60; rows = []; log = []
    for sign, name in ((-1.0, "max"), (1.0, "min")):
        es = cma.CMAEvolutionStrategy(7 * [0.5], 0.25, {"bounds": [0, 1], "seed": 1 + (sign > 0), "verbose": -9})
        n = 0
        while n < n_evals:
            X = es.ask(); vals = []
            for x in X:
                if n >= n_evals:
                    vals.append(0.0); continue
                th = decode(x); t0 = time.time()
                f = f_theta(th, seeds, device, steps, 100 if not quick else 30, rows, tag=name)
                log.append(dict(search=name, eval=n, f=f, **th)); vals.append(sign * f); n += 1
                print(f"[g5 {name} {n}/{n_evals}] f={f:+.3f} {th} ({time.time() - t0:.0f}s)", flush=True)
            es.tell(X, vals)
    for i, u in enumerate(l27() if not quick else l27()[:2]):
        th = decode(u); f = f_theta(th, seeds, device, steps, 100 if not quick else 30, rows, tag="factorial")
        log.append(dict(search="factorial", eval=i, f=f, **th))
        print(f"[g5 factorial {i + 1}/27] f={f:+.3f}", flush=True)
    lg = pd.DataFrame(log); lg.to_parquet(os.path.join(RES, "phaseg5_search.parquet"), index=False)
    # validation: top-3 and bottom-3 theta with 10 fresh seeds, 200 episodes
    srch = lg[lg.search != "factorial"].sort_values("f")
    for lab, sub in (("top", srch.tail(3)), ("bottom", srch.head(3))):
        for _, r in sub.iterrows():
            th = {k: r[k] for k in AXES}; th["n_demo"] = int(th["n_demo"])
            fv = f_theta(th, tuple(range(10, 13 if quick else 20)), device, steps, 200, rows, tag=f"validate_{lab}")
            log.append(dict(search=f"validate_{lab}", eval=-1, f=fv, **th))
            print(f"[g5 validate {lab}] f={fv:+.3f} {th}", flush=True)
    pd.DataFrame(log).to_parquet(os.path.join(RES, "phaseg5_search.parquet"), index=False)
    return rows


# ---------------------------------------------------------------- g6
def phase6(seed, quick=False):
    device = dev(); steps = 300 if quick else STEPS; rows = []; layout = "L1"
    sources = ("BRIDGE-slip", "PD-noise", "DART")
    mild, rough = 0.35, 1.4
    for gen_sc, ev_sc, lab in ((mild, rough, "mild->rough"), (rough, mild, "rough->mild")):
        tk = std_task(layout, "none", 64, 1000 + seed, device, slip_scale=gen_sc)
        demos = demos_for(tk, layout, 500, seed, device)
        evals = [(f"slip", std_task(layout, "slip", N_EVAL, 50_000 + seed, device, slip_scale=ev_sc))]
        for source in sources:
            G, U, meta = build(source, tk, layout, seed, device, demos, tag=f"_sc{gen_sc}")
            _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
            for r in rs:
                rows.append(base_row(phase="g6a", source=source, layout=layout, seed=seed, slip_scale_gen=gen_sc,
                                     slip_scale_eval=ev_sc, transfer=lab, **meta, **r))
            print(f"[g6a s{seed} {lab} {source}] {rs[0]['success']:.2f}", flush=True)
    for flip in ((0.1, 0.3, 0.5) if not quick else (0.3,)):     # wrong terrain model in the generator
        tk = std_task(layout, "none", 64, 1000 + seed, device, map_flip=flip)
        demos = demos_for(tk, layout, 500, seed, device)
        evals = [("slip", std_task(layout, "slip", N_EVAL, 50_000 + seed, device))]
        for source in sources:
            G, U, meta = build(source, tk, layout, seed, device, demos, tag=f"_flip{flip}")
            _, rs = train_eval("diffusion", tk, G, U, seed, device, steps, evals)
            for r in rs:
                rows.append(base_row(phase="g6b", source=source, layout=layout, seed=seed, map_flip=flip, **meta, **r))
            print(f"[g6b s{seed} flip={flip} {source}] {rs[0]['success']:.2f}", flush=True)
    return rows


PHASES = {"g0": phase0, "g1": phase1, "g2": phase2, "g3": phase3, "g4": phase4, "g5": phase5, "g6": phase6}


def run(phase, seed, quick=False):
    os.makedirs(RES, exist_ok=True)
    t0 = time.time(); rows = PHASES[phase](seed, quick)
    write(rows, phase + ("q" if quick else ""), seed)
    tp = os.path.join(RES, "timing.json"); old = json.load(open(tp)) if os.path.exists(tp) else {}
    old[f"{phase}_seed{seed}"] = round(time.time() - t0, 1); json.dump(old, open(tp, "w"), indent=1)
    print(f"=== generator phase {phase} seed {seed} took {time.time() - t0:.0f}s ===", flush=True)


def merge():
    import glob
    fs = sorted(glob.glob(os.path.join(RES, "phaseg[0-9]_seed*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df.to_parquet("results_generator.parquet", index=False); print(f"merged {len(fs)} shards, {len(df)} rows")
