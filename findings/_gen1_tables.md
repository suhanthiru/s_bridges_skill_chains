## Success (L1)

| source | layout | none | slip | rain | push |
|---|---|---|---|---|---|
| BRIDGE-slip | L1 | 0.993 ± 0.010 | 0.509 ± 0.084 | 0.191 ± 0.017 | 0.210 ± 0.016 |
| BRIDGE-brownian | L1 | 0.953 ± 0.032 | 0.167 ± 0.022 | 0.078 ± 0.014 | 0.036 ± 0.023 |
| BRIDGE-unicycle | L1 | 0.949 ± 0.025 | 0.158 ± 0.062 | 0.055 ± 0.024 | 0.040 ± 0.019 |
| PD-noise | L1 | 0.992 ± 0.007 | 0.619 ± 0.121 | 0.296 ± 0.034 | 0.505 ± 0.070 |
| PD-iso | L1 | 0.995 ± 0.006 | 0.708 ± 0.046 | 0.327 ± 0.023 | 0.536 ± 0.037 |
| MPC-relabel | L1 | 0.002 ± 0.003 | 0.198 ± 0.076 | 0.226 ± 0.052 | 0.069 ± 0.036 |

## Isolations (paired, L1)

| layout | disturbance | BRIDGE-slip − BRIDGE-unicycle | 95% CI | p |
|---|---|---|---|---|
| L1 | slip | +0.351 | [+0.260, +0.442] | 0.000 |
| L1 | rain | +0.136 | [+0.096, +0.176] | 0.001 |
| L1 | push | +0.170 | [+0.157, +0.183] | 0.000 |

| layout | disturbance | PD-noise − PD-iso | 95% CI | p |
|---|---|---|---|---|
| L1 | slip | -0.089 | [-0.228, +0.050] | 0.151 |
| L1 | rain | -0.031 | [-0.076, +0.014] | 0.127 |
| L1 | push | -0.031 | [-0.092, +0.030] | 0.229 |

| layout | disturbance | MPC-relabel − BRIDGE-slip | 95% CI | p |
|---|---|---|---|---|
| L1 | slip | -0.311 | [-0.455, -0.167] | 0.004 |
| L1 | rain | +0.035 | [-0.015, +0.085] | 0.124 |
| L1 | push | -0.141 | [-0.170, -0.112] | 0.000 |

| layout | disturbance | BRIDGE-slip − BRIDGE-brownian | 95% CI | p |
|---|---|---|---|---|
| L1 | slip | +0.342 | [+0.273, +0.411] | 0.000 |
| L1 | rain | +0.113 | [+0.086, +0.140] | 0.000 |
| L1 | push | +0.174 | [+0.158, +0.190] | 0.000 |

## Coverage

| source | off-path cells | off-path fraction | mean displacement from demos | success (slip) |
|---|---|---|---|---|
| BRIDGE-slip | 416 | 0.289 | 0.0378 | 0.509 |
| BRIDGE-brownian | 393 | 0.321 | 0.0410 | 0.167 |
| BRIDGE-unicycle | 375 | 0.299 | 0.0392 | 0.158 |
| PD-noise | 44 | 0.004 | 0.0072 | 0.619 |
| PD-iso | 11 | 0.001 | 0.0057 | 0.708 |
| MPC-relabel | 422 | 0.311 | 0.0395 | 0.198 |

Pearson r(coverage cells, success under slip) over sources: -0.820 (p = 0.046)
