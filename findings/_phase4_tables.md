## 4a width sweep

| w | slip x0.5: success | slip x1.0: success | slip x2.0: success | handoff-2 md (nominal cov, x1) |
|---|---|---|---|---|
| 0.5 | 0.331 ± 0.101 | 0.138 ± 0.034 | 0.028 ± 0.008 | 1.79 |
| 1.0 | 0.374 ± 0.112 | 0.162 ± 0.030 | 0.033 ± 0.014 | 1.97 |
| 2.0 | 0.426 ± 0.044 | 0.189 ± 0.035 | 0.040 ± 0.010 | 2.28 |
| 4.0 | 0.426 ± 0.071 | 0.202 ± 0.029 | 0.049 ± 0.008 | 3.30 |

| slip level | argmax_w [bootstrap 95% CI] | success |
|---|---|---|
| x0.5 | 2.0 [1.0, 4.0] | 0.426 |
| x1.0 | 4.0 [2.0, 4.0] | 0.202 |
| x2.0 | 4.0 [4.0, 4.0] | 0.049 |

## 4b bimodal

| kind | success | collision | P(upper) | handoff-2 inside either mode | W2 goal |
|---|---|---|---|---|---|
| multi | 0.293 ± 0.024 | 0.611 | 0.60 | 0.25 | 0.283 |
| two_bridge | 0.390 ± 0.015 | 0.447 | 0.45 | 0.41 | 0.237 |

Pearson r(P(upper), slip_lower − slip_upper) over 10 maps, multi-marginal bridge: r = -0.305, p = 0.391
