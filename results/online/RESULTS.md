# Multi-seed online replanning result

Traffic seeds: 2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035. Demand and solver seeds are fixed, and all three policies receive an identical traffic trace within each seed.

Approximately 5% of bins are selected as fixed disruption centers. Traffic remains normal for 10 minutes, speeds on edges within a 500 m road-distance radius then decrease linearly from 100% to 35% over 10 minutes, remain at 35% for 10 minutes, and recover linearly to 100% over 25 minutes.

Planning is modeled as asynchronous and nonblocking: vehicles do not stop, and planning wall time contributes neither auxiliary energy nor operating time. Energy trajectories use the idealized assumption that the new solution is available at the replanning epoch; measured wall time and 300 s deadline compliance are therefore reported separately. At every replanning epoch, a candidate is accepted only when exact evaluation under the current traffic state uses less residual energy than the incumbent route.

The line in each tracking panel is the arithmetic mean. The shaded cloud and final error bars show the full minimum-to-maximum range. Standard deviations are retained in the statistics CSV files.

| Method | Adaptive mean (kWh) | Live-navigation static (kWh) | Frozen-route static (kWh) | Gain vs. live static (%) | Wins vs. live | Gain vs. frozen static (%) | Wins vs. frozen | Cumulative replanning (s) | Mean / max single replan (s) | 300 s deadline hits | Accepted / attempted replans |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Proposed | 158.313 | 157.951 | 168.869 | -0.26 | 6/10 | 6.04 | 10/10 | 585.7 | 63.0 / 149.1 | 93/93 | 1.4 / 9.3 |
| NN | 196.911 | 216.542 | 238.052 | 9.04 | 10/10 | 16.85 | 10/10 | 53.3 | 6.7 / 11.2 | 80/80 | 2.5 / 8.0 |
| ACO | 159.199 | 162.383 | 171.974 | 1.93 | 6/10 | 7.26 | 9/10 | 654.6 | 67.5 / 229.4 | 97/97 | 3.0 / 9.7 |
| PSO | 180.038 | 200.824 | 220.297 | 10.24 | 10/10 | 17.86 | 10/10 | 1926.0 | 207.1 / 819.4 | 62/93 | 3.5 / 9.3 |
| GA | 184.704 | 180.291 | 190.496 | -2.48 | 3/10 | 2.75 | 8/10 | 2644.0 | 352.5 / 1256.9 | 41/75 | 2.3 / 7.5 |

## Proposed versus adaptive baselines

| Baseline | Mean Proposed difference (kWh) | Mean reduction (%) | Proposed wins |
|---|---:|---:|---:|
| NN | -38.597 | 18.64 | 9/10 |
| ACO | -0.885 | 0.47 | 5/10 |
| PSO | -21.725 | 11.44 | 8/10 |
| GA | -26.390 | 13.86 | 9/10 |

## Interpretation

Proposed adaptive minus live-navigation static was 0.362 kWh, and Proposed adaptive minus frozen-route static was -10.555 kWh on average. The exact acceptance gate was evaluated only under the currently observed traffic snapshot; it did not forecast the scheduled recovery. Therefore, a route that was cheaper at acceptance could become more expensive after the disruption weakened.

The seed-level Proposed difference relative to live-navigation static had correlation -0.356 with the number of accepted replans and correlation -0.020 with the number of reassigned bins. With only ten seeds and weak correlations, these data do not support a monotonic claim that more accepted or larger replans alone caused the final differences. No traffic forecast or next-commit-horizon model was added, as required by the experiment scope.
