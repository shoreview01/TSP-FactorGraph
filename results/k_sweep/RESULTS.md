# K sweep with 11,700 kg total demand

## Experiment

- Instance: 84 bins, seed 7, total demand 11,700 kg
- Starting hours: 04:00, 11:00, and 19:00
- Exact nonempty vehicle counts: K=6,...,14
- Methods: Proposed, NN, ACO, PSO, and GA
- Proposed: unrestricted labeled/canonical dual path, 32 rounds per path,
  24 assignment rounds, damping 0.5, exact-trellis limit 84

All 135 method/hour/K observations are feasible, serve all 84 bins and all
11,700 kg of demand, and use exactly K nonempty vehicles.

## Main results

Proposed is the lowest-energy method in all 27 fixed hour/K comparisons.

| Starting hour | Proposed optimum | Best baseline optimum | Saving |
|---:|---:|---:|---:|
| 04:00 | K=7, 73.042 kWh | ACO K=9, 80.287 kWh | 7.245 kWh (9.02%) |
| 11:00 | K=6, 150.306 kWh | ACO K=8, 163.150 kWh | 12.844 kWh (7.87%) |
| 19:00 | K=6, 97.503 kWh | ACO K=7, 107.036 kWh | 9.532 kWh (8.91%) |

The three-hour Proposed mean is 107.100 kWh at K=6, 108.019 kWh at K=7,
110.610 kWh at K=8, and then generally increases through 121.035 kWh at
K=14. Thus, the requested interior U-shaped optimum is not observed over
K=6,...,14: the mean minimum remains at the lower boundary K=6. The 04:00
curve alone has a shallow interior minimum at K=7.

## Files

- `k_sweep_3h_all_methods.csv`: all 135 validated observations
- `method_k_average.csv`: three-hour averages by method and K
- `winner_by_hour_k.csv`: fixed-hour/fixed-K winner table
- `method_hour_energy_optima.csv`: optimum K by method and hour
- `experiment_manifest.json`: experiment and validation manifest
- `../../figures/k_sweep/`: energy, remaining-battery, and makespan figures in PNG,
  PDF, and EPS formats
- `h04/`, `h11/`, and `h19/`: raw per-K results and route diagnostics
