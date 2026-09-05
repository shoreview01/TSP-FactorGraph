# Unrestricted symmetry-dual-path: 24-hour result

## Configuration

- Bins: 84
- Vehicles: 10, with every vehicle constrained to serve at least one bin
- Hours: 00--23
- Seed: 7
- Outer rounds: 32 per path
- Assignment rounds: 24 per outer round
- Damping: 0.5
- Message tolerance: 0.001
- Exact-trellis bin limit: 84 (no assigned-cluster cardinality cutoff)
- Paths: labeled vehicle assignment and canonical symmetry-quotient assignment
- Selection: the lower-energy feasible incumbent from the two paths

## Validation

- 24 unique hours were produced.
- All 24 solutions are coverage-, capacity-, battery-, and return-feasible.
- All 24 solutions use exactly 10 active vehicles.
- The labeled path was selected in 20 hours and the canonical path in 4 hours.
- Strict message convergence was reached in 4 of 24 hours. The other hours use the best feasible incumbent retained during the complete round budget; the results therefore do not claim global optimality.

## Mean results

| Method | Energy (kWh) | Mean remaining battery (kWh/vehicle) | Makespan (s) |
|---|---:|---:|---:|
| Proposed | 116.893 | 168.311 | 2834.5 |
| ACO | 121.961 | 167.804 | 2929.7 |
| GA | 130.857 | 166.914 | 2805.8 |
| PSO | 141.951 | 165.805 | 2940.7 |
| NN | 148.240 | 165.176 | 2884.7 |

The proposed method has the lowest energy in all 24 hours against every baseline. Relative to ACO, it saves 5.068 kWh per hour on average (4.16%); the smallest hourly saving is 1.623 kWh.

Compared with the earlier restricted proposed run (`maximum_exact_trellis_bins=20`), the unrestricted dual-path run is better in 16 hours and worse in 8 hours, with a mean improvement of 0.784 kWh. The restricted run is not included as a baseline because it uses a different search-space setting.

## Files

- `hourly_dual_path_seed7.csv`: proposed results for all 24 hours
- `hourly_dual_path_with_baselines_seed7.csv`: 120-row comparison data
- `hourly_energy_comparison_seed7.csv`: hour-wise energy and proposed margins
- `energy_win_summary_seed7.csv`: win counts and average savings
- `dual_path_branch_diagnostics_seed7.csv`: both path outcomes and final residuals
- `hourly_routes/`: 24 raw route and diagnostic JSON files
- `../../figures/hourly_24h/`: energy, battery, makespan, and combined figures in PDF, EPS, and PNG
