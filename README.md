# TSP-FactorGraph

Reproducible EV waste-collection experiments for the assignment/trellis
message-passing solver and the NN, ACO, PSO, and GA baselines.

## Repository layout

```text
solver/       proposed solver, message updates, exact trellis, and baselines
simul/        operational EV energy model and required SUMO/traffic/bin data
scripts/      experiment runner, experiment implementations, and plotters
experiments/  one JSON configuration for each retained paper experiment
results/      retained CSV/JSON/log outputs
figures/      publication figures, separated from raw results
tests/        solver and online-framework regression tests
docs/         manuscript and energy-model documentation
```

The three retained experiments are:

- `k_sweep`: K=6,...,14 at 04:00, 11:00, and 19:00 with 11,700 kg demand.
- `hourly_24h`: K=10 over all 24 hourly traffic snapshots.
- `online`: paired adaptive, live-static, and frozen-static policies over ten
  traffic seeds with gradual disruption and recovery.

## Setup

### Git LFS runtime input

The 226 MB elevated SUMO network
`simul/seongbuk_buffer_elevation.net.xml` is stored with Git Large File Storage
(Git LFS). Install Git LFS before cloning so that Git replaces the small pointer
in the repository with the actual XML file.

On macOS:

```bash
brew install git-lfs
git lfs install
```

On Windows, install Git LFS from [git-lfs.com](https://git-lfs.com/), open a
new PowerShell or Git Bash session, and run:

```powershell
git lfs install
```

Then clone the repository normally. The final `git lfs pull` is harmless when
the file was already downloaded automatically and ensures that it is present:

```bash
git clone https://github.com/shoreview01/TSP-FactorGraph.git
cd TSP-FactorGraph
git lfs pull
```

For an existing clone, install Git LFS and retrieve the runtime input with:

```bash
git lfs install
git lfs pull
```

Verify that the network is managed by Git LFS:

```bash
git lfs ls-files
```

The output should include `simul/seongbuk_buffer_elevation.net.xml`. Prefer
`git clone` over GitHub's **Download ZIP**, because a source archive may contain
only the LFS pointer instead of the 226 MB XML file.

### Python environment

Use Python 3.11 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The runtime road inputs are already under `simul/`; SUMO itself is not required
to rerun these experiments because the operational hourly edge table is checked
in.

## Run an experiment

Run from the repository root. Each configuration fixes the instance, solver,
baseline, traffic, parallelism, and destination paths. Completed shards/seeds
are reused while `"resume": true`.

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json
python -m scripts.run_experiment experiments/hourly_24h/config.json
python -m scripts.run_experiment experiments/online/config.json
```

To inspect the commands without starting a long run:

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json --dry-run
```

To regenerate only the paper figures from retained CSV files:

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json --figures-only
python -m scripts.run_experiment experiments/hourly_24h/config.json --figures-only
python -m scripts.run_experiment experiments/online/config.json --figures-only
```

New tabular outputs stay under `results/<experiment>/`; plots are written only
to `figures/<experiment>/` in PDF, EPS, and PNG formats.

## Validation

```bash
python -m unittest discover -s tests -v
```

The physical energy terms and parameter definitions are documented in
`docs/energy_model.md`. The latest manuscript is `docs/manuscript.pdf`.
