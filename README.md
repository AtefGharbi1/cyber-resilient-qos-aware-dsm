# CQ-DSM: Cyber-Physical Quality-of-Service-Aware Demand-Side Management

This repository contains the simulation code and experiment runner used for the accompanying research paper on cyber-physical, QoS-aware demand-side management under communication degradation and cyberattack scenarios.

The code implements the CQ-DSM simulation workflow, including:

- prosumer and multi-day data generation;
- home energy management system (HEMS) MILP optimization;
- rolling-horizon HVAC, appliance, and battery scheduling;
- anomaly-detection-assisted cyberattack handling;
- comparative experiments across baseline and proposed methods;
- JSON export of experimental results.

## Repository structure

```text
.
├── config.py                    # Global simulation parameters
├── data_gen.py                  # Synthetic prosumer, QoS, price, PV, and attack data generation
├── hems_milp.py                 # HEMS MILP and day-ahead appliance scheduling models
├── lstm_detector.py             # Sliding-window temporal anomaly detector
├── simulator.py                 # Full-day simulation engine
├── run_experiments.py           # Main experiment runner
├── nyiso_20230815_nyc_lbmp.csv  # NYISO LBMP price profile used by the experiments
└── requirements.txt             # Python dependencies
```

## Requirements

The code was tested with Python 3.9+.

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The MILP components use the `mip` Python package with the CBC solver backend.

## Reproducing the experiments

For a quick validation run using one seed:

```bash
python run_experiments.py --quick
```

For the full configured experiment set:

```bash
python run_experiments.py
```

To run sequentially for debugging:

```bash
python run_experiments.py --quick --workers 1
```

The main output file is written to:

```text
results_v11.json
```

## Main command-line options

```bash
python run_experiments.py --quick              # one-seed validation run
python run_experiments.py --seed 42 --quick    # one-seed validation with selected seed
python run_experiments.py --seeds 42 7 13      # custom full-run seed list
python run_experiments.py --workers 1          # disable method-level parallelism
```

## Notes for paper submission

If the target venue uses double-blind review, use an anonymized repository or an archival anonymous link as required by the venue. Avoid author names, institutional names, personal GitHub accounts, and self-identifying commit metadata until after review.

Recommended manuscript wording:

> The implementation and scripts needed to reproduce the reported experiments are available in the accompanying code repository. The repository includes the simulator, MILP-based HEMS optimizer, anomaly detector, data-generation scripts, NYISO price profile, and instructions for reproducing the experiments.

After acceptance, replace the anonymous review link with the permanent GitHub/Zenodo DOI link.

## Citation

Please cite the associated paper if you use this code. A `CITATION.cff` template is included and should be updated with the final paper title, author list, venue, year, and DOI when available.
