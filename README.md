# tep-fault-diagnosis

This repo is me trying a few different model types on the Tennessee Eastman Process fault diagnosis dataset and seeing what actually works.

Right now the main comparison is:
- a simple MLP baseline
- a 1D CNN for time-series windows
- a GNN using a process graph

The whole point is to keep the setup the same and compare the models fairly instead of just assuming the graph model should be better.

## What this project does
The raw TEP data is turned into sliding windows.

Each window is:
- `60` time steps
- `52` process variables
- step size `10`

That gives me two dataset formats:

1. Flattened windows for the MLP
- shape: `(N, 3120)`
- because `60 * 52 = 3120`

2. 2D windows for the CNN and GNN
- shape: `(N, 60, 52)`

The labels are fault numbers `0..20`.

## Raw data columns
The raw CSVs have:
- `faultNumber`
- `simulationRun`
- `sample`
- `xmeas_1..41`
- `xmv_1..11`

So there are `52` total variables.

## Repo layout
Main scripts:
- `src/build_small_dataset.py`
- `src/build_small_dataset_2d.py`
- `src/train_baseline.py`
- `src/train_cnn1d.py`
- `src/train_gnn.py`
- `src/evaluate_model.py`

Model files:
- `src/models/simplemlp.py`
- `src/models/cnn1d.py`
- `src/models/gnn_model.py`

## Build datasets
Both dataset builders take fault and run ranges from the command line.

Build the flattened dataset for the MLP:

```bash
python src/build_small_dataset.py --fault-start 0 --fault-end 20 --run-start 1 --run-end 20
```

That gives a file like:

```bash
data/processed/small_fault0_20_runs1_20_W60_step10.npz
```

Build the 2D dataset for the CNN / GNN:

```bash
python src/build_small_dataset_2d.py --fault-start 0 --fault-end 20 --run-start 1 --run-end 20
```

That gives a file like:

```bash
data/processed/small_fault0_20_runs1_20_W60_step10_2d.npz
```

## Train models
MLP:

```bash
python src/train_baseline.py
```

CNN:

```bash
python src/train_cnn1d.py
```

GNN:

```bash
python src/train_gnn.py
```

## Evaluate models
Use:

```bash
python src/evaluate_model.py --model mlp
python src/evaluate_model.py --model cnn
python src/evaluate_model.py --model gnn
```

The evaluation script prints:
- top-1 / top-2 / top-3 accuracy
- per-fault accuracy
- confusion matrix

It also saves the confusion matrix to CSV.

## Split logic
The train/validation split is done by `simulationRun`, not by random windows.

That matters because windows from the same run are strongly related, and I do not want leakage between train and validation.

The training scripts also handle smaller datasets now. So if I switch from a large run range to something like `1..20`, the validation split adjusts instead of breaking.

## Current model setup
### MLP
This is the baseline.
It takes flattened windows and gives me a simple reference point for everything else.

### CNN
This is the time-series model.
It takes `(60, 52)` windows, converts them to channel-first format, and runs 1D convolutions over time.

### GNN
This is the graph model.
Right now it uses a spatiotemporal graph where:
- each node is a `(sensor, time step)` pair
- node features are scalar values
- process edges come from a fixed `EDGES_75` list
- temporal edges connect the same sensor forward in time

This is easily the hardest model in the repo to get right, and right now it is also the weakest one.

## Notes
- The checkpoint files (`.pt`) are just local training artifacts. (ignored by gitignore)
- The processed `.npz` datasets are generated files. (ignored by gitignore)
- The current setup is mainly for comparing model behavior, not pretending the graph model is automatically the best just because it is more complicated.

## TLDR
Fault diagnosis on the Tennessee Eastman Process using MLP, CNN, and graph neural network models.
