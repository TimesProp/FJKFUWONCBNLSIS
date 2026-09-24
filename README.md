# Reproducibility Package

This repository contains a standalone and simplified implementation of the model used in the submitted paper. It includes only the model, the required forecasting backbone, the necessary layers, the datasets, and the training and evaluation code.

The directory can be copied and executed independently. It does not depend on any external project directory or additional local source files.

## File Structure

```text
mini/
├── datasets/            # Eight complete datasets, hierarchy matrices, and data validation files
├── model.py             # Model implementation
├── backbone.py          # Forecasting backbone and auxiliary hierarchical losses
├── layers/              # Required embedding and convolution layers
├── train.py             # Training, early stopping, evaluation, and multi-seed aggregation
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Installation and Execution

The code was tested with Python 3.11.

Enter this directory and install the required packages:

```bash
python -m pip install -r requirements.txt
```

A small CPU test can be used to verify the installation and the complete training pipeline:

```bash
python train.py --smoke-test --device cpu
```

To run the default experiment:

```bash
python train.py
```

The training script automatically uses CUDA when it is available and otherwise uses the CPU. GPU execution requires a compatible CUDA-enabled PyTorch installation and GPU driver.

The code is self-contained and does not depend on source files outside this directory.

By default, complete training uses three random seeds:

```text
42, 43, 44
```

Each run is trained for at most 50 epochs. Early stopping is applied when the validation loss does not improve for three consecutive epochs.

The default dataset is:

```text
mixedhumid_sfd
```

## Example Commands

Switch to another dataset:

```bash
python train.py --dataset cold_sfd
```

Run a single experiment with customized settings:

```bash
python train.py --runs 1 --epochs 50 --seq 24 --pred-len 24 --dh-steps 1 --seed 42
```

Specify a separate output directory:

```bash
python train.py --dataset marine_sfd --output-dir outputs/marine
```

Display all available command-line options:

```bash
python train.py --help
```

Dataset paths are always resolved relative to the location of `train.py`. Therefore, the training script can also be called from another working directory.

Relative paths provided through `--output-dir` are resolved from the current working directory. Without this option, results are saved to:

```text
outputs/
```

## Default Experimental Settings

| Setting                              | Default Value   |
| ------------------------------------ | --------------- |
| Input length / prediction length     | 24 / 24         |
| Train / validation / test split      | 70% / 10% / 20% |
| Adjacent temporal-scale factors      | `[1, 2, 3]`     |
| Cumulative temporal scales           | `[1, 2, 6]`     |
| Batch size                           | 64              |
| Learning rate                        | 0.001           |
| Main forecasting loss                | Huber loss      |
| Hierarchical consistency loss weight | 0.01            |
| Bottom-up auxiliary loss weight      | 0.01            |
| Backbone hidden dimension            | 16              |
| Feed-forward dimension               | 64              |
| Number of backbone layers            | 2               |
| Top-k setting                        | 5               |
| Number of convolution kernels        | 6               |
| Dropout                              | 0.1             |
| Hierarchical hidden dimension        | 16              |
| Hierarchical update steps            | 1               |
| Maximum residual gate value          | 0.5             |

The dataset is divided chronologically into training, validation, and test sets using a 70% / 10% / 20% split.

Normalization statistics are fitted using the training set only. Statistics required for temporal-scale processing are also computed only from the training data.

Validation and test samples may use historical observations before the corresponding split boundary as input, while all prediction targets remain strictly inside their assigned split.

Temporal-scale sequences are constructed using sum aggregation, following the experimental configuration used in the paper.

The input length and prediction length must both be divisible by 6.

## Training and Evaluation

For each run, the model with the best validation performance is saved as:

```text
outputs/*_best.pth
```

The best checkpoint is reloaded before evaluation on the test set.

The complete experimental results are stored in:

```text
outputs/results.json
```

The JSON file contains:

* the experimental configuration;
* the raw results from each random seed;
* forecasting metrics;
* hierarchical evaluation metrics at different temporal scales and structural levels.

The terminal also reports the mean and sample standard deviation across runs.

The sample standard deviation is computed using:

```text
ddof = 1
```

For a single run, the reported standard deviation is set to zero.

Running the same configuration with the same output directory will overwrite files with the same names.

## Smoke Test

The smoke test is intended only to verify the installation and execution pipeline:

```bash
python train.py --smoke-test --device cpu
```

It uses the complete training data to estimate normalization and temporal-scale statistics, but training, validation, and testing are each restricted to one batch containing at most two samples.

The smoke test uses:

```text
1 epoch
1 random seed
```

Its results are saved to:

```text
outputs/smoke/
```

The smoke test is intended only for checking code execution and must not be used as an experimental result.

## Validation

The following checks were performed before packaging this repository.

### Environment

The code was tested with:

```text
Python 3.11
torch 2.13.0+cu132
numpy 2.4.6
pandas 2.3.3
scikit-learn 1.7.2
```

### Implementation Consistency

The standalone implementation was compared with the implementation used for the experiments.

Under the same random seed, the following quantities were verified to be identical on CPU:

* parameter initialization;
* model state dictionaries;
* model predictions;
* total training loss;
* parameter gradients.

The comparisons were performed element by element.

### Dataset Validation

All eight packaged datasets were checked against the datasets used in the experiments.

The packaged files are byte-identical to the corresponding experimental files.

Additional checks confirmed that:

* all numerical values are finite;
* the hierarchy matrices are valid;
* the structural relationships form valid trees or forests.

### End-to-End Execution

A complete CPU smoke test was successfully executed.

A GPU test was also performed using the complete `mixedhumid_sfd` dataset. The test completed:

* one training epoch;
* validation;
* best-checkpoint loading;
* complete test-set evaluation.

The resulting normalized metrics were approximately:

```text
MSE: 0.090291
MAE: 0.215858
```

These values are reported only as an execution check and are not intended to reproduce the final results reported in the paper.

## Code Organization

`model.py` contains the model computation used in the experiments.

`backbone.py` contains the required forecasting backbone and the auxiliary hierarchical loss functions.

`layers/` contains only the embedding and convolution components required by the model.

`train.py` contains the complete training and evaluation pipeline, including:

* dataset loading;
* preprocessing;
* training;
* validation;
* early stopping;
* checkpoint saving and loading;
* test-set evaluation;
* multi-seed experiments;
* metric aggregation;
* JSON result export;
* input validation;
* smoke testing.

Code unrelated to the submitted model and experiments has been removed from this package.

Dataset information and release details are provided in:

```text
datasets/README.md
```

This repository does not contain trained checkpoints, cached files, generated experimental outputs, or implementations of additional model variants.
