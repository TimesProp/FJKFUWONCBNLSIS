Reproducibility Package

This repository contains a standalone implementation of the model and the code required to run the experiments in the submitted paper.

Only the necessary model, training, evaluation, and data-processing components are included.

File Structure
project/
├── datasets/            # Datasets and hierarchy information
├── model.py             # Model implementation
├── backbone.py          # Forecasting backbone and auxiliary losses
├── layers/              # Required neural network layers
├── train.py             # Training and evaluation
├── requirements.txt
├── LICENSE
└── .gitignore
Installation

The code is designed for Python 3.11.

Install the required packages with:

python -m pip install -r requirements.txt
Running the Code

Run a small test to verify the environment:

python train.py --smoke-test --device cpu

Run the default experiment:

python train.py

Available command-line options can be viewed with:

python train.py --help

CUDA is used automatically when available. Otherwise, the code runs on CPU.

Training and Evaluation

The training script includes:

data loading and preprocessing;
model training;
validation and early stopping;
checkpoint saving and loading;
test-set evaluation;
repeated experiments with multiple random seeds;
aggregation of evaluation results.

The best checkpoint from each run is reloaded before test evaluation.

Experimental results are saved in the output directory in machine-readable format.

Smoke Test

The smoke test runs a reduced version of the complete pipeline and is intended only to verify that the code and environment are working correctly.

It should not be used to reproduce the experimental results reported in the paper.

Data

The required datasets and hierarchy information are included in the datasets/ directory.

Additional information about the data is provided in:

datasets/README.md
Notes

The repository is self-contained and does not require source files outside this directory.

Training outputs, cached files, and pretrained checkpoints are not included.
