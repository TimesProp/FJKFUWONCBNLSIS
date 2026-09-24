# Reproducibility Code

This repository contains the code used for the experiments in the submitted paper.

## File Structure

```text
project/
├── datasets/          # Datasets and hierarchy information
├── model.py           # Model implementation
├── backbone.py        # Forecasting backbone
├── layers/            # Required model layers
├── train.py           # Training and evaluation script
├── requirements.txt   # Python dependencies
├── LICENSE
└── .gitignore
```

## Installation

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running

Run the experiment with:

```bash
python train.py
```

Available options can be viewed with:

```bash
python train.py --help
```

Different datasets or experimental settings can be specified through command-line arguments.

The training results and model checkpoints are saved to the output directory.
