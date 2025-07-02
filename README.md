# JTAP: Joint Physical Tracking and Prediction


`jtap` is a Python library for probabilistic psychophysical modeling and inference of 2D Mental Physics. Project page: https://arijit-dasgupta.github.io/jtap/


## Installation

#### Requirements

To run `jtap`, you must use a machine with at least a single NVIDIA GPU (minimum 24GB Memory) and have CUDA 12 (with the appropriate driver version).

#### Setup

Follow the following code block to get `jtap` running on your machine.

```bash
conda create -n jtap python=3.11
conda activate jtap
pip install -r requirements.txt
pip install -e .
```

## Reproducing CogSci 2025 Results
To reproduce the results from the paper, run 

```bash
python run_jtap_cogsci2025.py
```
Since JTAP is probabilistic, results might vary slightly, but draw the same conclusions as the paper. Running the experiment to reproduce the paper results make take 10-15 minutes.

Three pdfs will be generated when running the experiment, which are identically formatted to the figures shown in the paper.