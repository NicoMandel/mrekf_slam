# KISS - Keep It Static SLAMMOT

This is the repository to accompany code for the paper titled `KISS - Keep-It-Static-SLAMMOT`, which integrates moving landmarks into an EKF-SLAM algorithm.
Think of this analogy: you are on a ship and want to navigate at night. There is a cliff in front of you, with one lighthouse and a single car parked on the cliff. In traditional SLAM algorithms, you would discard the car as a moving object. However, your information would be incomplete and you would be unable to uniquely determine your position. Even though the car moves, at that instant it yields information which may contribute to the localisation of your ship. This is what this repository does.
![A cliff at night with a lighthouse and a car parked on the cliff. A ship in front of it.](./docs/20240703_GraphAbs.png)


## Installation
This package relies on the [robotics-toolbox](https://github.com/petercorke/robotics-toolbox-python) as a basis.
Please install the conda environment from `conda env create -f environment.yml`, activate the environment with `conda activate mrekf` and then pip-install the files locally editable through `pip install -e .`
For further reference, see the [Good Research Code Handbook](https://goodresearch.dev/index.html)

## Usage
All executable scripts are in the [scripts](./scripts/) subfolder and make use of source files in [src](./src/mrekf/). To run a script, please activate the conda environment first with `conda activate mrekf`.
[`arguments.py`](./scripts/arguments.py) is the main script, which will execute using [hydra](https://hydra.cc/). The config files for hydra can be found in the [config folder](./config/)

## Data
Data files are available on the [repository of the University of Lubeck](https://srv01.rob.uni-luebeck.de/~mandel/downloads/)
* `submisson_results.xlsx` is the spreadsheet for analysis with [`eval_csv.py`](./scripts/eval_csv.py)
* `histories.zip` includes the Groundtruth histories as .pkl files for 1996 cases included in the evaluation and can be used to re-run all filters deterministically. [`plot_case.py`](./scripts/plot_case.py) can be used to plot single case results.

## Feedback
please use the issues of this repository. or provide direct feedback to nicolas.mandel@uni-luebeck.de 