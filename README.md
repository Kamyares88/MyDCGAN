# DCGAN Modularization Project

This repository shows how I refactored the [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) into conventional Python modules plus a single demonstration notebook. Rather than defining everything inline inside a notebook, I separated the configuration, dataset, models, utils, and training helpers so they can be reused or tested like any other package code. The notebook `train_run.ipynb` is the sole entry point for running experiments—`train.py` only exists so that the training helpers can be imported, not executed directly.

## What Makes This Different from the Tutorial
- Each component lives in its own file: `config.py`, `dataset.py`, `model.py`, `utils.py`, and `train.py`. That structure makes it easier to unit test, swap datasets, or move the training loop into another project.
- The notebook just orchestrates these modules, which keeps the experimental narrative clean while still giving the flexibility of Python functions/classes.
- The code mirrors the DCGAN architecture from the tutorial, but it is organized in a way that feels closer to production code—standard modules, reusable classes, and helper utilities.

## Modeling Notes I Care About
- **Generator output (`model.py:33`)** — The final activation is `tanh` so images are emitted in the `[-1, 1]` range, matching the normalization applied to the real dataset. This alignment keeps the discriminator from trivially rejecting synthetic samples.
- **Weight initialization (`utils.py:4`)** — Every conv layer starts with a zero-mean, `0.02` std normal distribution, while batch norm scales start around `1.0`. This follows the DCGAN paper’s heuristic and prevents the discriminator from immediately saturating.
- **Loss function (`train.py:34`)** — I stay with `nn.BCELoss` and the classic GAN objective (`log D(x) + log(1 - D(G(z)))`). The training loop tracks generator and discriminator losses so it’s easy to spot divergence or mode collapse as the notebook runs.

## Repository Guide
- `config.py` — Hyperparameters, dataset paths, optimizer settings, GPU count, etc.
- `dataset.py` — A simple `torch.utils.data.Dataset` that glob-loads JPEG faces and applies transforms passed in from the notebook.
- `model.py` — Generator and discriminator modules lifted from the tutorial, implemented as standard `nn.Module` classes.
- `utils.py` — Contains the custom weight initialization routine.
- `train.py` — Houses `train_one_epoch` and `train` helper functions. Do **not** run this file directly; import these helpers inside `train_run.ipynb`.
- `train_run.ipynb` — The one place to interact with the project. Open it, configure transforms/dataloaders, and call the helpers above to train and visualize progress.
- `environment.yml` — Optional Conda environment spec for reproducing my setup.

## How to Use the Notebook
1. Create an environment with PyTorch, torchvision, matplotlib, pandas, and Pillow (the provided `environment.yml` works with Conda or Mamba).
2. Update `Config.dataroot` so it points to your local CelebA (or similar) dataset path.
3. Launch Jupyter / VS Code / Colab with `train_run.ipynb`.
4. Follow the notebook cells to instantiate the config, dataset, data loader, models, and training loop. All heavy lifting happens through the modules described earlier.

Feel free to fork this repo and adapt the modules for other GAN experiments—the notebook is there purely to showcase the flow end-to-end.
