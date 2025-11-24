# DCGAN with Modular PyTorch Components

This repository is an adaptation of the [PyTorch DCGAN tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) that keeps the modeling ideas intact while moving the implementation out of a single notebook and into testable, reusable Python modules. The notebook `train_run.ipynb` demonstrates the workflow end-to-end, while the actual training logic lives inside standard classes and helper functions.

## How This Project Differs from the Tutorial
- `config.py`, `model.py`, `dataset.py`, and `train.py` each encapsulate a single concern, so hyperparameters, model definitions, datasets, and training utilities can be re-used from scripts or notebooks alike.
- The notebook calls into these modules instead of redefining layers, transforms, or training loops inline, which keeps exploratory work focused on experiments instead of boilerplate.
- Utility helpers such as `utils.py::weights_init` and `train.train_one_epoch` make it easier to write tests or plug the components into other projects (e.g., CLI training scripts or services).

## Key Modeling Notes
- **Generator activation (`model.py:33`)** – The last layer is a `tanh`, which maps pixel intensities to `[-1, 1]`. This matches the tutorial’s recommendation and the pre-processing pipeline that normalizes real images into the same range. Without `tanh`, the generator’s outputs would not align with the discriminator’s input distribution, making convergence unstable.
- **Weight initialization (`utils.py:4`)** – Both generator and discriminator layers are initialized with a zero-mean normal distribution (std `0.02`), while batch norm gains are centered around `1.0`. Initializing weights this way prevents the discriminator from overpowering the generator early on and mirrors the heuristic used in the original DCGAN paper.
- **Loss function (`train.py:34`)** – Training uses binary cross-entropy (`nn.BCELoss`) with real labels set to `1` and fake labels set to `0`. The discriminator maximizes `log D(x) + log (1 - D(G(z)))`, and the generator minimizes `log (1 - D(G(z)))` (equivalently maximizing `log D(G(z))`) by relabeling its outputs as real. Tracking `G_losses` and `D_losses` helps spot divergence or mode collapse.

## Repository Layout
- `config.py` – Simple dataclass-style container for paths, optimizer settings, latent size, etc. Update `dataroot` before training.
- `dataset.py` – A `torch.utils.data.Dataset` that glob-loads JPEG faces and applies any torchvision transforms passed from the notebook or a script.
- `model.py` – Generator and discriminator modules mirroring the tutorial’s topology with transposed convolutions and batch norm.
- `utils.py` – Contains the custom weight initializer that is applied before training starts.
- `train.py` – Training loop split into `train_one_epoch` (single pass with loss bookkeeping) and `train` (epoch orchestration plus visualization hooks).
- `train_run.ipynb` – The notebook used to showcase data loading, training, and visualization while relying on the reusable modules above.

## Getting Started
1. Create the environment defined in `environment.yml`, or install PyTorch, torchvision, matplotlib, pandas, and Pillow manually.
2. Place your dataset under the folder referenced by `Config.dataroot` (default points to a CelebA extract).
3. From a terminal run:
   ```bash
   python train.py
   ```
   The notebook can also be opened to run the same training helpers interactively.

Once the training artifacts look good, initialize a Git repository, add a remote on GitHub, and push this codebase:
```bash
git init
git add .
git commit -m "Initial DCGAN project import"
git remote add origin git@github.com:<username>/<repo>.git
git push -u origin main
```

