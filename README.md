# Rényi-based concentration metrics for Kolmogorov–Arnold geometry in MLPs

Code and data for the paper

> **A Class of Rényi-based Concentration Metrics for Kolmogorov–Arnold Geometry in Multilayer Perceptrons**

The repository reproduces every experimental result of the paper (Section 4, *Computational experiments*):
Table 2, Table 3, Figure 1, Figure 2 and all the numbers quoted in the text. Sections 2 and 3 are
theoretical and their examples are worked out by hand, so they need no code.

## Experiment in brief

- Target: continuous XOR $f(x) = \prod_{i=1}^{3} \sin(\pi x_i)$ on $[-1, 1]^3$, with 4096 training and 1024 test points per seed.
- Model: single-hidden-layer MLP $B\,\sigma(Ax + a) + b$ with the exact GeLU, widths $m \in \{8, 32\}$, orthogonal initialization.
- Training: Adam, learning rate 0.01, mini-batches of 250, 500 epochs, seeds 0 to 3. That gives 8 networks, measured at 501 states each (epoch 0 is the initialization).
- Metrics: pointwise $\alpha$-RC metrics $C^{(k)}_\alpha(x)$ for $k \in \{1, 2, 3\}$ and $\alpha \in \{1/2, 1, 2, \infty\}$, computed with every Jacobian minor enumerated, in float64 and in log scale. The Participation Ratio is obtained from $C_2$.

## Contents

```
data/runs/m{8,32}_seed{0..3}/   datasets, weights of every epoch and predictions (81 MB in total)
training/                       training code, exactly as executed (see below)
analysis/rc_metrics.py          alpha-RC metrics of a GeLU MLP from its weights
analysis/compute_concentrations.py   C^(k)_alpha(x) for every network, epoch, point, k and alpha
analysis/make_paper_assets.py   tables, figures and numbers of Section 4
results/                        the outputs used in the paper
```

| Paper element | Produced by | File |
|---|---|---|
| Table 2 ($R^2$ at epochs 0 and 500) | `analysis/make_paper_assets.py` | `results/tables/table_R2.tex` |
| Table 3 ($C^{(k)}_\alpha$ and PR at $x^\ast$ and pooled) | `analysis/make_paper_assets.py` | `results/tables/table_C_example.tex` |
| Figure 1 (mean $C^{(3)}_\alpha$ during training, $m = 32$) | `analysis/make_paper_assets.py` | `results/figures/C_mean_epochs_m32_k3.pdf` |
| Figure 2 (test $R^2$ coloured by $C^{(3)}_\alpha$, variance halos) | `analysis/make_paper_assets.py` | `results/figures/R2_magma_m32_k3.pdf` |
| Numbers quoted in Section 4 | `analysis/make_paper_assets.py` | `results/paper_numbers.json` |
| Pointwise $C^{(k)}_\alpha(x)$, all networks, epochs and points | `analysis/compute_concentrations.py` | release asset `C_m{w}_seed{s}.npy` |

## Reproducing the results

Python 3.10 or later. The analysis only needs NumPy, SciPy and Matplotlib and runs on a CPU.

```bash
pip install -r requirements.txt
```

**Option 1: from the stored weights (no training).** Recompute every pointwise value of $C^{(k)}_\alpha(x)$
(about 30 minutes with 8 CPU workers, 2 GB written to `outputs/C/`), then build the tables and figures:

```bash
python analysis/compute_concentrations.py --workers 8
```

```bash
python analysis/make_paper_assets.py
```

**Option 2: from the released values.** The 246,251,520 pointwise values (8 files of 246 MB, float64)
are attached to the GitHub release of this repository. Download them to a folder and pass it with `--c-dir`:

```bash
python analysis/make_paper_assets.py --c-dir path/to/release_files
```

**Option 3: retraining.** `training/run_study.py` trains one network and stores its dataset, weights and
predictions at every epoch. It needs PyTorch and a CUDA GPU (about 40 s per network on an NVIDIA B200):

```bash
pip install -r requirements-training.txt
```

```bash
python training/run_study.py --width 32 --seed 0 --epochs 500 --output runs/m32_seed0
```

The networks of the paper were trained with `--width {8,32} --seed {0,1,2,3}`. Bitwise identity of the
retrained weights is only expected on the same hardware and software (see `training_config.json` of each run).

### Checks

- `compute_concentrations.py` checks at every point that $0 \le C_{1/2} \le C_1 \le C_2 \le C_\infty \le 1$, as required by the monotonicity of the class in $\alpha$.
- `make_paper_assets.py` checks the identity $\mathrm{PR}^{(k)} = N_k^{(1 - C_2)/2}$ against the PR computed from explicit $k \times k$ determinants of the Jacobian, rebuilt from the stored weights (maximum relative error $1.2 \cdot 10^{-14}$, stored in `paper_numbers.json`).
- Running Option 1 from scratch reproduces the released values to within $6 \cdot 10^{-14}$. The tables and figures it produces are byte-identical to those in `results/`, which are the ones in the paper, and the numbers agree to within $4 \cdot 10^{-15}$.

## Data format

Each `data/runs/m{w}_seed{s}/` folder contains:

| File | Content |
|---|---|
| `dataset.npz` | `x` (5120, 3), `y` (5120,), `sample_id` (5120,), `split` (5120,), with 0 = train and 1 = test. Ids 0 to 4095 are the training points and 4096 to 5119 the test points. |
| `parameters.npy` | float32, shape (501, P), with all the weights at epochs 0 to 500. P = 41 for $m = 8$ and 161 for $m = 32$. |
| `parameter_layout.json` | position and shape of `fc1.weight` ($A$), `fc1.bias` ($a$), `fc2.weight` ($B$) and `fc2.bias` ($b$) within each row of `parameters.npy` |
| `predictions.npy` | float32, shape (501, 5120), with the network output at every epoch and point |
| `training_config.json` | hyperparameters, seeds and software of the run |

For a given seed both widths share the same points and mini-batch order. The pointwise metrics are stored as
`C_m{w}_seed{s}.npy`, float64, shape (501, 5120, 3, 4), indexed `C[epoch, sample_id, k - 1, alpha_index]`
with `alpha_index` 0, 1, 2, 3 for $\alpha$ = 1/2, 1, 2, $\infty$. They can be read without loading them in memory:

```python
import numpy as np
C = np.load("C_m32_seed0.npy", mmap_mode="r")
C[500, 4096:, 2, 1]   # C^(3)_1 at the 1024 test points, end of training
```

## About the training code

`training/run_study.py`, `training/exact_metrics.py` and `training/run_experiments_original.py` are the
files executed to train the networks, unchanged (`training/SHA256SUMS.txt`). `run_experiments_original.py`
provides the dataset, the model and the seeding. While training, `run_study.py` also writes GPU
concentration values to `C.npy` for the orders $\alpha$ used at that stage of the project. Those values
are not used in the paper. All the reported metrics are computed by `analysis/` from the stored weights.

## Contact

- **Juan Carlos Fernández** — [juancarlosfdezmlg@gmail.com](mailto:juancarlosfdezmlg@gmail.com).
  University of Málaga, Málaga, Spain.
- **Asunción Jiménez-Cordero** — [asuncionjc@uma.es](mailto:asuncionjc@uma.es).
  Department of Mathematical Analysis, Statistics and Operations Research and Applied Mathematics,
  Area of Statistics and Operations Research, University of Málaga, Málaga, Spain.
  OASYS research group, University of Málaga, Málaga, Spain.
- **José Francisco Paris** — [jfparis@uma.es](mailto:jfparis@uma.es).
  TELMA Institute, University of Málaga, Málaga, Spain.

## Citation

If you use this code or data, please cite the paper (reference to be completed upon publication).

## License

Code under the MIT License (see `LICENSE`). Data and results under CC BY 4.0.
