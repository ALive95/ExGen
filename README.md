# Neural ODEs: Numerical Experiments

Code accompanying the numerical section of *Exact interpolation and nonparametric
generalization rates for neural ODEs: a controllability perspective* (Alvarez-López,
Liverani, Zuazua).

The repository reproduces the figures and tables of Section 5:

1. **Partition figure** (Figure 3) — illustrates the histogram and Voronoi
   estimators on a 2D smooth target.
2. **Width sweep** (Section 5.1, Table 1, Figure 5) — saNODE test risk as a function
   of width $p$, compared against histogram and Voronoi nonparametric baselines, on
   smooth and Hölder-$1/2$ targets in $d \in \{3, 8\}$.
3. **Checkerboard sorting** (Section 5.2, Figures 6–7) — comparison of three
   architectures on a piecewise-constant target: saNODE (time-dependent),
   aNODE (autonomous, single hidden layer), and 2aNODE (autonomous, two hidden
   layers). All three are parameter-matched.

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install the appropriate `jaxlib` CUDA build separately following
the [JAX installation guide](https://github.com/google/jax#installation).

## Reproducing the experiments

```bash
python estimator_plot.py
python run_width_experiment.py
python run_checkerboard.py
```

The width and checkerboard experiments cache trained models and per-run results
under `saved_models/` and write figures to `figures/`. Re-running a script loads
cached results and skips training for completed entries.

The width experiment uses 3 seeds per (target, $N$, $p$) and the checkerboard
experiment uses 3 seeds per ($K$, architecture). On a single CPU, each experiment
takes several hours; both can be parallelised across seeds by editing the seed
list in the corresponding script.

## Repository layout

```
models/
  sanode.py          Supervised (time-dependent) Neural ODE
  anode.py           Autonomous Neural ODE, single hidden layer
  twoanode.py        Autonomous Neural ODE, two hidden layers (2aNODE)
utils/
  targets.py         Smooth and Hölder target functions (JAX + NumPy)
  estimators.py      Histogram and Voronoi estimator errors
  training.py        Training and evaluation loops
  plotting.py        Figure generation
estimator_plot.py    Standalone script for the partition figure (Figure 3)
run_width_experiment.py
run_checkerboard.py
```

## Reproducibility

All experiments use explicit JAX PRNG keys derived from a single integer seed.
Setting the same seed guarantees bit-identical results on the same hardware and
JAX version.
