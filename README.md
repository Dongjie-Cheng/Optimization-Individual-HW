# Certifiably Fast Sparse Logistic Regression via Gap-Safe Screening and Adaptive Primal–Dual Splitting

Code for the experiments in:

> Certifiably Fast Sparse Logistic Regression via Gap-Safe Screening and Adaptive Primal–Dual Splitting

The repository implements certified solvers for L1-regularized logistic regression with a shared duality-gap certificate:

- CD-screen: coordinate descent with Strong Rules and Gap-Safe screening
- ADMM-adapt: ADMM with diagonal Newton steps and residual-balanced penalty updates

Both solvers use the same dual construction and are compared by time-to-epsilon.

------------------------------------------------------------

## Quick Start

### Clone

```bash
git clone https://github.com/Dongjie-Cheng/Optimization-Individual-HW.git
cd Optimization-Individual-HW
```

###  Install dependencies
```bash
pip install -r requirements.txt
```

###  Run all experiments and plots
```bash
python experiment.py --dataset 20ng --data_root data/libsvm --eps 1e-3 --save_json results_20ng.json
python plots.py --summary results_20ng.json --outdir figs

python experiment.py --dataset ijcnn1 --data_root data/libsvm --eps 1e-3 --save_json results_ijcnn1.json
python plots.py --summary results_ijcnn1.json --outdir figs

python experiment.py --dataset real-sim --data_root data/libsvm --eps 1e-3 --save_json results_realsim.json
python plots.py --summary results_realsim.json --outdir figs
```

## Datasets

The project uses three public datasets:

- **ijcnn1** — dense small-scale tabular data (LIBSVM format)
- **20NG** — TF–IDF sparse text (downloaded automatically by the code)
- **real-sim** — TF–IDF sparse web text (LIBSVM format)

### Downloading ijcnn1 and real-sim (LIBSVM)

Both ijcnn1 and real-sim are available from the official LIBSVM data page:  
<https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/>

We use the LIBSVM-format files and expect them to be **unzipped** plain text
files under `data/libsvm/`:

```bash
mkdir -p data/libsvm
cd data/libsvm

# ijcnn1: training and test sets
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2

# real-sim: single split, train/test split is done in our code
wget https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/real-sim.bz2

# unzip
bunzip2 ijcnn1.tr.bz2
bunzip2 ijcnn1.t.bz2
bunzip2 real-sim.bz2

# rename ijcnn1.tr -> ijcnn1 to match the experiment script
mv ijcnn1.tr ijcnn1
```

------------------------------------------------------------

## Experiments

### 20ng
`python experiment.py --dataset 20ng --data_root data/libsvm --eps 1e-3 --save_json results_20ng.json`

`python plots.py --summary results_20ng.json     --outdir figs`

### ijcnn1
`python experiment.py --dataset ijcnn1 --data_root data/libsvm --eps 1e-3 --save_json results_ijcnn1.json`

`python plots.py --summary results_ijcnn1.json   --outdir figs`

### real-sim
`python experiment.py --dataset real-sim --data_root data/libsvm --eps 1e-3 --save_json results_realsim.json`

`python plots.py --summary results_realsim.json  --outdir figs`

------------------------------------------------------------

## License

MIT License

### Dataset licensing

This repository includes copies of third-party datasets (ijcnn1 and real-sim)
under `data/libsvm/` for convenience. All rights remain with the original
authors and providers.
