# Certifiably Fast Sparse Logistic Regression via Gap-Safe Screening and Adaptive Primal–Dual Splitting

Code for the experiments in:

> Certifiably Fast Sparse Logistic Regression via Gap-Safe Screening and Adaptive Primal–Dual Splitting

The repository implements certified solvers for L1-regularized logistic regression with a shared duality-gap certificate:

- CD-screen: coordinate descent with Strong Rules and Gap-Safe screening
- ADMM-adapt: ADMM with diagonal Newton steps and residual-balanced penalty updates

Both solvers use the same dual construction and are compared by time-to-epsilon.

------------------------------------------------------------

## Datasets

The project uses three public datasets:

- ijcnn1 — dense small-scale tabular data
- 20NG — TF–IDF sparse text
- real-sim — TF–IDF sparse web text

Place processed datasets under:

data/
    ijcnn1/
    20ng/
    real-sim/

------------------------------------------------------------

## Experiments

### a9a
python experiment.py --dataset a9a --data_root data/libsvm --eps 1e-3 --save_json results_a9a.json

### ijcnn1
python experiment.py --dataset ijcnn1 --data_root data/libsvm --eps 1e-3 --save_json results_ijcnn1.json

### real-sim
python experiment.py --dataset real-sim --data_root data/libsvm --eps 1e-3 --save_json results_realsim.json

------------------------------------------------------------

## License

MIT License
