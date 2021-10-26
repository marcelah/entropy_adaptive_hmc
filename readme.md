
# Entropy-based adaptive Hamiltonian Monte Carlo

This repository  contains code for the paper

M. Hirt, M. Titsias, P. Dellaportas: Entropy-based adaptive Hamiltonian Monte Carlo. Neural Information Processing Systems (NeurIPS), 2021.

The adaptation is illustrated in the notebook examples.ipynb with Gaussian targets.
Sampling while performing the adaptation is implemented in AdaptivePreconditionedHamiltonianMonteCarlo in adaptive_linear_preconditioned_hmc.py which is a subclass of the TFP class hmc.HamiltonianMonteCarlo.

Examples how to run the entropy-based adaptation for different targets are in jobscripts