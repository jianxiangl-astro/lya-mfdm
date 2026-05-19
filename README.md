# lya-mfdm

Code and data for Lyman-alpha forest constraints on pure and mixed fuzzy dark matter.

This repository contains some of the main code, data, trained emulators, and supplementary results used in the MFDM analysis of Liu, Jianxiang et al. (2026).

The `data/` directory contains the processed 1D flux power spectra used for emulator training, together with the observational data.

The `emu/` directory contains the trained emulators.

The `plots/` directory contains the full corner plots from the mock MCMC tests and from the MCMC analyses using the real data. It also includes several additional reference results. For example, `compare_CV_0_to_99_N80.pdf` shows the cross-validation results for all training data at fixed effective optical depth, while `compare_10_validation_N100.pdf` shows the validation results for the independent validation set at fixed effective optical depth.
