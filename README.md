# lya-mfdm

Code and data for Lyman-alpha forest constraints on pure and mixed fuzzy dark matter.

This repository contains some of the main code, data, trained emulators, and supplementary results used in the MFDM analysis of [Liu, Jianxiang et al. (2026)](https://arxiv.org/abs/2606.06969).

## Repository structure

### `data/`

The `data/` directory contains the processed 1D Lyman-alpha forest flux power spectra used for emulator training, together with the observational data used in the analysis.

The simulated 1D flux power spectra were generated with [`fake_spectra`](https://github.com/sbird/fake_spectra) from hydrodynamical simulations produced using [`MP-Gadget`](https://github.com/MP-Gadget/MP-Gadget/tree/master).

The observational data are taken from [Boera et al. (2019)](https://iopscience.iop.org/article/10.3847/1538-4357/aafee4).

### `emu/`

The `emu/` directory contains the trained emulators used in the analysis.

### `plots/`

The `plots/` directory contains the full corner plots and supplementary results.

In particular:

- `plots/mock/` contains the full corner plots from the 10 mock MCMC tests.
- `plots/mcmc/` contains the full corner plots from the MCMC analyses using the real data.
- `compare_CV_0_to_99_N80.pdf` shows the cross-validation results for all training data at fixed effective optical depth.
- `compare_10_validation_N100.pdf` shows the validation results for the independent validation set at fixed effective optical depth.
- `box_and_resolution.pdf` shows the convergence tests for the simulation box size and mass resolution.
- `corner_cut_with_patchy_fixed_f_95.pdf` shows posterior distributions and 95% credible levels.
- `correction.pdf` shows the amplitudes of several additional corrections applied in the analysis.
- `neyman_xxx.pdf` shows the 95% confidence intervals obtained using the Neyman construction with a profile-likelihood test statistic.
- `tgu_test.pdf` shows tests of the radial basis function interpolator used for mapping the thermal history parameters.
- `comparison_drop_all.pdf` shows the effect of removing small-scale data points from the analysis.

## Scripts

This repository also includes several Python scripts used in the analysis.

