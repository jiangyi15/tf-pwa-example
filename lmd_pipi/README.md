Example of $\Lambda_c^{+}\rightarrow \Lambda \pi^{+} \pi^{0}, \Lambda \rightarrow p \pi^{-}$
============================================================================================

Model for [JHEP12(2022)033](https://doi.org/10.1007/JHEP12%282022%29033) and preprint [arxiv:2209.08464](https://arxiv.org/abs/2209.08464)

The precision of parameters keep the same as [the paper](https://doi.org/10.1007/JHEP12%282022%29033), so some other values (such as fit fractions) would be a little different.

# Basic imformation file
## config.yml

The configuration file. The same model as [the paper](https://doi.org/10.1007/JHEP12%282022%29033).

## Resonances.yml

Resonances used in the `config.yml`

## gen_params.json

The fit parameters from [the paper](https://doi.org/10.1007/JHEP12%282022%29033) Table 4 and Table 5.

# Simple Scripts
## gen_toy.py

Generate toy with the same number of events as [the paper](https://doi.org/10.1007/JHEP12%282022%29033) Table 3.

## fit.py

Simple fit script to fit the data. This script will produce `final_params.json` for fit parameters, `error_matrix.npy` for the error matrix and some plots of the distribution projections.

## plot.py 

Plot the distribution of masses and angles projection from fit results.

## cal_fitfrac.py

Calculate the fit fractions from fit results.

## cal_asym.py

Calculate the asymmetry paramters from fit results.
