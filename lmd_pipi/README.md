Example of $Lambda_c^{+}\rightarrow \Lambda \pi^{+} \pi^{0}, \Lambda \rightarrow p \pi^{-}$
------------------------------------------------------------

Model for [JHEP12(2022)033](https://doi.org/10.1007/JHEP12%282022%29033) [arxiv:2209.08464](https://arxiv.org/abs/2209.08464)

The precision of paramters keep the same as [JHEP12(2022)033](https://doi.org/10.1007/JHEP12%282022%29033), so some other values would be a litte different.

# Basic imformation file
## config.yml

The configuration file. The same model as [paper](https://doi.org/10.1007/JHEP12%282022%29033).

## Resonances.yml

Resonances used in the `config.yml`

## gen_params.json

The fit parameters from the [paper](https://doi.org/10.1007/JHEP12%282022%29033)

# Simple Scripts
## gen_toy.py

Generate toy with the same number of events as in [paper](https://doi.org/10.1007/JHEP12%282022%29033). 

## fit.py

Simple fit script to fit the data.

## plot.py 

Plot the distribution of masses and angles from fit results.

## cal_fitfrac.py

Calculate the fit fractions from fit results.

## cal_asym.py

Calculate the asymmetry paramters from fit results.