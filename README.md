# EUSIPCO-2025-CARMA

This provides an implementation of the algorithm proposed in our paper:
> R.J McDougall and S.J. Godsill, "On the Construction of Discrete-Time ARMA Models from Continuous-Time State-Space Models".

This will be presented at the [European Signal Processing Conference (EUSIPCO)](https://eusipco2025.org/) in Palermo, Italy in September 2025.

If you would like to use or copy any of the code provided here, please cite the original paper.

## Reproducing results
The code and environment configuration required to reproduce our results can be found in the branch `reproduce-paper`.
The necessary script can be run in the following ways:
1.  Navigate to your local folder containing this repository and start a Julia REPL session using `> julia` in the terminal.
2.  Activate the project environment through: `using Pkg; Pkg.activate("."); Pkg.instantiate();`.
3.  Run the experiments: `include("scripts/run_experiments.jl")`

All generated figures will be in the `figs` folder.

## Project structure

All model conversion code is available in the `ModelConversions` package within `src/`.
