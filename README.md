# Stochastic Deep Gaussian Processes over Graphs
*code and results for a NeurIPS2020 paper submission*


## Prerequests
our implementation is mainly based on following packages:

```
python 3.7
pip install keras==2.3.1
pip install gpuinfo
pip install tensorflow-gpu==1.15
pip install gpflow==1.5
```
Besides, some basic packages like `numpy` are also needed.

## Specification

Source code and experiment result are both provided.

### Files

- `src/`
	- `doubly_stochastic_dgp`: codes from repository [DGP](https://github.com/ICL-SML/Doubly-Stochastic-DGP)
	- `compatible`: codes to make the DGP source codes compatible with gpflow1.5.
	- `gpflow_monitor`: monitoring tool for gpflow models, from [this repo](https://github.com/markvdw/gpflow-monitor).
	- `data`: datasets.
	- `dgp_graph`: implemetation of our model.
	- `*.ipynb`: jupyter notebooks for experiments.
	- `run_toy.sh`: shell script to run additional experiment.
	- `toy_main.py`: code for additional experiment (Traditional ML methods and DGPG with linear kernel).
- `results/`: contains results of experiments
	- `*.html`: experiment results demonstrated by static HTML files.

### Experiments
The experiments are demonstrated by jupyter notebooks. The source is under directory `src/` and the corresponding result is exported as a static HTML file stored in the directory `results/`. They are organized by dataset names:

1. Synthetic Datasets
	- `demo_toy_run1.ipynb`
	- `demo_toy_run2.ipynb`
	- `demo_toy_run3.ipynb`
	- `demo_toy_run4.ipynb`
	- `demo_toy_run5.ipynb`
2. Small Datasets
	- `demo_city45.ipynb`
	- `demo_city45_linear.ipynb` (linear kernel)
	- `demo_city45_baseline.ipynb` (traditional regression methods)
	- `demo_etex.ipynb`
	- `demo_etex_linear.ipynb`
	- `demo_etex_baseline.ipynb`
	- `demo_fmri.ipynb`
	- `demo_fmri_linear.ipynb`
	- `demo_fmri_baseline.ipynb`
3. Large Datasets (traffic flow)
	- LA
		- `demo_la_15min.ipynb`
		- `demo_la_30min.ipynb`
		- `demo_la_60min.ipynb`
 	- BAY
		- `demo_bay_15min.ipynb`
		- `demo_bay_30min.ipynb`
		- `demo_bay_60min.ipynb`
