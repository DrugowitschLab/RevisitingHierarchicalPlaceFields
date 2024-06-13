# RevisitingHierarchicalPlaceFields
Code used to generate figures in "Revisiting efficient representations of space in hierarchical place field populations"

All code uses Python 3.8.10.

Code to simulate populations and compute Fisher information can be found in `src/hierarchical_fisher.py`. This code also contains an object `FisherTheory`, which provides analytical expressions for the discriminability under various conditions. 

Use `src/mp_simrunner.py` to run simulations. This runner natively spawns a separate process to run simulations for populations instantiated with different parameters to improve runtime. See `src/hierarchical_fisher.py` for more details on how to run simulations. The notebook that computes the discriminability over the simulated populations uses files saved from `mp_simrunner.py`. If interested in recreating any of the findings, swap the file names used in the notebooks with the ones you generate from running `mp_simrunner.py`.

Place field analysis is run using data from [Rich et al. 2014](https://www.science.org/doi/10.1126/science.1255635), data available [here](https://crcns.org/data-sets/hc/hc-31). All analysis code is in the respective notebook.
