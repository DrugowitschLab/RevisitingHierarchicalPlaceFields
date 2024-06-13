from hierarchical_fisher import *
from multiprocessing import Pool

def run_sim(ar):
    runner, args, kwargs, save = ar
    runner.run(*args, sim_kwargs=kwargs, save_fname=save)

if __name__ == '__main__':
    aps_runner_gamma_unbound_ineff = APSRunner()
    aps_runner_gamma_bound_ineff = APSRunner()

    n_samples = 10
    n_neurons = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]

    width_samples = 32
    power_limit = 5
    widths = 2 ** np.linspace(0, -power_limit, num=width_samples)

    args = (n_samples, n_neurons, widths)
    sims = [(aps_runner_gamma_bound_ineff, args, \
             {'shape' : 2.5}, \
             'gamma_sims_full_2p5_ineff_super_small_sample')]

    with Pool(len(sims)) as p:
        p.map(run_sim, sims)
