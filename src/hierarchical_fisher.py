import numpy as np
import scipy as sp
import collections.abc
from scipy.stats import poisson
import matplotlib.pyplot as plt
import itertools
from functools import partial

class AnalyticPopulationSim(object):
    def __init__(self,
                 n_neurons,
                 beta,
                 n_samples,
                 stim_range=(0, 1),
                 allow_rotation=True,
                 center_range=None,
                 efficient=False,
                 fixed_tuning_curves=False,
                 shape=10,
                 anisotropic=True):
        self.n_neurons = n_neurons
        self.beta = beta
        self.n_samples = n_samples
        self.rot_prefactor = 1 if allow_rotation else 0
        self.s_min = stim_range[0]
        self.s_max = stim_range[1]
        self.efficient = efficient
        self.ftc = fixed_tuning_curves
        self.shape = shape
        self.anisotropic = anisotropic
        if center_range:
            self.c_min = center_range[0]
            self.c_max = center_range[1]
        else:
            self.c_min = self.s_min
            self.c_max = self.s_max

    def __setup(self):
        if self.ftc:
            self.widths = np.power(self.beta * np.ones((self.n_neurons, 2)), 2)
        else:
            if self.anisotropic:
                self.widths = np.power(
                    np.random.gamma(self.shape,
                                    scale=(self.beta / self.shape),
                                    size=(self.n_neurons, 2)), 2)
            else:
                self.widths = np.multiply(np.power(
                    np.random.gamma(self.shape,
                                    scale=(self.beta / self.shape),
                                    size=self.n_neurons), 2),
                                          np.ones((2, self.n_neurons))).T

        self.centers = np.random.uniform(
            low=self.c_min, high=self.c_max, size=(self.n_neurons, 2))
        self.thetas = self.rot_prefactor * \
            np.random.uniform(low=0, high=2 * np.pi, size=self.n_neurons)
        self.__build_covs()
        if self.efficient:
            self.__build_efficient_gains()
        else:
            self.gains = np.random.uniform(low=5, high=25, size=self.n_neurons)

    def __setup_vectorized(self, test_vectorized=False):
        if self.ftc:
            self.widths_v = np.power(
                self.beta * np.ones((self.n_neurons, self.n_samples, 2)),
            2)
        else:
            self.widths_v = np.power(
                np.random.gamma(self.shape,
                                scale=(self.beta / self.shape),
                                size=(self.n_neurons, self.n_samples, 2)), 2)
        self.centers_v = np.random.uniform(
            low=self.c_min,
            high=self.c_max,
            size=(self.n_neurons, self.n_samples, 2))
        self.thetas_v = self.rot_prefactor * \
            np.random.uniform(low=0,
                              high=2 * np.pi,
                              size=(self.n_neurons, self.n_samples))
        self.__build_covs_vectorized(check_vectorized=test_vectorized)
        if self.efficient:
            self.__build_efficient_gains_vectorized(
                test_vectorized=test_vectorized)
        else:
            self.gains_v = np.random.uniform(
                low=5,
                high=25,
                size=(self.n_neurons, self.n_samples))

    def __build_covs_vectorized(self, check_vectorized=False):
        cos_thetas = np.cos(self.thetas_v)
        sin_thetas = np.sin(self.thetas_v)
        rots = np.array([[cos_thetas, sin_thetas], [-sin_thetas, cos_thetas]])
        rots_T = np.transpose(rots, (1, 0, 2, 3))

        width_mats = np.zeros((self.n_neurons, self.n_samples, 2, 2))
        width_mats[:, :, [0, 1], [0, 1]] = self.widths_v

        self.sigmas_v = np.einsum('ijkl,kljp->klip',
                                  rots,
                                  np.einsum('klij,jpkl->klip',
                                            width_mats,
                                            rots_T))

        if check_vectorized:
            # we'll need the omegas later for the vectorization check,
            # so build them here
            self.omegas_v = np.linalg.inv(self.sigmas_v)
            for j in range(self.n_samples):
                for i in range(self.n_neurons):
                    t = self.thetas_v[i, j]
                    rot = np.array(
                        [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
                    sig_ij = rot @ np.diag(self.widths_v[i, j, :]) @ rot.T
                    assert np.allclose(sig_ij, self.sigmas_v[i, j, :, :])

    def __build_covs(self):
        self.sigmas = np.zeros((2, 2, self.n_neurons))
        self.omegas = np.zeros_like(self.sigmas)
        for i in range(self.n_neurons):
            t = self.thetas[i]
            rot = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            self.sigmas[:, :, i] = rot @ np.diag(self.widths[i]) @ rot.T
            self.omegas[:, :, i] = np.linalg.inv(self.sigmas[:, :, i])

    def __build_efficient_gains_vectorized(self, test_vectorized=False):
        self.gains_v = 1 / np.sqrt(np.linalg.det(self.sigmas_v))

        if test_vectorized:
            for j in range(self.n_samples):
                for i in range(self.n_neurons):
                    test = 1 / np.sqrt(np.linalg.det(self.sigmas_v[i, j, :, :]))
                    assert np.allclose(self.gains_v[i, j], test)

    def __build_efficient_gains(self):
        self.gains = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            self.gains[i] = 1 / np.sqrt(np.linalg.det(self.sigmas[:, :, i]))

    def f_idx(self, idx, off):
        return self.gains[idx] * \
            np.exp(-0.5 * off @ self.omegas[:, :, idx] @ off)

    def fish_neurons(self, s):
        analytic_fisher = 0 * np.eye(2)
        for i in range(self.n_neurons):
            omega_sample = self.omegas[:, :, i]
            off = s - self.centers[i]
            analytic_fisher += self.f_idx(i, off) * \
                np.outer(omega_sample @ off, off @ omega_sample)
        return analytic_fisher

    def fish_neurons_vectorized(self, stims, hold_sum=False):
        rep_stims = np.repeat([stims.T], self.n_neurons, axis=0)
        offs = rep_stims - self.centers_v

        if self.shape == 1:
            # In the exponential case, some numerical instability might cause
            # issues with np.linalg.solve(). Ordinarily, we would just use
            # .lstsq() on the same sigma_v tensor, but the expected matrix size
            # differs between .lstsq() and .solve(). This conditional check
            # tries to perform the vectorized .solve() first (as the cases of
            # instability are infrequent), and falls back to the less efficient
            # loop over .lstsq() if that doesn't work
            try:
                grad_s = np.linalg.solve(self.sigmas_v, offs)
            except:
                grad_s = np.array(
                    [ [ np.linalg.lstsq(self.sigmas_v[i, j, :, :],
                                      offs[i, j, :]) \
                      for i in range(self.n_neurons) ] \
                    for j in range(self.n_samples) ])
        else:
            grad_s = np.linalg.solve(self.sigmas_v, offs)

        grad_s_outers = np.einsum('ipj,ipk->ipjk', grad_s, grad_s)

        all_activations = np.multiply(
            self.gains_v, np.exp(-0.5 * np.einsum('ikj,ikj->ik', offs, grad_s)))
        all_fish = np.multiply(
            all_activations, np.transpose(grad_s_outers, (2, 3, 0, 1)))

        if hold_sum:
            return all_fish
        else:
            return np.sum(all_fish, axis=-2)

    def compute_fisher(self, test_vectorized=False):
        fishers = np.zeros((2, 2, self.n_samples))
        s = np.random.uniform(
            low=self.s_min, high=self.s_max, size=(2, self.n_samples))
        if test_vectorized:
            self.__setup_vectorized()
            vec_fishers = self.fish_neurons_vectorized(s)
        for i in range(self.n_samples):
            if test_vectorized:
                self.omegas = np.transpose(self.omegas_v[:, i, :, :], (1, 2, 0))
                self.centers = self.centers_v[:, i, :]
                self.gains = self.gains_v[:, i]
            else:
                self.__setup()
            stim = s[:, i]
            fishers[:, :, i] = self.fish_neurons(stim)
            if test_vectorized:
                test = self.fish_neurons_vectorized(stim)
                assert np.allclose(test, fishers[:,:,i])
                assert np.allclose(vec_fishers[:,:,i], fishers[:,:,i])

        return fishers

    def compute_fisher_vectorized(self, test_vectorized=False, hold_sum=False):
        s = np.random.uniform(
            low=self.s_min, high=self.s_max, size=(2, self.n_samples))
        self.__setup_vectorized(test_vectorized=test_vectorized)
        vec_fishers = self.fish_neurons_vectorized(s, hold_sum)

        if not test_vectorized:
            return vec_fishers

        fishers = np.zeros((2, 2, self.n_samples))
        for i in range(self.n_samples):
            self.omegas = np.transpose(self.omegas_v[:, i, :, :], (1, 2, 0))
            self.centers = self.centers_v[:, i, :]
            self.gains = self.gains_v[:, i]
            fishers[:, :, i] = self.fish_neurons_vectorized(s[:, i])
            assert(np.allclose(fishers[:, :, i], vec_fishers[:, :, i]))

        return fishers

class APSRunner(object):
    def run(self,
            n_samples,
            n_neurons,
            widths,
            stim_dim=2,
            sim_kwargs={},
            save_fname=None):
        res = np.zeros(
            (stim_dim, stim_dim, n_samples, len(widths), len(n_neurons)))
        for i, n_neuron in enumerate(n_neurons):
            for j, width in enumerate(widths):
                aps = AnalyticPopulationSim(
                    n_neuron, width, n_samples, **sim_kwargs)
                res[:, :, :, j, i] = aps.compute_fisher_vectorized()

        if save_fname:
            np.save(save_fname, res)
        return res

class FisherTheory(object):
    def __init__(self,
                 s_dim,
                 efficient=False,
                 bounded=False,
                 anisotropic=True,
                 shape=10,
                 bounded_kw={}):
        self.s_dim = s_dim
        self.efficient = efficient
        self.bounded = bounded
        self.shape = shape
        self.gain_mean = bounded_kw.get('gm', 15)
        self.s_min = bounded_kw.get('s_min', 0.0)
        self.s_max = bounded_kw.get('s_max', 1.0)
        self.rot = bounded_kw.get('rot', False)
        self.anisotropic = anisotropic
        if self.bounded:
            self.c_min = bounded_kw.get('c_min', 0.0)
            self.c_max = bounded_kw.get('c_max', 1.0)
            self.use_simplified = \
                (self.c_min == self.s_min and self.c_max == self.s_max)

            self.b_iters = bounded_kw.get('iters', 1000)
            self.stabilize = bounded_kw.get('stabilize', True)
            self.analytic_s_avg = bounded_kw.get('analytic_s', False)

    def __unbounded_fi_per_pparam(self, param, fixed_sigma):
        s_range = self.s_max - self.s_min
        K = np.power(2 * np.pi, self.s_dim / 2) * np.power(s_range, -self.s_dim)
        if self.efficient:
            if not fixed_sigma:
                K *= np.power(self.shape, 2) / \
                    ((self.shape - 1) * (self.shape - 2))
            return K * np.power(param, -2)
        else:
            K *= self.gain_mean

            if not fixed_sigma:
                if self.anisotropic:
                    K *= self.shape / (self.shape - 1)
                else:
                    K *= sp.special.gamma(self.shape + (self.s_dim - 2)) / \
                        (sp.special.gamma(self.shape) * \
                         np.power(self.shape, self.s_dim - 2))
            return K * np.power(param, self.s_dim - 2)

    def L_bounded_is(self, L, beta):
        scaled_L = L - self.s_min + self.s_max
        exp_diff = self.__expon(scaled_L, 0, beta) - self.__expon(L, 0, beta)
        erf_diff = scaled_L * sp.special.erf(scaled_L / (np.sqrt(2) * beta)) - \
            L * sp.special.erf(L / (np.sqrt(2) * beta))
        return np.sqrt(np.pi / 2) * ( \
            2 * beta * np.sqrt(np.pi / 2) * exp_diff + 2 * erf_diff
        ) * ( \
            4 * beta * exp_diff + np.sqrt(2 * np.pi) * erf_diff
        )

    def __is_idx_2d_simplified(self, idx, si):
        s_range = self.s_max - self.s_min

        def q_s(sig):
            return 2 * (np.exp(-np.power(s_range / (np.sqrt(2) * sig), 2)) - 1)

        def p_s(sig):
            return 2 * (self.s_max - self.s_min) * \
                sp.special.erf(s_range / (np.sqrt(2) * sig))

        prefactor = np.power(s_range, -4) / 2
        if idx == 1:
            if self.efficient:
                prefactor *= np.power(si[0], -2)
            else:
                prefactor *= self.gain_mean * si[1] / si[0]
            return prefactor * (
                4 * si[0] * q_s(si[0]) + np.sqrt(2 * np.pi) * p_s(si[0])
            ) * (
                si[1] * q_s(si[1]) + np.sqrt(np.pi / 2) * p_s(si[1])
            )
        elif idx == 2:
            if self.efficient:
                prefactor *= np.power(si[1], -2)
            else:
                prefactor *= self.gain_mean * si[0] / si[1]
            return prefactor * (
                4 * si[1] * q_s(si[1]) + np.sqrt(2 * np.pi) * p_s(si[1])
            ) * (
                si[0] * q_s(si[0]) + np.sqrt(np.pi / 2) * p_s(si[0])
            )
        else:
            assert idx == -1
            return 0

    def __is_idx_2d(self, idx, si):
        # Since analytic averaging over sigmas is approximate, we can just do
        # that averaging here with samples of tuning curve widths.
        #
        # Bounded theory after averaging over p(s)
        if self.use_simplified:
            return self.__is_idx_2d_simplified(idx, si)

        smn_cmx = self.s_min - self.c_max
        smx_cmn = self.s_max - self.c_min
        cmx_smx = self.c_max - self.s_max
        cmn_smn = self.c_min - self.s_min

        def q(sig):
            return np.exp(-np.power(smn_cmx / (np.sqrt(2) * sig), 2)) + \
                   np.exp(-np.power(smx_cmn / (np.sqrt(2) * sig), 2)) -  \
                   np.exp(-np.power(cmn_smn / (np.sqrt(2) * sig), 2)) - \
                   np.exp(-np.power(cmx_smx / (np.sqrt(2) * sig), 2))

        def p(sig):
            return smn_cmx * sp.special.erf(smn_cmx / (np.sqrt(2) * sig)) + \
                   smx_cmn * sp.special.erf(smx_cmn / (np.sqrt(2) * sig)) - \
                   cmn_smn * sp.special.erf(cmn_smn / (np.sqrt(2) * sig)) - \
                   cmx_smx * sp.special.erf(cmx_smx / (np.sqrt(2) * sig))

        def s(sig):
            return sp.special.erf(cmx_smx / (np.sqrt(2) * sig)) - \
                   sp.special.erf(smn_cmx / (np.sqrt(2) * sig)) - \
                   sp.special.erf(smx_cmn / (np.sqrt(2) * sig)) + \
                   sp.special.erf(cmn_smn / (np.sqrt(2) * sig))

        if idx == 1:
            sig_prf = np.power(si[0], -2) if self.efficient else si[1] / si[0]
            prefactor = self.gain_mean * sig_prf * np.power(s_range, -4) / 2
            return prefactor * (
                4 * si[0] * q_s(si[0]) + np.sqrt(2 * np.pi) * p_s(si[0])
            ) * (
                si[1] * q_s(si[1]) + np.sqrt(np.pi / 2) * p_s(si[1])
            )
        elif idx == 2:
            sig_prf = np.power(si[1], -2) if self.efficient else si[0] / si[1]
            prefactor = self.gain_mean * sig_prf * np.power(s_range, -4) / 2
            return prefactor * (
                4 * si[1] * q_s(si[1]) + np.sqrt(2 * np.pi) * p_s(si[1])
            ) * (
                si[0] * q_s(si[0]) + np.sqrt(np.pi / 2) * p_s(si[0])
            )
        else:
            assert idx == -1
            sig_prf = 1 if self.efficient else si[0] * si[1]
            return -0.5 * np.pi * sig_prf * np.power(s_range, -4) * \
                s(si[0]) * s(si[1])

    def __is_idx_2d_raw(self, idx, s, si):
        # Bounded theory before averaging over p(s)
        def eta_i(idx, s, si):
            s_idx = s[idx]
            si_idx = si[idx]
            return sp.special.erf((s_idx - self.c_min) / (np.sqrt(2) * si_idx)) - \
                   sp.special.erf((self.c_max - s_idx) / (np.sqrt(2) * si_idx))

        def lambda_i(idx, s, si):
            s_idx = s[idx]
            si_idx = si[idx]
            return np.exp(-np.power(s_idx - self.c_min, 2) / \
                          (2 * np.power(si_idx, 2))) * (self.c_min - s_idx) + \
                   np.exp(-np.power(s_idx - self.c_max, 2) / \
                          (2 * np.power(si_idx, 2))) * (s_idx - self.c_max)

        def omega(c0, c1, s, si):
            return np.exp(-np.power(s[1] - c0, 2) / (2 * np.power(si[1], 2)) - \
                          np.power(s[0] - c1, 2) / (2 * np.power(si[0], 2)))

        if idx == 1:
            prefactor = self.gain_mean * si[1] * np.power(si[0], -2) * \
                np.power(self.s_max - self.s_min, -2) * eta_i(1, s, si)
            return prefactor * (np.sqrt(2 * np.pi) * lambda_i(0, s, si) + \
                                np.pi * si[0] * eta_i(0, s, si))
        elif idx == 2:
            prefactor = self.gain_mean * si[0] * np.power(si[1], -2) * \
                np.power(self.s_max - self.s_min, -2) * eta_i(0, s, si)
            return prefactor * (np.sqrt(2 * np.pi) * lambda_i(1, s, si) + \
                                np.pi * si[1] * eta_i(1, s, si))
        else:
            assert idx == -1

            prefactor = self.gain_mean * np.power(self.s_max - self.s_min, -2)
            return prefactor * (omega(self.c_max, self.c_max, s, si) - \
                                omega(self.c_max, self.c_min, s, si) - \
                                omega(self.c_min, self.c_max, s, si) + \
                                omega(self.c_min, self.c_min, s, si))

    def __expon(self, stim, s_lim, sig):
        return np.exp(-np.power(stim - s_lim, 2) / (2 * np.power(sig, 2)))

    def __bounded_fi_per_pparam_2d(self, param, fixed_sigma, save_sem):
        if fixed_sigma:
            sigmas = param * np.ones((self.b_iters, 2))
        else:
            if self.anisotropic:
                sigmas =  np.random.gamma(self.shape,
                                    scale=param / self.shape,
                                    size=(self.b_iters, 2))
            else:
                sigmas = np.multiply(np.ones((2, self.b_iters)),
                                     np.random.gamma(self.shape,
                                                     scale=param / self.shape,
                                                     size=self.b_iters)).T

        def fn(fi_fn, si):
            Is_11 = fi_fn(1, si)
            Is_22 = fi_fn(2, si)
            Is_12_21 = fi_fn(-1, si)
            return np.array([[Is_11, Is_12_21], [Is_12_21, Is_22]])

        worker = partial(fn, self.__is_idx_2d)
        if not fixed_sigma:
            res = np.array(list(map(worker, sigmas)))
            return np.sqrt(np.linalg.det(np.mean(res, axis=0)))
        else:
            si = np.ones(2) * param
            Is_11 = self.__is_idx_2d(1, si)
            Is_22 = self.__is_idx_2d(2, si)
            Is_12_21 = self.__is_idx_2d(-1, si)
            return np.sqrt(Is_11 * Is_22 - np.power(Is_12_21, 2))

    def __bounded_fi_per_s_2d(self, s, param):
        tuning_widths = param
        if not isinstance(param, (list, np.ndarray)):
            tuning_widths = np.ones(2) * param

        Is_11 = self.__is_idx_2d(1, s, tuning_widths, self.gain_mean)
        Is_22 = self.__is_idx_2d(2, s, tuning_widths, self.gain_mean)
        Is_12_21 = self.__is_idx_2d(-1, s, tuning_widths, self.gain_mean)
        return Is_11, Is_22, Is_12_21, \
            np.sqrt(Is_11 * Is_22 - np.power(Is_12_21, 2))

    def fi_per_pparam(self, param, fixed_sigma=False, save_sem=False):
        if self.bounded:
            assert self.s_dim == 2
            return self.__bounded_fi_per_pparam_2d(param, fixed_sigma, save_sem)
        else:
            return self.__unbounded_fi_per_pparam(param, fixed_sigma)

    def bounded_fi_per_s_avg_param(self, param, s_samples):
        if self.anisotropic:
            sigmas =  np.random.gamma(self.shape,
                                scale=param / self.shape,
                                size=(self.b_iters, 2))
        else:
            sigmas = np.multiply(np.ones(self.b_iters, 2),
                                 np.random.gamma(self.shape,
                                                 scale=param / self.shape,
                                                 size=self.b_iters))

        stims = np.random.uniform(low=self.s_min,
                                  high=self.s_max,
                                  size=(s_samples, 2))

        def fn(fi_fn, s, si):
            Is_11 = fi_fn(1, si, s)
            Is_22 = fi_fn(2, si, s)
            Is_12_21 = fi_fn(-1, si, s)
            return np.array([[Is_11, Is_12_21], [Is_12_21, Is_22]])

        finfos = np.zeros((2, 2, s_samples))
        for i in range(s_samples):
            worker = partial(fn, self.__is_idx_2d_raw, stims[i])
            res = np.array(list(map(worker, sigmas)))
            finfos[:, :, i] = np.mean(res, axis=0)

        return finfos

    def fi_per_s(self, s, param_kwargs):
        assert 'param' in param_kwargs
        param = param_kwargs['param']
        if self.bounded:
            assert not self.efficient and self.s_dim == 2
            return self.__bounded_fi_per_s_2d(s, param)
        else:
            return self.__unbounded_fi_per_pparam(param)

