"""
Module implementing Champagne
"""

__author__ = 'Proloy DAS'
__licence__ = 'apache 2.0'

import numpy as np
from scipy import linalg
from eelbrain import *
from math import sqrt
import tqdm
import time

def _myinv(x):
    """

    Computes inverse

    parameters
    ----------
    x: ndarray
    array of shape (dc, dc)

    returns
    -------
    ndarray
    array of shape (dc, dc)
    """
    x = np.real(np.array(x))
    y = np.zeros(x.shape)
    y[x>0] = 1/x[x > 0]
    return y


def _compute_gamma_i(z, x):
    """

    Computes Gamma_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    parameters
    ----------
    z: ndarray
        array of shape (dc, dc)
        auxiliary variable,  z_i

    x: ndarray
        array of shape (dc, dc)
        auxiliary variable, x_i

    returns
    -------
    ndarray
    array of shape (dc, dc)

    """
    [e, v] = linalg.eig(z)
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real( np.dot(temp.T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig( (temp * e) * e[:, np.newaxis] )
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * _myinv(np.real(e)), u)
    return np.array( np.real(np.dot(temp * d, np.matrix(temp).H)) )


def _compute_objective(Cb, isigma_b):
    """

    Compute objective value at a given iteration

    parameters
    ----------
    Cb: ndarray
    array of shape (K, K)

    isigma_b: ndarray
    array of shape (K, K)

    returns
    -------
    float

    """
    return np.trace(np.dot(isigma_b, Cb)) - 2 * np.sum (np.log(np.diag(linalg.cholesky(isigma_b))))


class Champagne:
    """
    Champagne algorithm

    Reference--
    Wipf, David P., et al. "Robust Bayesian estimation of the location, orientation, and time
    course of multiple correlated neural sources using MEG." NeuroImage 49.1 (2010): 641-655.

    Parameters
    ----------
    lead_field: NDVar
        array of shape (K, N)
        lead-field matrix.
        both fixed or free orientation lead-field vectors can be used.

    orientation: 'fixed'|'free'
        'fixed': orientation-constrained lead-field matrix.
        'free': free orientation lead-field matrix.

    noise_covariance: ndarray
        array of shape (K, K)
        noise covariance matrix
        use empty-room recordings to generate noise covariance matrix at sensor space.

    n_iter: int, optionnal
        number of iterations
        default is 1000

    Attributes
    ----------
    Gamma: list
        list of length N
        individual source covariance matrices

    inverse_sigma_b: ndarray
        array of shape (K, K)
        inverse of data covariance under the model

    objective: list, optional
        list of objective values at each iteration
        returned only if verbose=1

    est_data_covariance: ndarray
        array of shape (K, K)
        estimated data covariance under the model
        returned only if verbose=1

    emp_data_covariance: ndarray
        array of shape (K, K)
        empirical data covariance
        returned only if verbose=1

    inverse_kernel: ndarray
        array of shape (K, K)
        inverse imaging kernel
        returned only if return_inverse_kernel=1


    """
    def __init__(self, lead_field, noise_covariance, n_iter=1000):
        if lead_field.has_dim('space'):
            self.lead_field = lead_field.get_data (dims=('sensor', 'source', 'space')).astype('float64')
            self.sources_n = self.lead_field.shape[1]
            self.lead_field = self.lead_field.reshape(self.lead_field.shape[0], -1)
            self.orientation = 'free'
            self.space = lead_field.space
        else:
            self.lead_field = lead_field.get_data(dims=('sensor', 'source')).astype('float64')
            self.sources_n = self.lead_field.shape[1]
            self.orientation = 'fixed'
        self.source = lead_field.source
        self.sensor = lead_field.sensor
        self.noise_covariance = noise_covariance
        self.n_iter = n_iter

    def solve(self, meg, verbose=0):
        """

        Champagne algorithm

        parameters
        ----------
        meg: NDVar
            meg data

        verbose: {0,1}
            verbosity of the method : 1 will display informations while 0 will display nothing
            default = 0

        returns
        -------
        self
        """

        y = meg.get_data(('sensor', 'time'))
        Cb = np.dot(y, y.T) / y.shape[1]    # empirical data covariance
        yhat = linalg.cholesky(Cb, lower=True)

        # noise_covariance = np.eye(self.noise_covariance.shape[0])  # since the data is pre whitened
        noise_covariance = self.noise_covariance

        # Choose dc
        if self.orientation == 'fixed': dc = 1
        elif self.orientation == 'free': dc = 3

        # initializing gamma
        wf = linalg.cholesky(self.noise_covariance, lower=True)
        ytilde = linalg.solve(wf, yhat)
        eta = 0.1 * (ytilde.shape[0] / np.trace (np.dot(ytilde, ytilde.T)))
        gamma = [eta * np.eye (dc, dtype='float') for _ in range (self.sources_n)]  # Initial gamma
        # print "Gamma = {:10f}".format(eta)

        # model data covariance
        sigma_b = noise_covariance
        for j in range(self.sources_n):
            sigma_b = sigma_b + np.dot(self.lead_field[:, j * dc:(j + 1) * dc],
                                        np.dot (gamma[j], self.lead_field[:, j * dc:(j + 1) * dc].T))
        isigma_b = linalg.inv(sigma_b)

        if verbose == 1:
            self.objective = []
            self.objective.append(_compute_objective(Cb, isigma_b))

        if verbose:
            start = time.time()

        # champagne iterations
        for it in xrange(self.n_iter):
            # pre-compute some useful matrices
            lhat = np.dot(isigma_b, self.lead_field)

            # compute sigma_b for the next iteration
            sigma_b_next = np.copy(noise_covariance)

            for i in xrange (self.sources_n):
                # update Xi
                x = np.dot(gamma[i], np.dot(yhat.T, lhat[:, i * dc:(i + 1) * dc]).T)

                # update Zi
                z = np.dot(self.lead_field[:, i * dc:(i + 1) * dc].T, lhat[:, i * dc:(i + 1) * dc])

                # update Ti
                if dc == 1:
                    gamma[i] = sqrt(np.dot(x, x.T)) / np.real(sqrt(z))
                else:
                    gamma[i] = _compute_gamma_i(z, x)

                # update sigma_b for next iteration
                sigma_b_next = sigma_b_next + np.dot(self.lead_field[:, i * dc:(i + 1) * dc],
                                                    np.dot(gamma[i], self.lead_field[:, i * dc:(i + 1) * dc].T))

            # update sigma_b
            sigma_b = sigma_b_next
            isigma_b = linalg.inv(sigma_b)
            if verbose == 1:
                self.objective.append(_compute_objective(Cb, isigma_b))
                print("Iteration : {:}, objective value : {:f}\n".format(it+1, self.objective[it]))

        if verbose:
            end = time.time()
            print "total time elapsed : {:f}s".format(end - start)

        self.Gamma = gamma
        self.inverse_sigma_b = isigma_b

        # # inverse kernel
        # if return_inverse_kernel:
        #     inverse_kernel = np.copy(self.lead_field.T).astype('float64')
        #     for i in xrange(sources_n):
        #         inverse_kernel[i * dc:(i + 1) * dc, :] \
        #             = np.dot (gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])

        # self.inverse_kernel = np.dot(inverse_kernel, isigma_b)

        if verbose:
            self.est_data_covariance = sigma_b
            self.emp_data_covariance = Cb

    def return_inverse_operator(self):
        """

        Returns inverse operator

        returns:
        -------
        ndarray
        array of shape (N, K)

        """
        if self.orientation == 'free': dc = 3
        else: dc = 1

        inverse_kernel = np.copy(self.lead_field.T).astype ('float64')
        for i in xrange (self.sources_n):
            inverse_kernel[i * dc:(i + 1) * dc, :] \
                = np.dot (self.Gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])

        inverse_kernel = np.dot (inverse_kernel, self.inverse_sigma_b)

        return inverse_kernel

    def apply_inverse_operator(self, meg):
        """

        parameters
        ----------
        meg: NDVar
            meg data

        returns:
        -------
        ndvar
        source estimates

        """
        y = meg.get_data(('sensor', 'time')) # meg data

        if self.orientation == 'free': dc = 3
        else: dc = 1

        inverse_kernel = np.copy(self.lead_field.T).astype ('float64')
        for i in xrange (self.sources_n):
            inverse_kernel[i * dc:(i + 1) * dc, :] \
                = np.dot (self.Gamma[i], inverse_kernel[i * dc:(i + 1) * dc, :])

        inverse_kernel = np.dot(inverse_kernel, self.inverse_sigma_b)

        se = np.dot(inverse_kernel, y)
        if self.orientation == 'fixed':
            ndvar = NDVar(se, dims=(self.source, meg.time))
        else:
            ndvar = NDVar(se.T.reshape(-1, self.sources_n, dc), dims=(meg.time, self.source, self.space))

        return ndvar
