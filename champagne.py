# -*- coding: utf-8 -*-
"""
Module implementing Champagne:

An fast implementation of Champagne algorithm.

Reference--
Wipf, David P., et al. "Robust Bayesian estimation of the location, orientation, and time
course of multiple correlated neural sources using MEG." NeuroImage 49.1 (2010): 641-655.

Created on Tues Mar 24 2018

@author: Proloy Das

@licence = apache 2.0
"""
import numpy as np
from scipy import linalg
from math import sqrt
import tqdm
import time


def champagne(L, B, noise_cov, T, idata_cov, dc, max_Iter=30, verbose=True ):
    """

    :param L: lead-field matrix
            (K, N) 2D array
            uses either fixed or free orientation lead-field vectors.
    :param B: meg data
            (K, T) 2D array
    :param noise_cov: noise co-variance
            (K, K) 2D array
    :param T: previous estimates of covariance of source activities
            list [#sources (dc, dc) 2D arrays]
    :param idata_cov: previous estimates of inverse data covariances
            (K, K) 2D array
    :param dc: orientation
            scalar integer
            1 = fixed, 3 = free
    :param max_Iter: self-explained
            scalar integer
            Maximum number of iterations, usually set at 30.

    :return: dict of relevant variables:
            "Tau": list [#sources (dc, dc) 2D arrays]
            "inverse_kernel": (N, M) 2D array
            "isigma_b": (M, M) 2D array
            "objective": 1D array

    Examples:

    # for free orientation
    >>> dc = 3
    # pre-whiten data
    >>> wf = linalg.cholesky(noise_cov, lower=True)
    >>> y = linalg.solve(wf, y)
    >>> L = linalg.solve(wf, L)
    >>> noise_cov = np.eye(L.shape[0])
    # choose initial guess for Tau (same as mne)
    >>> gamma = L.shape[0]/np.trace(np.dot(y,y.T))
    >>> num_sources = L.shape[1] / dc
    >>> Tau = [gamma * np.eye (dc, dtype='float') for _ in range (num_sources)]
    # Construct inverse data covariances corresponding to that Tau
    >>> idata_cov = linalg.solve(noise_cov + np.dot(L, np.dot(linalg.block_diag(Tau), L.T)), np.eye(L.shape[0], dtype='float'))
    >>> output = champagne(L, y, inoise_cov, Tau, idata_cov, dc, max_Iter=50)
    # Construct source-estimate
    >>> source_estimate = np.dot (output["inverse_kernel"], np.dot (output["isigma_b"], y))

    """
    # we replace B with a matrix Bhat K * rank(B) (=K)
    Cb = np.dot(B, B.T) / B.shape[1]
    Bhat = linalg.cholesky(Cb, lower=True)

    # Change this to previous to see any change!
    # [E, V] = linalg.eig(Cb)
    # Bhat = np.real( np.dot(V, np.dot(np.diag(np.sqrt(E)), np.matrix(V).H)) )

    num_source = (L.shape[1])/dc

    isigma_b = idata_cov
    inoise_cov = linalg.inv (noise_cov)

    # store values to return
    obj = np.empty(0)
    new_obj = np.trace (np.dot (isigma_b, Cb)) - 2 * np.sum (np.log (np.diag (linalg.cholesky (isigma_b))))
    obj = np.append (obj, new_obj)

    T = list(T)

    # Champagne iterations
    for _ in tqdm.tqdm(xrange(max_Iter)):
        time.sleep(0.001)
        # pre-compute some useful matrices
        Lhat = np.dot(isigma_b, L)

        if dc == 1:
            # compute isigma_b for the next iteration
            new_isigma_b = inoise_cov

            for i in xrange (num_source):

                # update Xi
                x = np.dot (T[i], np.dot (Bhat.T, Lhat[:, i * dc:(i + 1) * dc]).T)

                # update Zi
                z = np.dot (L[:, i * dc:(i + 1) * dc].T, Lhat[:, i * dc:(i + 1) * dc])

                # update Ti
                T[i] = sqrt (np.dot (x, x.T)) / np.real (sqrt (z))

                # update new_isigma_b
                temp1 = np.dot (new_isigma_b, L[:, i * dc:(i + 1) * dc])
                new_isigma_b = new_isigma_b - np.dot (temp1, temp1.T) * T[i] / \
                                   (1 + T[i] * np.dot (L[:, i * dc:(i + 1) * dc].T, temp1))


            # Update isigma_b
            isigma_b = np.copy (new_isigma_b)

        else:
            # compute isigma_b for the next iteration
            new_isigma_b = noise_cov

            for i in xrange (num_source):

                # update Xi
                x = np.dot (T[i], np.dot (Bhat.T, Lhat[:, i * dc:(i + 1) * dc]).T)

                # update Zi
                z = np.dot (L[:, i * dc:(i + 1) * dc].T, Lhat[:, i * dc:(i + 1) * dc])

                # update Ti
                T[i] = solve_for_Tau_n(z, x)

                # update new_isigma_b
                new_isigma_b = new_isigma_b + np.dot (L[:, i * dc:(i + 1) * dc],
                                                          np.dot (T[i], L[:, i * dc:(i + 1) * dc].T))

            # update isigma_b
            isigma_b = linalg.inv (new_isigma_b)

        # Calculate fval
        new_obj = np.trace(np.dot (isigma_b, Cb)) + 2 * np.sum(np.log(np.diag(linalg.cholesky (new_isigma_b))))
        obj = np.append (obj, new_obj)

    # Return all relevant variables
    # prepare the inverse kernel
    inverse_kernel = np.zeros(L.T.shape)
    for i in xrange(num_source):
        inverse_kernel[i * dc:(i + 1) * dc, :] = np.dot(T[i], L[:, i * dc:(i + 1) * dc].T)

    out = {"Tau": T,
            "inverse_kernel": inverse_kernel,
           "isigma_b": isigma_b,
           "objective": obj}

    return out


def myinv(x):
    x = np.real(np.array(x))
    y = np.zeros(x.shape)
    y[x>0] = 1/x[x > 0]
    return y


def solve_for_Tau_n(z, x):
    """
    Computes Tau_i = Z**(-1/2) * ( Z**(1/2) X X' Z**(1/2)) ** (1/2) * Z**(-1/2)
                   = V(E)**(-1/2)V' * ( V ((E)**(1/2)V' X X' V(E)**(1/2)) V')** (1/2) * V(E)**(-1/2)V'
                   = V(E)**(-1/2)V' * ( V (UDU') V')** (1/2) * V(E)**(-1/2)V'
                   = V (E)**(-1/2) U (D)**(1/2) U' (E)**(-1/2) V'

    :param z: auxiliary variable,  z_i
            (dc, dc) 2D array

    :param x: auxiliary variable, x_i
            (dc * K) 2D array

    :return: Tau_i

    """
    [e, v] = linalg.eig(z)
    e[e < 0] = 0
    temp = np.dot(x.T, v)
    temp = np.real( np.dot(temp.T, temp))
    e = np.sqrt(e)
    [d, u] = linalg.eig( (temp * e) * e[:, np.newaxis] )
    d[d < 0] = 0
    d = np.sqrt(d)
    temp = np.dot(v * myinv(np.real(e)), u)
    return np.array( np.real(np.dot(temp * d, np.matrix(temp).H)) )
