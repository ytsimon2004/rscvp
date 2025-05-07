from typing import Union

import numpy as np


# noinspection Assert
def simple_decoder(rate_map: np.ndarray,
                   spike_count: np.ndarray,
                   binsize: Union[float, np.ndarray] = 1.0,
                   alpha=1.0,
                   baseline=0.1,
                   prior: np.ndarray = None) -> np.ndarray:
    """

    associated features a[1..n] in time window [t, t+Δt]
    waveform feature space: S = ∪(S[1..j]), where j ∈ J nonoverlapping regions.
    number of spikes: N(spikes) = Σ[j->J] n[j]
    S[j] observed n[j] which follow Poisson distribution: λ[S[j]](x) = ∫[S[j]] λ(a,x)da

    P(x=k) = Poisson(k, λ) = {λ^k * exp(-λ)} / k!

    P(a[1..n]|x)
    = P(n[1..J]|X)
    = Π[j->J] P(n[j]|x)
    = Π[j->J] Poisson( n[j], λ[S[j]](x) )
    = {Π[j->J] (Δt ∫[S[j]] λ(a,x)da) ^ n[j] } exp {-Δt Σ[j-J] ∫[S[j]] λ(a,x)da} / {Π[j->J] n[j]!}       (1)

    In limiting case when S[1..J] -> small, n[1..J] -> 0 or 1, within Δt

    P(a[1..n]|x)
    = Δt^n {Π[i->n] λ(a[i],x)} exp {-Δt λ(x) }                                                          (2)

    K sensors and n[k] spike events on the k-th electrode

    P(a[1..K]|x) = Π[k->K] P(a[1..n,k],x)                                                               (3)

    S[j] correspond single neuron c
    for λ[c](x) = λ[S[j]](x) -> n[c] = n[j]

    # Relation to encoding with spike-sorted units or multi-unit activity.

    (1) => P(n[1..C]|x)
    = Δt^N {Π[c->C] λ[c](x) ^ n[c]} exp {-Δt Σ[c->C] λ[c](x)} / {Π[c->C] n[c]!}                         (4)
    where N = Σ[c->C] n[c]

    consider all spike are part of a single multi-unit cluster, we can ignore feature `a`.

    P(N|X) = Δt^N {λ(x) ^ N} exp {-Δt λ(x)} / N!                                                        (5)

    # Evaluation of the likelihood

    generalized rate function : λ(a,x)
    = spike_count(a,x) / occupancy(x)
    = N p(a,x) / T π(x)
    = μ p(a,x) / π(x)                                                                                   (6)
    λ(x) = μ p(x) / π(x)                                                                                (7)

    where
        spike_count(a, x) = the number of spike with feature a that occurred at stimulus x
        occupancy(x) = the total presentation time of stimulus x
        N = total number of spikes recorded in the time interval (0, T].
        μ = average spiking rate.

    probability distributions
    p(a,x) = 1/N Σ[n->N] K[H[ax]] ({a, x} - {a'[n], x'[n]})                                             (8)
    p(x)   = 1/N Σ[n->N] K[H[ax]] (x - x'[n])                                                           (9)
    π(x)   = 1/N Σ[r->R] K[H[ax]] (x - x'[r])                                                           (10)

    where
        {a'[n], x'[n]}[n->N] = set of N spikes associated feature vector and stimuli
        {x'[r]}[r->R] = set of R observed (or chosen) stimuli
        K[H]() = kernel function with bandwidth matrix H.
        H[x] = combined bandwidth matrix for the stimulus only.
        H[ax] = combined bandwidth matrix for spike feature and stimulus.

    # Bayesian decoding.

    To infer uncertainty or probability of a hidden stimulus x at time t given
    the observed m spike events with associated features a[1..m]:

    P(x[t]|a[1..m]) = P(a[1..m]|x[t]) P(x[t]) / P(a[1..m])                                              (11)

    where:
        P(x[t]) = prior information about the stimulus.
        P(a[1:m]|x[t]) = observation through the likelihood function (eq2, 3)
        P(a[1:m]) = a normalizing constant
        P(x[t]|a[1:m]) = posterior

    P(x[t]|a[1..m]) ∝ P(a[1..m]|x[t]) P(x[t])

    Parameters
    ----------
    rate_map
        λ(a,x) = Array[rate, A=cluster, X=spatial_bins]
    spike_count
        p(a,t) = Array[count, A=cluster, T=time_bins]
    binsize
        Durations of time bins in which spikes were counted. Either a
        float scalar, a vector with time bin durations or a (n,2) array
        with time bin start and end times.
    alpha
        Rate map scaling factor. Either a float scalar, or a vector with
        a different alpha value for each cell.
    baseline
        Baseline rate that is added to each rate map
    prior
        Prior probability distribution = Array[probability, X=spatial_bins] array

    Returns
    -------
    posterior
        Posterior probability distribution =  Array[probability, T=time_bins, X=spatial_bins]

    References
    ----------

    * Bayesian decoding using unsorted spikes in the rat hippocampus (2014)

      https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3921373/

    * paper-subiculum fklab.papers.subiculum.decoder_for_sorted_cells.py simple_decoder()

    """

    # rate map verify
    rate_map = np.asarray(rate_map)
    n_cluster, spatial_bins = rate_map.shape

    if n_cluster == 0:
        raise ValueError('empty features')

    # spike count verify
    spike_count = np.asarray(spike_count)
    if spike_count.shape[0] != n_cluster:
        raise ValueError('number of features mismatched')
    time_bins = spike_count.shape[1]

    if time_bins == 0:
        return np.zeros((0, spatial_bins), dtype=float)

    # binsize verify
    # Array[Δt, T=time_bins]
    if isinstance(binsize, np.ndarray):
        if binsize.shape != (time_bins,):
            raise ValueError('binsize shape mismatched with time_bins')
    else:
        binsize = np.full((time_bins,), binsize)

    # alpha verify
    # Array[float, A=cluster]
    if isinstance(alpha, np.ndarray):
        if alpha.shape != (n_cluster,):
            raise ValueError('alpha shape mismatched with n_cluster')
    else:
        alpha = np.full((n_cluster,), alpha)

    # prior verify
    if prior is not None:
        if prior.shape != (spatial_bins,):
            raise ValueError('prior shape mismatched with spatial_bins')

    """
    P(a[1..n]|x)
    = Δt^n {Π[i->n] λ(a[i],x)} exp {-Δt λ(x) }
    = Δt^n exp {Σ[k->K] ln λ(a[k],x) } exp {-Δt λ(x) }
    = Δt^n exp {-Δt λ(x) + Σ[k->K] ln λ(a[k],x) }
    """

    # λ(a,x) = A * λ(a,x) + B
    rate_map = alpha * rate_map + baseline  # Array[rate, A=cluster, X=spatial_bins]

    # make sure 0.0 doesn't present in log(0.0) and cause NaN in result
    assert np.all(rate_map > 0.0), f'0 in rate_map'

    # λ(x) = Σ[a] λ(a,x)
    lambda_x = np.sum(rate_map, axis=0)  # Array[Σ rate, X=spatial_bins]
    # -Δt λ(x)
    dt_lambda_x = np.multiply.outer(-binsize, lambda_x)  # Array[-rate, T=time_bins, X=spatial_bins]
    # Σ[k->K] ln λ(a[k],x)
    outer_multiply = np.vectorize(np.multiply.outer, signature='(t),(s)->(t,s)')
    lambda_s = np.sum(outer_multiply(spike_count, np.log(rate_map)), axis=0)
    # Array[float, T=time_bins, X=spatial_bins]

    posterior = np.exp(dt_lambda_x + lambda_s)

    if prior is not None:
        posterior = posterior * prior[None, :]

    # normalize
    posterior /= np.sum(posterior, axis=1, keepdims=True)

    return posterior
