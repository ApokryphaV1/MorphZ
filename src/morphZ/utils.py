import numpy as np
from scipy.signal import correlate
from statsmodels.tsa.stattools import acf

def compute_rho_f2_0_via_statsmodels(f2_values, nlags=None):
    """
    Estimate the integrated autocorrelation time for a 1D array of f2 values using statsmodels.acf.
    
    The integrated autocorrelation time is defined as:
        tau = 1 + 2 * sum_{lag>=1} acf(lag)
    where the sum stops at the first negative value.
    
    Args:
        f2_values: 1D numpy array of f2(theta) values computed from posterior samples.
        nlags: Maximum number of lags to compute (if None, defaults to len(f2_values) - 1).
    
    Returns:
        tau: An estimate of the integrated autocorrelation time (rho_f2(0)).
    """
    if nlags is None:
        nlags = len(f2_values) - 1
    # Compute the autocorrelation function using fft for speed.
    acf_values = acf(f2_values, nlags=nlags, fft=True)
    # Start with lag 0 which is 1; then sum positive autocorrelations until the first negative value.
    tau = 1.0
    for lag in range(1, len(acf_values)):
        if acf_values[lag] > 0:
            tau += 2 * acf_values[lag]
        else:
            break
    return tau

def compute_rho_f2_0_via_correlate(f2_values):
    """
    Estimate the integrated autocorrelation time for a 1D array of f2 values using scipy.signal.correlate.
    
    The integrated autocorrelation time is defined as:
        tau = 1 + 2 * sum_{lag>=1} acf(lag)
    where the sum stops at the first negative value.
    
    Args:
        f2_values: 1D numpy array of f2(theta) values computed from posterior samples.
    
    Returns:
        tau: An estimate of the integrated autocorrelation time (rho_f2(0)).
    """
    x = f2_values - np.mean(f2_values)
    n = x.size
    # Full correlation has length 2*n - 1
    corr = correlate(x, x, mode='full')
    # The 'center' index for lag=0:
    mid = n - 1
    # Keep only nonnegative lags: corr[mid:] => lags 0,1,...,n-1
    corr = corr[mid:]
    # Normalize so corr[0] = 1
    corr /= corr[0]

    # Sum the positive portion of corr
    # (some strategies sum all lags until correlation first becomes negative).
    sum_pos = 0.0
    for val in corr[1:]:  # skip lag0 because it's 1
        if val > 0:
            sum_pos += val
        else:
            break

    # Integrated autocorr time approx: tau = 1 + 2 * sum_{k>0} corr(k)
    tau = 1.0 + 2.0 * sum_pos
    return tau

def log_plus(x,y):
    """
    Computes log(exp(x) + exp(y)) in a numerically stable way.
    """
    if x > y:
      summ = x + np.log(1+np.exp(y-x))
    else:
        summ = y + np.log(1+np.exp(x-y))
    return summ

def log_sum(vec): 
    """
    Computes the log of the sum of exponentials of a vector of numbers.
    log(sum(exp(vec)))
    """
    r = -np.inf
    for i in range(len(vec)):
       r =log_plus(r, vec[i])
    return r
def error_bound_from_oscillation(x):
    """
    Given a sequence x of iterates (assumed to be oscillatory about the fixed point)
    this function returns an error bound computed as half the distance between the min
    and max of the last two iterates.
    discard 20% of the iterates to avoid the initial transient.
    Parameters:
       x : list or np.array
           A sequence of iterates.
    
    Returns:
       err_bound : float
           An error bound estimate: (max(x) - min(x)) .
    """
    x = np.array(x, dtype=float)
    x = x[int(0.2*len(x)):] 
    if len(x) < 2:
        raise ValueError("Need at least two iterates to compute oscillation bounds.")
    lower = min(x)
    upper = max(x)
    return (upper - lower)
