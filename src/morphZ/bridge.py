import numpy as np
from scipy.special import logsumexp

from . import utils


def bridge_sampling_ln(
    f,
    g,
    samples_post,
    log_post,
    samples_prop,
    tol=1e-4,
    max_iter=40000,
    verbose: bool = True,
):
    """
    Estimate log marginal likelihood log p(y) using bridge sampling in log space.

    Args:
        f: Callable that returns the log of the unnormalized target density
            for a single parameter vector, i.e. ``log p(y, theta) = log p(y|theta) + log p(theta)``.
            Called as ``f(theta)`` with ``theta`` shaped ``(d,)``.
        g: Callable that returns the log proposal density. Called as
            ``g(theta_T)`` where ``theta_T`` has shape ``(d, n)`` (samples in columns).
        samples_post (ndarray): Posterior samples of shape ``(N1, d)`` used on the
            target side of the bridge.
        log_post (ndarray): Precomputed log target values for ``samples_post`` of
            shape ``(N1,)``. This saves recomputation of expensive likelihoods.
        samples_prop (ndarray): Proposal samples of shape ``(N2, d)`` used on the
            proposal side of the bridge. These are typically drawn from a KDE.
        tol (float): Convergence tolerance on successive log‑evidence updates.
            Smaller values increase accuracy but may require more iterations.
        max_iter (int): Maximum number of bridge iterations.
        verbose (bool): When True, print progress, warnings, and convergence info.

    Returns:
        list[float, float]: ``[log_evidence, rmse_estimate]`` where the second
        element is an approximate RMSE estimate computed via ``compute_bridge_rmse``.

    Suggestions:
        - If ``f`` occasionally fails (e.g., numerical domain errors), this
          function automatically skips those proposal samples and continues.
        - Use a moderate ``tol`` (1e‑4 to 1e‑3) and cap ``max_iter`` to prevent
          long runs on difficult posteriors.
    """

    # Compute logs of densities for posterior samples
    log_f_post = log_post
    log_g_post = g(samples_post.T)

    # Compute logs of densities for proposal samples
    successful_samples = []
    log_f_prop_results = []
    failure_count = 0
    num_samples = len(samples_prop)
    for i, theta in enumerate(samples_prop):
        try:
            result = f(theta)
            # Only append if the evaluation is successful
            successful_samples.append(theta)
            log_f_prop_results.append(result)
        except Exception:
            # If f(theta) fails, increment counter and skip this sample
            failure_count += 1
            continue
        finally:
            if verbose:
                # Ensure progress is always updated
                print(f"\rEvaluating target distribution: {i+1}/{num_samples}", end="")

    # Rebuild arrays from the lists of successful evaluations
    samples_prop = np.array(successful_samples)
    log_f_prop = np.array(log_f_prop_results)
    if verbose:
        print()

    if failure_count > 0 and verbose:
        print(
            f"\nWarning: Evaluation of target distribution failed for {failure_count} samples, which were skipped."
        )

    # As a safeguard, filter out any remaining non-finite values (e.g., if f(theta) returned inf)
    finite_mask = np.isfinite(log_f_prop)

    samples_prop = samples_prop[finite_mask]
    log_f_prop = log_f_prop[finite_mask]
    if verbose:
        print(
            f"Filtered proposal samples: {len(samples_prop)} valid samples out of {num_samples} total samples."
        )
    # Now compute log_g for the filtered samples. g expects (n_dims, n_samples)
    log_g_prop = g(samples_prop.T)

    N1 = len(log_f_post)
    N2 = len(log_f_prop)

    if N2 == 0:
        if verbose:
            print(
                "\nWarning: No valid samples from the proposal distribution. Bridge sampling failed."
            )
        return [np.nan, np.nan]

    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)

    # Initial guess for p(y) using importance sampling on proposal samples:
    log_p_old = logsumexp(log_f_prop - log_g_prop) - np.log(N2)

    term1 = np.log(s1) + log_f_prop
    term1_post = np.log(s1) + log_f_post

    for t in range(max_iter):
        # For proposal samples:
        term2 = np.log(s2) + log_p_old + log_g_prop
        log_den_prop = logsumexp(np.vstack((term1, term2)), axis=0)
        log_terms_prop = log_f_prop - log_den_prop
        log_num = -np.log(N2) + utils.log_sum(log_terms_prop)

        # For posterior samples:
        term2_post = np.log(s2) + log_p_old + log_g_post
        log_den_post = logsumexp(np.vstack((term1_post, term2_post)), axis=0)
        log_terms_post = log_g_post - log_den_post
        log_den = -np.log(N1) + utils.log_sum(log_terms_post)

        log_p_new = log_num - log_den
        if verbose:
            print(
                f"\r iteration: {t+1} log(z) old: {log_p_old} log(z) New: {log_p_new}",
                end="",
            )
        # Check convergence:
        if np.abs(log_p_new - log_p_old) < tol:
            log_p_final = np.max([log_p_new, log_p_old])
            rmse_est = compute_bridge_rmse(
                log_p_final,
                f,
                g,
                samples_prop,
                samples_post,
                log_f_prop,
                log_g_prop,
                log_f_post,
                log_g_post,
                s1,
                s2,
            )
            if verbose:
                print(
                    f"\r\nConverged in {t+1} iterations. log(z): {log_p_final:.4f} +/-: {rmse_est:.4f}",
                    end="",
                )
            return [log_p_final, rmse_est]

        log_p_old = log_p_new

    log_p_final = np.max([log_p_new, log_p_old])
    rmse_est = compute_bridge_rmse(
        log_p_final,
        f,
        g,
        samples_prop,
        samples_post,
        log_f_prop,
        log_g_prop,
        log_f_post,
        log_g_post,
        s1,
        s2,
    )
    if verbose:
        print(f"\nConvergence not reached within {max_iter} iterations.")
        print(f"Final log(z): {log_p_final:.4f} +/-: {rmse_est:.4f}")

    return [log_p_final, rmse_est]


def compute_bridge_rmse(
    log_p_final,
    f,
    g,
    samples_prop,
    samples_post,
    log_f_prop,
    log_g_prop,
    log_f_post,
    log_g_post,
    s1,
    s2,
    posterior_acf_func=utils.compute_rho_f2_0_via_correlate,
):
    """
    Estimate the root mean square error (RMSE) for the bridge sampling result.

    Args:
        log_p_final (float): Final bridge estimate of log evidence.
        f, g (callable): Same target/proposal log‑density callables as used in
            ``bridge_sampling_ln``.
        samples_prop (ndarray): Proposal samples ``(N2, d)``.
        samples_post (ndarray): Posterior samples ``(N1, d)``.
        log_f_prop, log_g_prop, log_f_post, log_g_post (ndarray): Cached log
            densities evaluated on corresponding samples.
        s1, s2 (float): Mixing coefficients N1/(N1+N2) and N2/(N1+N2).
        posterior_acf_func (callable): Function estimating the integrated
            autocorrelation time of ``f2`` terms from the posterior side. By
            default a fast correlate‑based estimator is used; if you have
            statsmodels available, consider ``utils.compute_rho_f2_0_via_statsmodels``.

    Returns:
        float: Approximate RMSE of the log‑evidence estimate.

    Note:
        This is a heuristic approximation following common practice for bridge
        sampling diagnostics; it is useful for relative comparisons across runs.
    """
    eps = 1e-12
    N1 = len(log_f_post)
    N2 = len(log_f_prop)

    if N1 == 0 or N2 == 0:
        return np.inf

    # --- For PROPOSAL SAMPLES ---
    lp_py_prop = log_f_prop - log_p_final
    left_part = np.log(s1) + lp_py_prop
    right_part = np.log(s2) + log_g_prop
    log_denom_prop = logsumexp(np.vstack((left_part, right_part)), axis=0)
    log_f1_prop = lp_py_prop - log_denom_prop

    max_log_f1 = np.max(log_f1_prop)
    f1_prop = np.exp(log_f1_prop - max_log_f1)

    mean_f1_prop = f1_prop.mean()
    var_f1_prop = f1_prop.var(ddof=1)

    # --- For POSTERIOR SAMPLES ---
    lp_py_post = log_f_post - log_p_final
    left_part_post = np.log(s1) + lp_py_post
    right_part_post = np.log(s2) + log_g_post
    log_denom_post = logsumexp(np.vstack((left_part_post, right_part_post)), axis=0)
    log_f2_post = log_g_post - log_denom_post

    max_log_f2 = np.max(log_f2_post)
    f2_post = np.exp(log_f2_post - max_log_f2)
    mean_f2_post = f2_post.mean()
    var_f2_post = f2_post.var(ddof=1)

    # --- Autocorrelation correction ---
    rho_f2_0 = posterior_acf_func(f2_post) if posterior_acf_func else 1.0

    # --- Compute relative MSE using the formula ---
    term1 = (var_f1_prop / ((mean_f1_prop + eps) ** 2)) / N2
    term2 = (rho_f2_0 * var_f2_post / ((mean_f2_post + eps) ** 2)) / N1
    re2 = term1 + term2

    return np.sqrt(re2)
