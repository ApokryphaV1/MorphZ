import logging
import multiprocessing as mp
import os
import pickle
from functools import partial
from typing import Optional

import numpy as np
from scipy.special import logsumexp

from . import utils

logger = logging.getLogger(__name__)


def _maybe_print(message: str = "", *, verbose: bool = False, **kwargs) -> None:
    """Print helper that honors the verbose flag."""
    if verbose:
        print(message, **kwargs)


def _ensure_picklable(obj):
    try:
        pickle.dumps(obj)
    except Exception as exc:  # pragma: no cover - depends on user inputs.
        raise TypeError(
            "The provided target log-density 'f' must be picklable to use multiprocessing."
        ) from exc


def _pool_eval_callable(task, func):
    idx, theta = task
    try:
        value = func(theta)
        return idx, value, None
    except Exception as exc:  # pragma: no cover - propagated to main for accounting.
        return idx, None, repr(exc)


def _resolve_worker_count(requested_workers, num_samples, ctx):
    available = ctx.cpu_count() if hasattr(ctx, "cpu_count") else mp.cpu_count()
    worker_count = requested_workers if requested_workers and requested_workers > 0 else available
    return max(1, min(worker_count, num_samples))


def _evaluate_samples_serial(f, samples_prop, logger, verbose: bool = False):
    success_pairs = []
    failure_count = 0
    num_samples = len(samples_prop)
    for i, theta in enumerate(samples_prop):
        try:
            result = f(theta)
            success_pairs.append((i, result))
            _maybe_print(
                f"Number of evaluated proposed samples: {i + 1}/{num_samples}",
                verbose=verbose,
                end="\r",
            )
        except Exception:
            failure_count += 1
            continue
        finally:
            logger.debug("Evaluating target distribution: %s/%s", i + 1, num_samples)

    success_pairs.sort(key=lambda pair: pair[0])
    successful_samples = [samples_prop[idx] for idx, _ in success_pairs]
    log_f_prop_results = [value for _, value in success_pairs]
    return successful_samples, log_f_prop_results, failure_count


def _evaluate_samples_parallel(f, samples_prop, pool_spec, logger, verbose: bool = False):
    num_samples = len(samples_prop)
    if num_samples == 0:
        return [], [], 0
    if pool_spec is None or (isinstance(pool_spec, int) and pool_spec <= 1):
        return _evaluate_samples_serial(f, samples_prop, logger, verbose=verbose)

    ctx = mp.get_context("spawn")
    owns_pool = False
    pool = None
    chunk = None

    if isinstance(pool_spec, int):
        worker_count = _resolve_worker_count(pool_spec, num_samples, ctx)
        if worker_count <= 1:
            return _evaluate_samples_serial(f, samples_prop, logger, verbose=verbose)
        _ensure_picklable(f)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        pool = ctx.Pool(processes=worker_count)
        owns_pool = True
        chunk = max(1, num_samples // (worker_count * 4)) if num_samples > worker_count else 1
        logger.info("Using internal multiprocessing pool with %s workers for bridge sampling.", worker_count)
    elif hasattr(pool_spec, "map"):
        pool = pool_spec
        worker_hint = None
        for attr in ("processes", "num_workers", "nthreads", "size"):
            worker_hint = getattr(pool, attr, None)
            if worker_hint:
                break
        if worker_hint:
            logger.info(
                "Using external pool (%s) with %s workers for bridge sampling.",
                type(pool).__name__,
                worker_hint,
            )
        else:
            logger.info("Using external pool (%s) for bridge sampling.", type(pool).__name__)
    else:  # pragma: no cover - defensive programming for user inputs.
        raise TypeError("pool must be None, int, or an object with a .map method")

    success_pairs = []
    failure_count = 0
    processed = 0
    eval_func = partial(_pool_eval_callable, func=f)
    tasks = enumerate(samples_prop)

    try:
        mapper = getattr(pool, "imap_unordered", None)
        if callable(mapper):
            if owns_pool and chunk:
                results_iter = mapper(eval_func, tasks, chunksize=chunk)
            else:
                results_iter = mapper(eval_func, tasks)
        else:
            map_func = getattr(pool, "map", None)
            if map_func is None:  # pragma: no cover - contract requires map
                raise TypeError("Provided pool must implement a map method.")
            if owns_pool and chunk:
                results_iter = map_func(eval_func, tasks, chunk)
            else:
                results_iter = map_func(eval_func, tasks)

        for idx, value, error_msg in results_iter:
            processed += 1
            _maybe_print(
                f"Number of evaluated proposed samples: {processed}/{num_samples}",
                verbose=verbose,
                end="\r",
            )
            logger.debug("Evaluating target distribution: %s/%s", processed, num_samples)
            if error_msg is None:
                success_pairs.append((idx, value))
            else:
                failure_count += 1
    finally:
        if owns_pool:
            pool.close()
            pool.join()

    success_pairs.sort(key=lambda pair: pair[0])
    successful_samples = [samples_prop[idx] for idx, _ in success_pairs]
    log_f_prop_results = [value for _, value in success_pairs]

    return successful_samples, log_f_prop_results, failure_count


def bridge_sampling_ln(
    f,
    g,
    samples_post,
    log_post,
    samples_prop,
    tol=1e-2,
    max_iter=5000,
    estimation_label: Optional[str] = None,
    verbose: bool = False,
    pool=None,
    num_workers: Optional[int] = None,
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
        pool: Optional parallel pool. Accepts:
            - None: run serially.
            - int: create an internal multiprocessing pool with that many workers.
            - any object with a ``map`` method: treated as an external pool.
        num_workers (Optional[int]): Deprecated alias for ``pool`` kept for
            backward compatibility.

    Returns:
        list[float, float]: ``[log_evidence, rmse_estimate]`` where the second
        element is an approximate RMSE estimate computed via ``compute_bridge_rmse``.

    Suggestions:
        - If ``f`` occasionally fails (e.g., numerical domain errors), this
          function automatically skips those proposal samples and continues.
        - Use a moderate ``tol`` (1e‑4 to 1e‑3) and cap ``max_iter`` to prevent
          long runs on difficult posteriors.
    """
    if pool is not None and num_workers is not None:
        logger.warning("Both pool and num_workers were provided; using pool and ignoring num_workers.")

    # Compute logs of densities for posterior samples
    log_f_post = log_post
    log_g_post = g(samples_post.T)

    # Compute logs of densities for proposal samples
    num_samples = len(samples_prop)
    pool_spec = pool if pool is not None else num_workers
    successful_samples, log_f_prop_results, failure_count = _evaluate_samples_parallel(
        f, samples_prop, pool_spec, logger, verbose=verbose
    )

    # Rebuild arrays from the lists of successful evaluations
    samples_prop = np.array(successful_samples)
    log_f_prop = np.array(log_f_prop_results)
    if failure_count > 0:
        logger.warning(
            "Evaluation of target distribution failed for %s samples, which were skipped.",
            failure_count,
        )

    # As a safeguard, filter out any remaining non-finite values (e.g., if f(theta) returned inf)
    finite_mask = np.isfinite(log_f_prop)

    samples_prop = samples_prop[finite_mask]
    log_f_prop = log_f_prop[finite_mask]
    message = (
        f"Filtered proposal samples: {len(samples_prop)} valid samples out of {num_samples} total samples."
    )
    _maybe_print(message, verbose=verbose)
    logger.info(message)
    # Now compute log_g for the filtered samples. g expects (n_dims, n_samples)
    log_g_prop = g(samples_prop.T)

    N1 = len(log_f_post)
    N2 = len(log_f_prop)

    if N2 == 0:
        warning_msg = "No valid samples from the proposal distribution. Bridge sampling failed."
        _maybe_print(f"\n{warning_msg}", verbose=verbose)
        logger.warning(warning_msg)
        return [np.nan, np.nan]

    s1 = N1 / (N1 + N2)
    s2 = N2 / (N1 + N2)

    # Initial guess for p(y) using importance sampling on proposal samples:
    log_p_old = logsumexp(log_f_prop - log_g_prop) - np.log(N2)

    term1 = np.log(s1) + log_f_prop
    term1_post = np.log(s1) + log_f_post

    last_iteration_msg = ""
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
        last_iteration_msg = f"iteration: {t+1} log(z) old: {log_p_old} log(z) New: {log_p_new}"
        if verbose:
            print(last_iteration_msg, end="\r")
        logger.info(last_iteration_msg)
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
            success_msg = f"Converged in {t+1} iterations. log(z): {log_p_final:.4f} +/-: {rmse_est:.4f}"
            if verbose:
                if estimation_label is not None:
                    block = "\n".join(
                        line for line in (estimation_label, last_iteration_msg, success_msg) if line
                    )
                    _maybe_print(block, verbose=True)
                else:
                    _maybe_print(last_iteration_msg, verbose=True)
                    _maybe_print(success_msg, verbose=True)
            logger.info(
                "Converged in %s iterations. log(z): %.4f +/-: %.4f",
                t + 1,
                log_p_final,
                rmse_est,
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
    failure_msg = f"Convergence not reached within {max_iter} iterations."
    final_msg = f"Final log(z): {log_p_final:.4f} +/-: {rmse_est:.4f}"
    if verbose:
        if estimation_label is not None:
            block_lines = [estimation_label]
            if last_iteration_msg:
                block_lines.append(last_iteration_msg)
            block_lines.extend([failure_msg, final_msg])
            _maybe_print("\n".join(block_lines), verbose=True)
        else:
            _maybe_print(failure_msg, verbose=True)
            _maybe_print(final_msg, verbose=True)
    logger.warning("Convergence not reached within %s iterations.", max_iter)
    logger.info("Final log(z): %.4f +/-: %.4f", log_p_final, rmse_est)

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
