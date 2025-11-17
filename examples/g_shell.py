import logging
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import numpy as np
from dynesty import NestedSampler
from dynesty import utils as dyfunc
from scipy.special import logsumexp

import morphZ
from morphZ import setup_logging

setup_logging(level=logging.INFO)
# Prior cube: [-L, L]^D 
L = 6.0
ndim = 30
prior_volume = (2*L)**ndim

# Parameters of the twin shells (from paper)
w1 = w2 = 0.1  # thickness of both shells
r1 = r2 = 2.0  # radius of both shells
c1 = np.array([-3.5] + [0.0] * (ndim - 1))  # center of first shell
c2 = np.array([3.5] + [0.0] * (ndim - 1))   # center of second shell

def logprior(theta):
    """Uniform prior over [-L, L]^D."""
    if np.all(np.abs(theta) <= L):
        return -np.log(prior_volume)
    else:
        return -np.inf

def loglikelihood(theta):
    """
    Twin Gaussian shell log-likelihood from equation (38) in the paper.
    L(θ) = (1/√(2πw₁)) * exp[-(|θ-c₁|-r₁)²/(2w₁²)] + (1/√(2πw₂)) * exp[-(|θ-c₂|-r₂)²/(2w₂²)]
    """
    # Distance from centers
    r_c1 = np.linalg.norm(theta - c1)
    r_c2 = np.linalg.norm(theta - c2)
    
    # Log-likelihood components for each shell
    # Note: We work in log space for numerical stability
    log_norm1 = -0.5 * np.log(2 * np.pi * w1**2)
    log_norm2 = -0.5 * np.log(2 * np.pi * w2**2)
    
    log_exp1 = log_norm1 - 0.5 * ((r_c1 - r1) / w1)**2
    log_exp2 = log_norm2 - 0.5 * ((r_c2 - r2) / w2)**2
    
    # Use logsumexp for numerical stability when adding exponentials
    return logsumexp([log_exp1, log_exp2])

def prior_transform(u):
    """
    Map unit cube [0,1]^D to [-L, L]^D.
    u: array in [0,1]^D
    """
    return -L + 2*L*u

# Optional: Visualization function for 2D case
def plot_twin_shells_2d():
    """Plot the twin Gaussian shells for visualization (2D only)."""
    if ndim != 2:
        print("Visualization only available for 2D case")
        return
    
    # Create grid
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate likelihood on grid
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            theta = np.array([X[i,j], Y[i,j]])
            Z[i,j] = np.exp(loglikelihood(theta))
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(label='Likelihood')
    plt.xlabel('θ₁')
    plt.ylabel('θ₂')
    plt.title('Twin Gaussian Shells')
    
    # Mark centers
    plt.plot(c1[0], c1[1], 'r*', markersize=15, label='Center 1')
    plt.plot(c2[0], c2[1], 'r*', markersize=15, label='Center 2')
    
    # Draw circles showing the shell radii
    circle1 = plt.Circle(c1[:2], r1, fill=False, color='red', linestyle='--', alpha=0.7)
    circle2 = plt.Circle(c2[:2], r2, fill=False, color='red', linestyle='--', alpha=0.7)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    
    plt.legend()
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.show()

# Example usage with dynesty
def run_dynesty_example():
    """Example of how to use this with dynesty."""
    try:
        import dynesty
        from dynesty import plotting as dyplot
        
        # Run nested sampling
        sampler = dynesty.NestedSampler(loglikelihood, prior_transform, ndim)
        sampler.run_nested()
        results = sampler.results
        
        print(f"Log evidence: {results.logz[-1]:.2f} ± {results.logzerr[-1]:.2f}")
        
        # Plot results
        fig, axes = dyplot.runplot(results)
        plt.show()
        
        return results
        
    except ImportError:
        print("Dynesty not installed. Install with: pip install dynesty")
        return None

# Analytical log-evidence values from the paper (for reference)
analytical_logz = {
    20: -36.09, # 20D case from paper
    30: -60.13  # 30D case from paper
}

# -------------------------------
# Run dynesty Nested Sampling
# -------------------------------
def run_nested_sampling(n_runs: int = 1, nlive: int = 200):
    """Run dynesty nested sampling and return posterior samples plus logZ stats."""
    log_z_NS = np.zeros(n_runs)
    log_z_NS_err = np.zeros(n_runs)
    posterior_samples = None

    for i in range(n_runs):
        sampler = NestedSampler(
            loglikelihood,
            prior_transform,
            ndim,
            nlive=nlive,      # number of live points; increase for accuracy
            sample="rwalk",   # random-walk proposals; can try "unif", "slice"
            bound="multi",    # bounding method ("multi" good for multimodal)
        )

        print("Running dynesty Nested Sampling on Gaussian shell...")
        sampler.run_nested(dlogz=0.1, print_progress=True)
        res = sampler.results
        samples, weights = res.samples, np.exp(res.logwt - res.logz[-1])
        posterior_samples = dyfunc.resample_equal(samples, weights)
        log_z_NS[i] = res.logz[-1]
        log_z_NS_err[i] = res.logzerr[-1]

    return posterior_samples, log_z_NS, log_z_NS_err


def lnprobfn(theta):
    """Log-probability combining prior and likelihood."""
    return logprior(theta) + loglikelihood(theta)


def main():
    print(f"Twin Gaussian Shell Problem Setup:")
    print(f"Dimensions: {ndim}")
    print(f"Prior bounds: [{-L}, {L}]^{ndim}")
    print(f"Shell centers: c1={c1}, c2={c2}")
    print(f"Shell radii: r1={r1}, r2={r2}")
    print(f"Shell widths: w1={w1}, w2={w2}")

    if ndim == 2:
        print("\nGenerating 2D visualization...")
        plot_twin_shells_2d()

    posterior_samples, log_z_NS, log_z_NS_err = run_nested_sampling()
    true_logz = analytical_logz.get(ndim, -36.09)
    print("\n==== RESULTS ====")
    print(f"dynesty estimated logZ = {log_z_NS.mean():.3f} ± {log_z_NS_err.mean():.3f}")
    print(f"True logZ (literature) = {true_logz:.3f}")
    print(f"Error = {log_z_NS.mean() - true_logz:.3f}")

    samples = posterior_samples[::5, :]
    tot_len, ndim_samples = samples.shape
    print("Total samples:", tot_len, "Dimensions:", ndim_samples)
    n_cpus = 8
    print(f"Running with {n_cpus} CPUs.")
    # Use a shared-memory pool for both log-prob evaluation and MorphZ evidence.
    with Pool(n_cpus) as pool:
        log_prob = np.array(pool.map(lnprobfn, samples))
        log_p_estimate = morphZ.evidence(
            samples,
            log_prob,
            lnprobfn,
            n_resamples=10000,
            thin=1,
            n_estimations=5,
            morph_type="2_group",
            kde_bw="silverman",
            output_path="./morphZ_gaussian_shell_group/",
            pool=pool,  # external pool passed through to bridge sampling
        )
    print("True:", true_logz)
    print("MorphZ logZ estimates:", log_p_estimate)


if __name__ == "__main__":
    main()
