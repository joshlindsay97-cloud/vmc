"""
Variational Monte Carlo (VMC) for the hydrogen atom 1s ground state.

Author: Josh Lindsay, University of Strathclyde (2025)
Tasks 1 and 2 of PH510 Assignment 5.
Atomic units: e = hbar = me = 1. Energy in Hartree.
"""

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Trial wavefunction and local energy
# =============================================================================

def psi_trial(r, alpha):
    """
    Trial wavefunction for the hydrogen 1s state.
    psi_T(r, alpha) = alpha * exp(-alpha * r)
    """
    return alpha * np.exp(-alpha * r)


def local_energy(r, alpha):
    """
    Analytical local energy for the hydrogen atom trial wavefunction.
    E_L(r) = (1/psi_T) * H * psi_T = -1/r - (alpha/2)(alpha - 2/r)

    At the exact solution alpha=1, E_L(r) = -0.5 everywhere (zero variance).
    """
    return -1.0 / r - (alpha / 2.0) * (alpha - 2.0 / r)


def log_prob(r, alpha):
    """
    Log of the 3D radial probability density including volume element:
        P(r) proportional to |psi_T|^2 * r^2
    so log P = 2*log|psi_T| + 2*log(r)
    The r^2 factor is essential — without it the walker spends too much
    time near r=0 where 1/r diverges, causing large energy fluctuations.
    """
    return 2.0 * np.log(np.abs(psi_trial(r, alpha))) + 2.0 * np.log(r)


# =============================================================================
# Metropolis Monte Carlo sampler
# =============================================================================

def metropolis_vmc(alpha, n_steps=200000, step_size=0.5, r_init=1.0, rng=None):
    """
    Metropolis sampling of P(r) = |psi_T(r, alpha)|^2 * r^2
    to estimate <H> and Var(E_L).

    Parameters
    ----------
    alpha     : float  - variational parameter
    n_steps   : int    - number of Monte Carlo steps
    step_size : float  - maximum displacement per step
    r_init    : float  - starting position (r > 0)
    rng       : numpy RNG

    Returns
    -------
    energy      : float - estimated <H>
    variance    : float - estimated Var(E_L)
    accept_rate : float - fraction of accepted moves
    """
    if rng is None:
        rng = np.random.default_rng()

    r_current     = r_init
    log_p_current = log_prob(r_current, alpha)

    energies   = np.zeros(n_steps)
    n_accepted = 0

    for i in range(n_steps):
        r_proposed = r_current + rng.uniform(-step_size, step_size)

        if r_proposed > 1e-6:
            log_p_proposed = log_prob(r_proposed, alpha)
            if np.log(rng.uniform()) < log_p_proposed - log_p_current:
                r_current     = r_proposed
                log_p_current = log_p_proposed
                n_accepted   += 1

        energies[i] = local_energy(r_current, alpha)

    energy   = np.mean(energies)
    variance = np.var(energies)
    accept_rate = n_accepted / n_steps

    return energy, variance, accept_rate


# =============================================================================
# Scan over alpha values
# =============================================================================

def scan_alpha(alpha_values, n_steps=200000, step_size=0.5, seed=42):
    """Run VMC for a range of alpha values and collect results."""
    rng     = np.random.default_rng(seed)
    results = []

    for alpha in alpha_values:
        energy, variance, accept_rate = metropolis_vmc(
            alpha, n_steps=n_steps, step_size=step_size, rng=rng
        )
        results.append({
            "alpha": alpha, "energy": energy,
            "variance": variance, "accept_rate": accept_rate,
        })
        print(f"alpha={alpha:.2f} | <E>={energy:+.5f} Ha | "
              f"Var={variance:.5f} | Accept={accept_rate:.2%}")

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_results(results):
    """Plot energy and variance as functions of alpha."""
    alphas    = [r["alpha"]    for r in results]
    energies  = [r["energy"]   for r in results]
    variances = [r["variance"] for r in results]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    ax1.plot(alphas, energies, "o-", color="steelblue", label="VMC <E>")
    ax1.axhline(-0.5, color="crimson", linestyle="--", label="Exact: -0.5 Ha")
    ax1.set_ylabel("Energy (Hartree)")
    ax1.legend()
    ax1.set_title("VMC: Hydrogen atom 1s ground state")

    ax2.plot(alphas, variances, "s-", color="darkorange", label="Variance")
    ax2.axvline(1.0, color="crimson", linestyle="--", label="alpha=1 (exact)")
    ax2.set_xlabel("Variational parameter alpha")
    ax2.set_ylabel("Variance of E_L")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("vmc_hydrogen_results.png", dpi=150)
    plt.show()
    print("Figure saved to vmc_hydrogen_results.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    alpha_values = np.linspace(0.5, 1.5, 11)

    print("Running VMC for hydrogen atom...\n")
    print(f"{'alpha':>6}  {'<E> (Ha)':>12}  {'Var':>10}  {'Accept':>8}")
    print("-" * 45)

    results = scan_alpha(alpha_values, n_steps=200000, step_size=0.5)

    best = min(results, key=lambda r: r["variance"])
    print(f"\nBest alpha = {best['alpha']:.2f}")
    print(f"  Energy   = {best['energy']:+.5f} Ha  (exact: -0.50000 Ha)")
    print(f"  Variance = {best['variance']:.6f}   (exact: 0.000000)")

    plot_results(results)

