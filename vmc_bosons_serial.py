"""
Variational Monte Carlo for two hard-sphere bosons in a 3D harmonic trap.

Adapted from the serial VMC code for the Helium atom by Magnar K. Bugge,
as found in the repository:
    Hjorth-Jensen, M. "Advanced Topics in Computational Physics:
    Computational Quantum Mechanics", CompPhysics/ComputationalPhysics2,
    LecturePrograms/programs/VMC-py/, GitHub (2021).
    Licensed under CC0 1.0 Universal (public domain):
    https://creativecommons.org/publicdomain/zero/1.0/

Changes from the original:
  - Updated from Python 2 to Python 3 (xrange -> range, print statements)
  - Physics replaced: Helium atom -> two hard-sphere bosons in 3D harmonic trap
  - Trial wavefunction: Psi = exp(-alpha/2 * |R|^2 + r12/(1 + beta*r12))
  - Local energy computed numerically via finite differences
  - Scan over two variational parameters (alpha, beta) instead of one
  - Added 3D surface and 2D slice plots of energy landscape

Task 3 of PH510 Assignment 5.
Atomic units: e = hbar = me = 1. Energy in Hartree.
"""

import random
from math import sqrt, exp
import matplotlib.pyplot as plt
from numpy import zeros, double, linspace, array, meshgrid, unravel_index, argmin

# =============================================================================
# Parameters
# =============================================================================

MCYCLES = 100000    # MC cycles for main run
MCYCLES2 = 10000   # MC cycles for delta optimisation
DELTA_MIN = 0.01
DELTA_MAX = 3.0
TOLERANCE = 0.01
HARD_CORE = 0.0043  # hard sphere diameter a
OMEGA = 1.0         # harmonic trap frequency
H = 0.001           # finite difference step for numerical local energy
NDIM = 3            # number of dimensions


# =============================================================================
# Physics functions
# =============================================================================

def has_singularity(r1, r2):
    """
    Check for hard-core overlap between particles.
    Adapted from hasSingularity() in original Helium code.
    """
    dist = sqrt(sum((r1[k]-r2[k])**2 for k in range(NDIM)))
    return dist < HARD_CORE


def psi_trial(r1, r2, alpha, beta):
    """
    Trial wavefunction for two bosons in a 3D harmonic trap (Jastrow form):
        Psi = exp(-alpha/2 * (|r1|^2 + |r2|^2)) * exp(r12 / (1 + beta*r12))

    The first factor confines both particles in the harmonic trap.
    The Jastrow factor exp(u(r12)) introduces two-particle correlations,
    keeping particles from getting too close.

    Replaces Psi_trial() from original Helium code.
    """
    r_sq = sum(r1[k]**2 for k in range(NDIM)) + sum(r2[k]**2 for k in range(NDIM))
    dist = sqrt(sum((r1[k]-r2[k])**2 for k in range(NDIM)))
    u = dist / (1.0 + beta * dist)
    return exp(-alpha * r_sq / 2.0 + u)


def local_energy(r1, r2, alpha, beta):
    """
    Local energy E_L = (1/Psi) H Psi computed numerically via finite differences.
    H = -1/2 nabla_1^2 - 1/2 nabla_2^2 + 1/2 omega^2 (|r1|^2 + |r2|^2) + V_int

    Kinetic energy: central differences on all NDIM coordinates per particle.
    Potential energy: 1/2 * omega^2 * |R|^2 computed analytically.
    Hard sphere interaction handled by Metropolis rejection.

    Replaces E_local() from original Helium code.
    """
    psi0 = psi_trial(r1, r2, alpha, beta)
    laplacian = 0.0

    for coord in range(NDIM):
        r1f = list(r1)
        r1f[coord] += H
        r1b = list(r1)
        r1b[coord] -= H
        laplacian += (psi_trial(r1f, r2, alpha, beta)
                      - 2*psi0
                      + psi_trial(r1b, r2, alpha, beta))

        r2f = list(r2)
        r2f[coord] += H
        r2b = list(r2)
        r2b[coord] -= H
        laplacian += (psi_trial(r1, r2f, alpha, beta)
                      - 2*psi0
                      + psi_trial(r1, r2b, alpha, beta))

    laplacian /= H**2
    kinetic = -0.5 * laplacian / psi0
    r_sq = sum(r1[k]**2 for k in range(NDIM)) + sum(r2[k]**2 for k in range(NDIM))
    potential = 0.5 * OMEGA**2 * r_sq

    return kinetic + potential


# =============================================================================
# Monte Carlo sampler — adapted from runMC() in original
# =============================================================================

def run_mc(num_cycles, delta, alpha, beta):
    """
    Metropolis MC for two bosons in 3D.
    Structure directly adapted from runMC() in original Helium code.

    Returns: (energy sum, squared energy sum, sample count, accepted moves)
    """
    esum = 0.0
    squaresum = 0.0
    num_samples = 0
    accepted = 0

    r1 = [0.5*(random.random()*2 - 1) for _ in range(NDIM)]
    r2 = [0.5*(random.random()*2 - 1) for _ in range(NDIM)]

    for k in range(num_cycles):
        r1_trial = [r1[d] + delta*(random.random()*2 - 1) for d in range(NDIM)]
        r2_trial = [r2[d] + delta*(random.random()*2 - 1) for d in range(NDIM)]

        prob_trial = psi_trial(r1_trial, r2_trial, alpha, beta)**2
        prob_curr = psi_trial(r1, r2, alpha, beta)**2

        if prob_trial > prob_curr:
            accepted += 1
            r1, r2 = r1_trial, r2_trial
        elif random.random() < prob_trial / prob_curr:
            accepted += 1
            r1, r2 = r1_trial, r2_trial

        if (not has_singularity(r1, r2)) and k > num_cycles // 20:
            energy_local = local_energy(r1, r2, alpha, beta)
            esum += energy_local
            squaresum += energy_local**2
            num_samples += 1

    return esum, squaresum, num_samples, accepted


# =============================================================================
# Delta optimisation — bisection method from original
# =============================================================================

def difference(delta, alpha, beta):
    """
    Returns (acceptance rate - 0.5). Zero when acceptance is 50%.
    Adapted from difference() in original Helium code.
    """
    _, _, _, accepted = run_mc(MCYCLES2, delta, alpha, beta)
    return accepted / MCYCLES2 - 0.5


def optimise_delta(alpha, beta):
    """Bisection to find optimal delta. Same logic as original."""
    minimum = DELTA_MIN
    maximum = DELTA_MAX
    while maximum - minimum > TOLERANCE:
        if difference(minimum, alpha, beta) * difference(
                (minimum+maximum)/2, alpha, beta) < 0:
            maximum = (minimum + maximum) / 2
        else:
            minimum = (minimum + maximum) / 2
    return (minimum + maximum) / 2


# =============================================================================
# Parameter scan and plotting
# =============================================================================

def scan_parameters(alpha_vals, beta_vals):
    """Scan 2D grid of (alpha, beta) and collect energy and variance."""
    num_alpha = len(alpha_vals)
    num_beta = len(beta_vals)
    energy_grid = zeros((num_alpha, num_beta), double)
    variance_grid = zeros((num_alpha, num_beta), double)

    print(f"Scanning {num_alpha}x{num_beta} = {num_alpha*num_beta} combinations...\n")
    print(f"{'alpha':>7}  {'beta':>6}  {'<E>':>10}  {'Var':>10}  {'Accept':>8}")
    print("-" * 52)

    for i, alpha in enumerate(alpha_vals):
        for j, beta in enumerate(beta_vals):
            delta = optimise_delta(alpha, beta)
            esum, squaresum, num_samples, accepted = run_mc(MCYCLES, delta, alpha, beta)

            mean_e = esum / num_samples
            mean_e2 = squaresum / num_samples
            var = mean_e2 - mean_e**2
            accept = accepted / MCYCLES

            energy_grid[i, j] = mean_e
            variance_grid[i, j] = var

            print(f"a={alpha:.2f}  b={beta:.3f}  <E>={mean_e:+.4f}  "
                  f"Var={var:.4f}  Accept={accept:.1%}")

    return energy_grid, variance_grid


def plot_results(alpha_vals, beta_vals, energy_grid):
    """3D surface and 2D slice plots matching figure 11.1 in the handout."""
    beta_mesh, alpha_mesh = meshgrid(beta_vals, alpha_vals)

    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_subplot(111, projection="3d")
    surf = axes.plot_surface(beta_mesh, alpha_mesh, energy_grid,
                             cmap="coolwarm", alpha=0.85)
    axes.set_xlabel("beta")
    axes.set_ylabel("alpha")
    axes.set_zlabel("<E> / a.u.")
    axes.set_title("VMC: Two bosons in 3D harmonic trap")
    fig.colorbar(surf, shrink=0.5, label="Energy (a.u.)")
    plt.tight_layout()
    plt.savefig("vmc_bosons_surface.png", dpi=150)
    plt.show()

    _, axes2 = plt.subplots(figsize=(7, 5))
    for i, alpha in enumerate(alpha_vals):
        axes2.plot(beta_vals, energy_grid[i], "o-", label=f"alpha={alpha:.2f}")
    axes2.axhline(3.0, color="crimson", linestyle="--", label="Expected min ~3.0")
    axes2.set_xlabel("beta")
    axes2.set_ylabel("<E> / a.u.")
    axes2.set_title("VMC energy vs beta for different alpha values")
    axes2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("vmc_bosons_slices.png", dpi=150)
    plt.show()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    alpha_values = array([0.95, 1.00, 1.05, 1.10, 1.15])
    beta_values = linspace(0.22, 0.30, 7)

    e_grid, var_grid = scan_parameters(alpha_values, beta_values)

    best_idx = unravel_index(argmin(e_grid), e_grid.shape)
    best_alpha = alpha_values[best_idx[0]]
    best_beta = beta_values[best_idx[1]]
    best_energy = e_grid[best_idx]
    best_var = var_grid[best_idx]

    print(f"\nBest parameters found:")
    print(f"  alpha    = {best_alpha:.3f}")
    print(f"  beta     = {best_beta:.3f}")
    print(f"  <E>      = {best_energy:.5f} a.u.  (expected ~3.0)")
    print(f"  Variance = {best_var:.5f}")

    plot_results(alpha_values, beta_values, e_grid)
