"""
Variational Monte Carlo for two hard-sphere bosons in a 3D harmonic trap.
Parallelised version using mpi4py (MPI for Python).

Adapted from the parallel VMC code for the Helium atom by Magnar K. Bugge,
as found in the repository:
    Hjorth-Jensen, M. "Advanced Topics in Computational Physics:
    Computational Quantum Mechanics", CompPhysics/ComputationalPhysics2,
    LecturePrograms/programs/VMC-pypar/, GitHub (2021).
    Licensed under CC0 1.0 Universal (public domain):
    https://creativecommons.org/publicdomain/zero/1.0/

Changes from the original:
  - Updated from Python 2 to Python 3 (xrange -> range, print statements)
  - pypar replaced with mpi4py, the modern MPI interface for Python 3.
    The send/receive pattern is kept identical to the original pypar version:
      * Each rank runs MCcycles/nprocs steps independently
      * Non-master ranks send (esum, squaresum, N, accepted) to rank 0
      * Rank 0 receives and accumulates results from all other ranks
  - Physics replaced: Helium atom -> two hard-sphere bosons in 3D harmonic trap
  - Trial wavefunction: Psi = exp(-alpha/2 * |R|^2 + r12/(1 + beta*r12))
  - Local energy computed numerically via finite differences
  - Scan over two variational parameters (alpha, beta) instead of one

Run with:
    mpirun -n <nprocs> python3 vmc_bosons_parallel.py

Task 4 of PH510 Assignment 5.
Atomic units: e = hbar = me = 1. Energy in Hartree.
"""

import random
import time
from math import sqrt, exp
import matplotlib.pyplot as plt
from mpi4py import MPI  # pylint: disable=no-name-in-module
from numpy import zeros, double, linspace, array, meshgrid, unravel_index, argmin

# =============================================================================
# MPI setup — mirrors pypar.size() and pypar.rank()
# =============================================================================

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()   # total number of MPI ranks (was pypar.size())
myid = comm.Get_rank()     # this rank's id           (was pypar.rank())

# =============================================================================
# Parameters
# =============================================================================

MCYCLES = 2000000   # Total MC cycles split across all ranks
MCYCLES2 = 10000   # Cycles for delta optimisation
DELTA_MIN = 0.01
DELTA_MAX = 3.0
TOLERANCE = 0.01
HARD_CORE = 0.0043
OMEGA = 1.0
H = 0.001           # finite difference step
NDIM = 3            # number of dimensions


# =============================================================================
# Physics functions — identical to serial version
# =============================================================================

def has_singularity(r1, r2):
    """Check for hard-core overlap. Adapted from hasSingularity() in original."""
    dist = sqrt(sum((r1[k]-r2[k])**2 for k in range(NDIM)))
    return dist < HARD_CORE


def psi_trial(r1, r2, alpha, beta):
    """
    Trial wavefunction for two bosons in a 3D harmonic trap:
        Psi = exp(-alpha/2 * (|r1|^2 + |r2|^2)) * exp(r12 / (1 + beta*r12))
    Replaces Psi_trial() from original Helium code.
    """
    r_sq = sum(r1[k]**2 for k in range(NDIM)) + sum(r2[k]**2 for k in range(NDIM))
    dist = sqrt(sum((r1[k]-r2[k])**2 for k in range(NDIM)))
    u = dist / (1.0 + beta * dist)
    return exp(-alpha * r_sq / 2.0 + u)


def local_energy(r1, r2, alpha, beta):
    """
    Local energy via numerical finite differences.
    Identical to serial version — ensures parallel and serial give same results.
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
    Each MPI rank calls this with num_cycles = MCYCLES/nprocs steps, exactly
    as the original pypar code called runMC(MCcycles/nprocs, delta).

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
# Delta optimisation — bisection from original
# =============================================================================

def difference(delta, alpha, beta):
    """Target 50% acceptance. Adapted from difference() in original."""
    _, _, _, accepted = run_mc(MCYCLES2, delta, alpha, beta)
    return accepted / MCYCLES2 - 0.5


def optimise_delta(alpha, beta):
    """Bisection for optimal delta. Same logic as original."""
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
# Main — MPI parallelisation mirrors original pypar pattern
# =============================================================================

alpha_values = array([0.95, 1.00, 1.05, 1.10, 1.15])
beta_values = linspace(0.22, 0.30, 7)
num_alpha = len(alpha_values)
num_beta = len(beta_values)

energy_grid = zeros((num_alpha, num_beta), double)
variance_grid = zeros((num_alpha, num_beta), double)

if myid == 0:
    with open("vmc_bosons_results.txt", "w", encoding="utf-8") as outfile:
        outfile.write("# alpha  beta  energy  variance  acceptance\n")
    print(f"Running parallel VMC with {nprocs} MPI ranks")
    print(f"Total cycles: {MCYCLES} ({MCYCLES//nprocs} per rank)\n")
    print(f"{'alpha':>7}  {'beta':>6}  {'<E>':>10}  {'Var':>10}  {'Accept':>8}")
    print("-" * 52)

t_start = time.time()

for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):

        # All ranks find optimal delta independently
        opt_delta = optimise_delta(alpha, beta)

        # Each rank runs its share — mirrors: runMC(MCcycles/nprocs, delta)
        esum, squaresum, num_samples, accepted = run_mc(
            MCYCLES // nprocs, opt_delta, alpha, beta
        )

        # Non-master ranks send to master — mirrors: pypar.send(..., destination=0)
        if myid != 0:
            comm.send((esum, squaresum, num_samples, accepted), dest=0)

        # Master receives and accumulates — mirrors: pypar.receive(source=i)
        if myid == 0:
            for src in range(1, nprocs):
                r_esum, r_sq, r_n, r_acc = comm.recv(source=src)
                esum += r_esum
                squaresum += r_sq
                num_samples += r_n
                accepted += r_acc

            mean_e = esum / num_samples
            mean_e2 = squaresum / num_samples
            var = mean_e2 - mean_e**2
            accept = accepted / MCYCLES

            energy_grid[i, j] = mean_e
            variance_grid[i, j] = var

            print(f"a={alpha:.2f}  b={beta:.3f}  <E>={mean_e:+.4f}  "
                  f"Var={var:.4f}  Accept={accept:.1%}")

            with open("vmc_bosons_results.txt", "a", encoding="utf-8") as outfile:
                outfile.write(f"{alpha:.3f}  {beta:.3f}  {mean_e:.5f}  "
                              f"{var:.5f}  {accept:.3f}\n")

if myid == 0:
    t_total = time.time() - t_start

    best_idx = unravel_index(argmin(energy_grid), energy_grid.shape)
    best_alpha = alpha_values[best_idx[0]]
    best_beta = beta_values[best_idx[1]]
    best_energy = energy_grid[best_idx]
    best_var = variance_grid[best_idx]

    print(f"\nBest parameters found:")
    print(f"  alpha    = {best_alpha:.3f}")
    print(f"  beta     = {best_beta:.3f}")
    print(f"  <E>      = {best_energy:.5f} a.u.  (expected ~3.0)")
    print(f"  Variance = {best_var:.5f}")
    print(f"  Total time: {t_total:.1f}s with {nprocs} ranks")

    beta_mesh, alpha_mesh = meshgrid(beta_values, alpha_values)
    fig = plt.figure(figsize=(9, 6))
    axes = fig.add_subplot(111, projection="3d")
    surf = axes.plot_surface(beta_mesh, alpha_mesh, energy_grid,
                             cmap="coolwarm", alpha=0.85)
    axes.set_xlabel("beta")
    axes.set_ylabel("alpha")
    axes.set_zlabel("<E> / a.u.")
    axes.set_title(f"VMC: Two bosons in 3D harmonic trap ({nprocs} MPI ranks)")
    fig.colorbar(surf, shrink=0.5, label="Energy (a.u.)")
    plt.tight_layout()
    plt.savefig("vmc_bosons_surface.png", dpi=150)
    plt.show()

MPI.Finalize()
