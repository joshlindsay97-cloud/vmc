#!/usr/bin/env python3
"""
Speedup plot for parallel VMC on Wee ARCHIE (Task 4, PH510 Assignment 5).
Timing data from mpirun runs with 1, 2, 4, 8 MPI ranks.
"""

import matplotlib.pyplot as plt

# Timing results from Wee ARCHIE slurm output
ranks  = [1,      2,      4,      8    ]
times  = [4702.6, 2649.6, 1487.2, 946.0]

# Speedup = time with 1 rank / time with N ranks
speedup = [times[0] / t for t in times]
ideal   = [float(n)  for n in ranks]

# Efficiency = speedup / N ranks (perfect parallel = 100%)
efficiency = [s / n * 100 for s, n in zip(speedup, ranks)]

# Print table
print(f"{'Ranks':>6}  {'Time (s)':>10}  {'Speedup':>8}  {'Efficiency':>12}")
print("-" * 45)
for n, t, s, e in zip(ranks, times, speedup, efficiency):
    print(f"{n:>6}  {t:>10.1f}  {s:>8.2f}  {e:>11.1f}%")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Execution time
ax1.bar([str(n) for n in ranks], times, color="steelblue")
ax1.set_xlabel("Number of MPI ranks")
ax1.set_ylabel("Wall-clock time (s)")
ax1.set_title("Execution time vs MPI ranks\n")

# Speedup
ax2.plot(ranks, speedup, "o-", color="darkorange", label="Actual speedup")
ax2.plot(ranks, ideal,   "k--",                    label="Ideal speedup")
ax2.set_xlabel("Number of MPI ranks")
ax2.set_ylabel("Speedup")
ax2.set_title("Parallel speedup\n")
ax2.legend()
ax2.set_xticks(ranks)

plt.tight_layout()
plt.savefig("vmc_speedup.png", dpi=150)
plt.show()
print("Saved to vmc_speedup.png")
