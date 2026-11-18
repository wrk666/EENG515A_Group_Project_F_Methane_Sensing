"""
generate_lawnmower_data.py

Fly a lawnmower pattern over the simulated Gaussian plume and save the samples for the team.
Units everywhere: concentration [ppm], emission rate [g/s], positions [m], wind [m/s].
Only Particle is needed (no SrmEstimator / EMGR / MOABS data).

Outputs
  lawnmower_data.npz     all arrays, see the keys in np.savez_compressed below
  lawnmower_samples.csv  one noisy realization, human readable
  lawnmower.gif          animation of the flight (set MAKE_GIF = False to skip)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from main import X_MIN, X_MAX, Y_MIN, Y_MAX

from ParticleClass import Particle

# # ------------------------------------------------------------------
# # CONFIG
# # ------------------------------------------------------------------
# SOURCE_XY = (80.0, 60.0)          # true source location [m]
# Q_TRUE = 0.0349                   # true emission rate [g/s] (controlled-release rate in the paper)
# WIND = np.array([1.0, 0.0])       # constant wind [u, v] [m/s]; air moves toward +x
# MEAS_Z = 5.0                      # UAV height [m]
#
# X_MIN, X_MAX = 0.0, 200.0         # survey area [m]
# Y_MIN, Y_MAX = 0.0, 200.0
# LEG_SPACING = 10.0                # distance between legs [m]; legs run along y (crosswind)
# SAMPLE_SPACING = 2.0              # distance between samples along a leg [m]
# DT = 0.5                          # sampling period [s] (2 Hz sensor; 2 m / 0.5 s = 4 m/s flight speed)
#
# NOISE_SIGMAS_PPM = [0.01, 0.05, 0.1]   # additive Gaussian noise levels [ppm]
# N_REPEATS = 100                   # independent noisy copies per noise level
# N_GRID = 25                       # candidate source grid is N_GRID x N_GRID cells
# SEED = 0
# OUT_NPZ = "Out/lawnmower_data.npz"
# OUT_CSV = "Out/lawnmower_samples.csv"
#
# MAKE_GIF = True                   # also save an animation of the flight
# OUT_GIF = "Out/lawnmower.gif"
# GIF_LEVEL = 1                     # which noise level to show (index into NOISE_SIGMAS_PPM)
# GIF_STEP = 25                     # samples per frame (bigger = smaller file)




def make_gif(p, traj, t, wind, y_clean, y, src, sigma, out=OUT_GIF, step=GIF_STEP):
    """Left: true plume (background) + UAV path + measurements. Right: measured ppm vs time."""
    vmax = y_clean.max()
    cmap = "magma_r"                                   # low = near white, high = dark: visible on white
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), dpi=100)

    # background: noise-free plume field at flight height (so the plume is visible before the UAV reaches it)
    gx, gy = np.meshgrid(np.linspace(X_MIN, X_MAX, 100), np.linspace(Y_MIN, Y_MAX, 100))
    gz = np.array([[p.getTrueOutput(np.array([x, y_, traj[0, 2]]), wind[0]) for x in gx[0]] for y_ in gy[:, 0]])
    ax.contourf(gx, gy, gz, levels=np.linspace(0.02, vmax, 12), cmap=cmap, vmin=0, vmax=vmax, alpha=0.35, extend="max")

    ax.set_xlim(traj[:, 0].min() - 5, traj[:, 0].max() + 5)
    ax.set_ylim(traj[:, 1].min() - 5, traj[:, 1].max() + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]  (wind direction →)")
    ax.set_ylabel("y [m]")
    ax.plot(*src, "rx", ms=10, mew=2, label="source")
    ax.annotate("", xy=(30, 190), xytext=(10, 190), arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(10, 178, f"wind {np.linalg.norm(wind[0]):.0f} m/s", fontsize=9)
    path, = ax.plot([], [], color="0.75", lw=1)
    sc = ax.scatter([], [], c=[], s=16, cmap=cmap, vmin=0, vmax=vmax, edgecolors="none")
    uav, = ax.plot([], [], "wo", mec="k", ms=8)
    fig.colorbar(sc, ax=ax, shrink=0.8, label="measured [ppm] (< 0.1 shown blank)")
    ax.legend(loc="lower right")

    ax2.set_xlim(0, t[-1])
    ax2.set_ylim(y.min() - 0.05, y.max() + 0.05)
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("concentration [ppm]")
    ax2.axhline(0.1, color="r", ls="--", lw=1, label="0.1 ppm sensor resolution")
    ln_noisy, = ax2.plot([], [], lw=0.8, label=f"noisy (σ={sigma} ppm)")
    ln_clean, = ax2.plot([], [], "k", lw=1.2, label="clean")
    ax2.legend(loc="upper left", fontsize=8)
    title = fig.suptitle("")
    fig.tight_layout()

    def update(f):
        n = min((f + 1) * step, len(t))
        path.set_data(traj[:n, 0], traj[:n, 1])
        sc.set_offsets(traj[:n, :2])
        sc.set_array(np.where(y[:n] < 0.1, 0.0, y[:n]))   # below sensor resolution -> shown as blank
        uav.set_data([traj[n - 1, 0]], [traj[n - 1, 1]])
        ln_noisy.set_data(t[:n], y[:n])
        ln_clean.set_data(t[:n], y_clean[:n])
        title.set_text(f"Lawnmower sampling: t = {t[n - 1]:.0f} s, sample {n}/{len(t)}")

    FuncAnimation(fig, update, frames=int(np.ceil(len(t) / step))).save(out, writer=PillowWriter(fps=12))
    plt.close(fig)
