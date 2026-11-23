"""
online_main.py

Step-by-step sampling loop (workflow steps 4-6 in the figure): at every step get ONE new sample,
update Q_hat with it, and choose where to fly next.
Run:  python online_main.py
"""

import numpy as np

from Alex_IPP.path_planner import choose_next_position

from Ruikun_Sensing.ParticleClass import Particle
from Ruikun_Sensing.GridCellClass import GridCell
from Ruikun_Sensing.SrmEstimator import SrmEstimator
from Ruikun_Sensing.EmgrEstimator import EmgrEstimator
from Ruikun_Sensing.QMetrics import q_metrics, print_q_report
# from main import (SOURCE_XY, Q_TRUE, WIND, MEAS_Z, X_MIN, X_MAX, Y_MIN, Y_MAX,
#                   LEG_SPACING, SAMPLE_SPACING, SEED, lawnmower, candidate_grid)

SOURCE_XY = (80.0, 60.0)          # true source location [m]
Q_TRUE = 0.0349                   # true emission rate [g/s] (controlled-release rate in the paper)
WIND = np.array([1.0, 0.0])       # constant wind [u, v] [m/s]; air moves toward +x
MEAS_Z = 5.0                      # UAV height [m]

X_MIN, X_MAX = 0.0, 200.0         # survey area [m]
Y_MIN, Y_MAX = 0.0, 200.0
LEG_SPACING = 10.0                # distance between legs [m]; legs run along y (crosswind)
SAMPLE_SPACING = 2.0              # distance between samples along a leg [m]
# DT = 0.5                          # sampling period [s] (2 Hz sensor; 2 m / 0.5 s = 4 m/s flight speed)
#
# NOISE_SIGMAS_PPM = [0.01, 0.05, 0.1]   # additive Gaussian noise levels [ppm]
# N_REPEATS = 100                   # independent noisy copies per noise level
N_GRID = 25                       # candidate source grid is N_GRID x N_GRID cells
SEED = 0

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
NOISE_SIGMA_PPM = 0.05        # sensor noise [ppm]
RIDGE_BETA = 100.0            # SrmEstimator lambda_reg; ppm units, see the sweep (1e-5 fails with noise)
DETECT_PPM = 0.2              # "plume detected" threshold [ppm] = 4 sigma (old value 8.5e-8 was in MOABS units)
N_STEPS = 2121                # same length as the batch lawnmower
STEP = 2.0                    # distance per move [m]; 2 m per 0.5 s sample = 4 m/s
MOVES = np.array([[STEP, 0, 0], [-STEP, 0, 0], [0, STEP, 0], [0, -STEP, 0]])

# calcScore weights (from test_srm.py; anchor term not ported yet)
LAMBDA_EMGR, LAMBDA_VISIT, LAMBDA_BOUNDARY, LAMBDA_ANCHOR = 0.01, 0.1, 2.0, 0.05

def candidate_grid(n=N_GRID):
    """Cell centres of the n x n candidate-source grid; index = iy * n + ix (so Q.reshape(n, n) is [iy, ix])."""
    dx, dy = (X_MAX - X_MIN) / n, (Y_MAX - Y_MIN) / n
    gx, gy = np.meshgrid(np.arange(X_MIN + dx / 2, X_MAX, dx), np.arange(Y_MIN + dy / 2, Y_MAX, dy))
    return np.column_stack([gx.ravel(), gy.ravel()])

def build_estimators(p):
    cand = candidate_grid()
    cells = []
    for c in cand:
        cell = GridCell(x=c, dx=(X_MAX - X_MIN) / 25, dy=(Y_MAX - Y_MIN) / 25, level=0)
        cell.addParticle(Particle(source_pos=c, params=[1.0, c[0], c[1], 0]))
        cells.append(cell)
    srm = SrmEstimator(cells, p, lambda_reg=RIDGE_BETA, dx=2, dy=2, dz=1)
    return cand, srm, EmgrEstimator(srm)


def boundary_and_visit(xp, visit_map):
    b = 1 / (xp[0] + 1) + 1 / (200 - xp[0] + 1) + 1 / (xp[1] + 1) + 1 / (200 - xp[1] + 1)
    return LAMBDA_BOUNDARY * b + LAMBDA_VISIT * visit_map[int(xp[0] // 2), int(xp[1] // 2)]


def lawnmower(x_min, x_max, y_min, y_max, leg_spacing, sample_spacing, z):
    """Back-and-forth legs along y, stepping in x between legs. Returns (k, 3)."""
    xs = np.arange(x_min, x_max + 1e-9, leg_spacing)
    ys = np.arange(y_min, y_max + 1e-9, sample_spacing)
    pts = []
    for i, x in enumerate(xs):
        for y in (ys if i % 2 == 0 else ys[::-1]):
            pts.append([x, y, z])
    return np.array(pts)

# ------------------------------------------------------------------
# DENOISER INTERFACE (Skylor's part)
# ------------------------------------------------------------------
# This is a class for system running without denoiser, Skylor can add WindowPCADenoiser in parameters of function "run" below.
class IdentityDenoiser:
    """Default: no denoising."""

    def update(self, y_raw):
        return y_raw

# An example class of PCADenoiser to fill
class WindowPCADenoiser:
    "Skylor can define the PCA Denoiser here"

    def __init__(self, train, window=21, n_components=15):
        from sklearn.decomposition import PCA

        self.windowSize = window
        self.history = []

        ## Create training windows within each training flight ##
        trainWindows = []

        # f is flight #
        # s is start #
        for f in train:
            for s in range(len(f) - self.windowSize + 1):
                trainWindows.append(f[s:s + self.windowSize])

        trainWindows = np.array(trainWindows)

        ## Fitting pca using the noisy training windows ##
        self.pca = PCA(n_components=n_components, svd_solver='full')
        self.pca.fit(trainWindows)

    # Input y_raw with measurement, output the denoised data.
    def update(self, y_raw):

        ## save the latest noisy measurement ##
        self.history.append(float(y_raw))

        ## keeping only the most recent readings needed for one window ##
        self.history = self.history[-self.windowSize:]

        #Using raw measuremeants until we have a complted window #
        if len(self.history) < self.windowSize:
            return float(y_raw)

        # current window for pca and project into pca coord. #
        currentWindow = np.array(self.history).reshape(1, -1)
        score = self.pca.transform(currentWindow)

        # Reconstruct windows and add back the saved training means #
        rebuitWindows = self.pca.inverse_transform(score)

        return float(rebuitWindows[0, -1])


def make_training_flights(n_flights=60, seed=1):
    """Independent noisy copies of the lawnmower flight, for fitting a denoiser offline (same recipe as main.py)."""
    rng = np.random.default_rng(seed)
    p = Particle(source_pos=SOURCE_XY, params=[Q_TRUE, SOURCE_XY[0], SOURCE_XY[1], 0])
    path = lawnmower(X_MIN, X_MAX, Y_MIN, Y_MAX, LEG_SPACING, SAMPLE_SPACING, MEAS_Z)
    y_clean = np.array([p.getTrueOutput(xm, WIND) for xm in path])
    return y_clean[None, :] + NOISE_SIGMA_PPM * rng.standard_normal((n_flights, len(path)))
# ------------------------------------------------------------------
# DENOISER INTERFACE (Skylor's part)
# ------------------------------------------------------------------



def run(policy, seed=SEED, n_steps=N_STEPS, denoiser=None):
    denoiser = denoiser or IdentityDenoiser()               # pass a FRESH instance per run (it keeps history)
    rng = np.random.default_rng(seed)
    p = Particle(source_pos=SOURCE_XY, params=[Q_TRUE, SOURCE_XY[0], SOURCE_XY[1], 0])
    cand, srm, emgr = build_estimators(p)

    path = lawnmower(X_MIN, X_MAX, Y_MIN, Y_MAX, LEG_SPACING, SAMPLE_SPACING, MEAS_Z)
    if policy != "lawnmower":
        path = path[::-1]                                   # start at the downwind end
    xm = path[0].copy()
    visit_map = np.zeros((101, 101))        # path reaches 200 m -> index 100
    detected = False
    anchor_xy, anchor_y = None, -np.inf
    log = {"pos": [], "y_clean": [], "y_raw": [], "y": [], "q_sum": [], "q_argmax": [], "q_neigh": [], "loc": [], "q_neigh_rel_err": []}   # y = what the estimator sees

    if policy == "svd":
        measurement_rows = np.empty((n_steps, len(cand)))

    for k in range(n_steps):
        # 1. one new noisy sample at the current position [ppm]
        y_clean = p.getTrueOutput(xm, WIND)                         # truth: for evaluation only
        y_raw = y_clean + NOISE_SIGMA_PPM * rng.standard_normal()
        # 2. TODO: denoise it (Skylor's part): see the DENOISER INTERFACE above
        y = denoiser.update(y_raw)
        # 3. TODO: update Q_hat (RLS) with it: Kevin's part, you can refer to function "rlsUpdate" in SrmEstimator.py
        row = srm.getCachedSrm(xm, WIND)
        srm.rlsUpdate(row, y)
        if policy == "svd":
            measurement_rows[k] = row.ravel()
        visit_map[int(xm[0] // 2), int(xm[1] // 2)] += 1
        if y > DETECT_PPM:
            detected = True
            if y > anchor_y:                                        # anchor = strongest detection so far
                anchor_xy, anchor_y = xm[:2].copy(), y

        q = srm.Q_.ravel()
        qm = q_metrics(q, Q_TRUE)                                   # three readouts of Q, see QMetrics.py
        log["pos"].append(xm.copy()); log["y_clean"].append(y_clean); log["y_raw"].append(y_raw); log["y"].append(y)
        log["q_sum"].append(qm["global"]["Q_hat"]); log["q_argmax"].append(qm["argmax"]["Q_hat"])
        log["q_neigh"].append(qm["neighborhood"]["Q_hat"]); log["loc"].append(cand[qm["argmax_idx"]])
        log["q_neigh_rel_err"].append(qm["neighborhood"]["rel_error"])
        # 4. TODO: Path planning: choose the next position (Alex's part, you can refer to "SrmEstimator.py")
        if policy == "svd" and detected and k + 1 < n_steps:
            xm, _ = choose_next_position(
                measurement_rows[: k + 1],
                xm,
                MOVES,
                predict_row=lambda point: srm.getCachedSrm(point, WIND),
                bounds=((X_MIN, Y_MIN), (X_MAX, Y_MAX)),
                penalty=lambda point: boundary_and_visit(point, visit_map),
            )
        elif policy.startswith("emgr") and detected:
            best, best_xp, best_W = np.inf, None, None
            for mv in MOVES:
                xp = np.clip(xm + mv, 0, 198)
                if np.allclose(xp, xm):
                    continue
                info = emgr.step(xp, WIND, update_W=False)          # EMGR score if we were at xp
                if not np.isfinite(info["score"]):
                    continue
                score = LAMBDA_EMGR * info["score"] + boundary_and_visit(xp, visit_map)
                if policy == "emgr_anchor":                         # pull toward the strongest detection
                    score += LAMBDA_ANCHOR * np.linalg.norm(xp[:2] - anchor_xy)
                if score < best:
                    best, best_xp, best_W = score, xp, info["W"]
            xm = best_xp if best_xp is not None else xm
            if best_W is not None:
                emgr.update_W(best_W)
        else:
            if k + 1 < len(path):
                xm = path[k + 1].copy()
                emgr.step(xm, WIND, update_W=True)                 # keep W in step with the path
    out = {key: np.array(v) for key, v in log.items()}
    out["Q_hat"] = srm.Q_.ravel().copy()                            # final estimate and cells, for print_q_report
    out["grid_cells"] = srm.grid_cells_
    return cand, out


def report(name, log, steps=(300, 600, 900, 1200, 1600, N_STEPS)):
    print(f"{name}")
    for s in steps:
        loc = log["loc"][s - 1]
        print(f"   step {s:5d}:Q_true: {Q_TRUE} | 3x3 neighbor Q_hat: {log['q_neigh'][s - 1]:7.4f} g/s | "
              f"Q_rel_error: {log["q_neigh_rel_err"][s - 1]:7.2%} | location error "
              f"{np.linalg.norm(loc - np.array(SOURCE_XY)):6.1f} m")


if __name__ == "__main__":
    #Choose different policy to compare.
    policy_list = [
        "lawnmower",
        # "lawnmower_rev",
        # "emgr",
        "emgr_anchor",
        "svd",
    ]
    for policy in policy_list:
        cand, log = run(policy)
        report(policy, log)
        print_q_report(log["Q_hat"], Q_TRUE, log["grid_cells"], SOURCE_XY)

    # Skylor's test bench: same flight, with and without the denoiser (policy "lawnmower": the example PCA is trained on it)
    print("denoiser test, policy 'lawnmower':")
    train = make_training_flights()
    for name, denoiser in (("none", IdentityDenoiser()), ("PCA example", WindowPCADenoiser(train))):
        cand, log = run("lawnmower", denoiser=denoiser)
        rmse = np.sqrt(np.mean((log["y"] - log["y_clean"]) ** 2))
        loc = log["loc"][-1]
        print(f"   {name:12s}: RMSE vs clean {rmse:.4f} ppm | final sum(Q_hat) {log['q_sum'][-1]:.4f} g/s | "
              f"location error {np.linalg.norm(loc - np.array(SOURCE_XY)):.1f} m")