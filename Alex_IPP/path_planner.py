"""Score candidate sampling positions using the H matrix."""

import numpy as np

def score_next_move(H, candidate_rows):
    """Return (numerical rank, weakest observed singular value) per trial."""
    spectra = [np.linalg.svd(np.vstack((H, row)), compute_uv=False)
        for row in candidate_rows]
    if not spectra:
        return []

    # Apply the same numerical rank threshold to every candidate.
    largest = max(values[0] for values in spectra)
    tolerance = np.finfo(spectra[0].dtype).eps * max(H.shape[0] + 1, H.shape[1]) * largest
    scores = []
    for values in spectra:
        rank = int(np.count_nonzero(values > tolerance))
        weakest = float(values[rank - 1]) if rank else 0.0
        scores.append((rank, weakest))
    return scores

def choose_next_position(H, position, moves, predict_row, bounds, penalty):
    """Return the chosen position and (position, score) pairs for valid moves."""
    lower, upper = bounds
    candidates = []
    for move in moves:
        point = position + move
        if np.any(point[:2] < lower) or np.any(point[:2] > upper):
            continue
        if np.array_equal(point, position):
            continue
        candidates.append(point)

    scores = score_next_move(H, [predict_row(point) for point in candidates])
    if not scores:
        return position, []

    # Rank comes first. For nearly equal singular values, use the existing boundary and revisit penalty to break the tie.
    best_rank, best_weakest = max(scores)
    finalists = [
        point
        for point, (rank, weakest) in zip(candidates, scores)
        if rank == best_rank and np.isclose(weakest, best_weakest, rtol=1e-8, atol=0.0)]
    return min(finalists, key=penalty), list(zip(candidates, scores))
