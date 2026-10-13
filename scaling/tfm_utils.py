import numpy as np
import pandas as pd

FLOP_VALUES = np.array([
    1.00000000e+11, 1.77827941e+11, 3.16227766e+11, 5.62341325e+11,
    1.00000000e+12, 1.77827941e+12, 3.16227766e+12, 5.62341325e+12,
    1.00000000e+13, 1.77827941e+13, 3.16227766e+13, 5.62341325e+13,
    1.00000000e+14, 1.77827941e+14, 3.16227766e+14, 5.62341325e+14,
    1.00000000e+15, 1.77827941e+15, 3.16227766e+15, 5.62341325e+15,
    1.00000000e+16, 1.77827941e+16, 3.16227766e+16, 5.62341325e+16,
    1.00000000e+17, 1.77827941e+17, 3.16227766e+17, 5.62341325e+17,
    1.00000000e+18, 1.77827941e+18, 3.16227766e+18, 5.62341325e+18,
    1.00000000e+19
])


def interpolate_run_to_grid(
    run: pd.DataFrame,
    value_columns: list[str],
    grid: np.ndarray = FLOP_VALUES,
    flops_column: str = "total_flops",
    snap_tol_dex: float = 0.01,
    max_gap_dex: float = 0.35,
) -> pd.DataFrame:
    """Put one run's metrics onto the FLOP grid instead of snapping to the nearest grid point.

    Snapping mis-attributes compute for large models: a single optimizer step can burn a large
    fraction of a grid interval, so the first evaluations of a big run land far past the milestone
    (up to +39 % with linear nearest-neighbour snapping) and several evaluations can collapse into
    the same bucket. For each grid point this does instead:

      1. if an evaluation lies within `snap_tol_dex` (0.01 dex ~ 2.3 %) of it, take that
         evaluation's values as they are;
      2. otherwise, if the grid point lies between two evaluations at most `max_gap_dex` apart,
         interpolate linearly in log10(FLOPs) between those two neighbours only;
      3. otherwise drop the grid point (no extrapolation, no bridging of missing evaluations).

    Returns one row per kept grid point with `total_flops_rounded` (the grid value), the
    interpolated `value_columns`, `grid_idx` (index into `grid`), `interpolated` (rule 2 applied)
    and `flops_deviation_dex` (log10 distance of the used evaluation for rule 1, 0 otherwise).
    """
    run = run.sort_values(flops_column).drop_duplicates(flops_column, keep="last")
    log_x = np.log10(run[flops_column].to_numpy())
    log_grid = np.log10(grid)
    values = run[value_columns].to_numpy(dtype=float)

    # rule 1: nearest evaluation to each grid point
    nearest = np.abs(log_grid[:, None] - log_x[None, :]).argmin(axis=1)
    snap = np.abs(log_grid - log_x[nearest]) <= snap_tol_dex

    # rule 2: bracketing evaluations (left = right - 1)
    right = np.searchsorted(log_x, log_grid, side="left")
    inside = (right > 0) & (right < len(log_x))
    # Avoid interpolating when evaluations are too far away (max_gap_dex)
    right_c = np.clip(right, 1, len(log_x) - 1)
    left_c = right_c - 1
    gap = log_x[right_c] - log_x[left_c]
    interp = inside & ~snap & (gap <= max_gap_dex)

    keep = snap | interp
    out = np.empty((keep.sum(), len(value_columns))) # all remaining grid values and the value metrics
    for row, i in enumerate(np.flatnonzero(keep)):
        if snap[i]:
            out[row] = values[nearest[i]]
        else:
            w = (log_grid[i] - log_x[left_c[i]]) / gap[i]
            out[row] = (1 - w) * values[left_c[i]] + w * values[right_c[i]]

    result = pd.DataFrame(out, columns=value_columns)
    result.insert(0, "total_flops_rounded", grid[keep])
    result.insert(0, "grid_idx", np.flatnonzero(keep))
    result["interpolated"] = ~snap[keep]
    result["flops_deviation_dex"] = np.where(snap[keep], log_x[nearest[keep]] - log_grid[keep], 0.0)
    return result


def interpolate_all_runs(df, value_columns, constant_columns, run_level=0, **kwargs):
    pieces = []
    for run_id, run in df.groupby(level=run_level):
        valid = run.dropna(subset=value_columns)
        if len(valid) < 2:
            continue
        piece = interpolate_run_to_grid(valid, value_columns, **kwargs)
        for col in constant_columns:
            piece[col] = run[col].iloc[0]
        piece.insert(0, "run", run_id)
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)