# -*- coding: utf-8 -*-
import io
import math
import itertools
from typing import Literal
from pathlib import Path

import numpy as np
import pandas as pd



# =========================================================
# ChainLadder Bootstrap (cumulative input expected)
# =========================================================
def ChainLadderBootstrap(
    df: pd.DataFrame,
    Report: bool = True,
    Mode: str = "Med",
    B: int = 200,
    report_name: str | None = None
) -> pd.DataFrame:
    """
    Bootstrap around Chain Ladder using a positional-factor reference.
    Returns a *cumulative* completed triangle.

    Parameters
    ----------
    df : cumulative triangle (AY x Dev)
    Report : write an Excel report into SAVE_DIR if True
    Mode : one of {"Med","Mean","P50","P75","P95"} (case-insensitive)
    B : number of bootstrap simulations
    report_name : filename (without extension) for the Excel report
    """
    # -----------------------------
    # Helpers (vectorized; no pandas cell loops)
    # -----------------------------
    def _coerce_numeric_triangle(df_in: pd.DataFrame) -> pd.DataFrame:
        out = df_in.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        cols_sorted = sorted(out.columns, key=lambda x: float(pd.to_numeric(x, errors="coerce")))
        out = out[cols_sorted]
        return out

    def _to_incremental(cum_df: pd.DataFrame) -> pd.DataFrame:
        inc = cum_df.copy()
        vals_r = cum_df.iloc[:, 1:].to_numpy(dtype=float, copy=False)
        vals_l = cum_df.iloc[:, :-1].to_numpy(dtype=float, copy=False)
        inc.iloc[:, 1:] = vals_r - vals_l
        return inc

    def _to_cumulative_preserve_nans(inc_df: pd.DataFrame) -> pd.DataFrame:
        arr = inc_df.to_numpy(dtype=float)
        I, J = arr.shape
        out = np.empty_like(arr, dtype=float)
        for i in range(I):
            running = 0.0
            for j in range(J):
                x = arr[i, j]
                if np.isnan(x):
                    out[i, j] = np.nan
                else:
                    running += x
                    out[i, j] = running
        return pd.DataFrame(out, index=inc_df.index, columns=inc_df.columns)

    def _observed_incremental_mask_from_cumulative(cum_df: pd.DataFrame) -> pd.DataFrame:
        has = cum_df.notna()
        mask = has.copy()
        if cum_df.shape[1] >= 2:
            mask.iloc[:, 1:] = has.iloc[:, 1:].to_numpy() & has.iloc[:, :-1].to_numpy()
        return mask

    def _chainladder_factors_positional_ref(cum_df: pd.DataFrame) -> np.ndarray:
        m = cum_df.shape[1]
        f = np.ones(m - 1, dtype=float)
        for j in range(m - 1):
            cj  = cum_df.iloc[:, j]
            cj1 = cum_df.iloc[:, j + 1]
            mask = cj.notna() & cj1.notna() & (cj > 0)
            den = cj[mask].sum()
            num = cj1[mask].sum()
            fj = float(num / den) if den > 0 else 1.0
            if (not np.isfinite(fj)) or (fj < 1.0):
                fj = 1.0
            f[j] = fj
        return f

    def _complete_with_frozen_f(cum_partial: pd.DataFrame, f_pos: np.ndarray) -> pd.DataFrame:
        # Force a fresh, writable ndarray
        out = np.array(cum_partial.to_numpy(dtype=float, copy=True), dtype=float, copy=True, order="C")
        try:
            out.setflags(write=True)  # be explicit
        except Exception:
            pass

        I, J = out.shape
        for i in range(I):
            L = -1
            for j in range(J):
                if np.isfinite(out[i, j]):
                    L = j
            if L < 0:
                continue
            for j in range(L, J - 1):
                prev = out[i, j]
                if not np.isfinite(prev):
                    break
                out[i, j + 1] = prev * f_pos[j]
        return pd.DataFrame(out, index=cum_partial.index, columns=cum_partial.columns)

    def _backtrack_expected_cumulative_from_last_observed(cum_df: pd.DataFrame, f_pos: np.ndarray) -> pd.DataFrame:
        C = cum_df.to_numpy(dtype=float)
        I, J = C.shape
        out = np.full_like(C, np.nan, dtype=float)
        for i in range(I):
            L = -1
            for j in range(J):
                if np.isfinite(C[i, j]):
                    L = j
            if L < 0:
                continue
            out[i, L] = C[i, L]
            for k in range(L - 1, -1, -1):
                fk = f_pos[k]
                if (not np.isfinite(fk)) or (fk <= 0.0):
                    fk = 1.0
                out[i, k] = out[i, k + 1] / fk
        return pd.DataFrame(out, index=cum_df.index, columns=cum_df.columns)

    def _pearson_residuals_poisson(X_inc: pd.DataFrame, mu_inc: pd.DataFrame, mask: pd.DataFrame, eps=1e-12) -> pd.DataFrame:
        mu = mu_inc.clip(lower=eps)
        r = (X_inc - mu) / np.sqrt(mu)
        return r.where(mask & X_inc.notna())

    # -----------------------------------
    # Normalize input; build base objects
    # -----------------------------------
    C_input = _coerce_numeric_triangle(df)
    X_inc   = _to_incremental(C_input)
    mask_obs_inc = _observed_incremental_mask_from_cumulative(C_input)

    f_pos = _chainladder_factors_positional_ref(C_input)
    C_back = _backtrack_expected_cumulative_from_last_observed(C_input, f_pos)
    mu_back = _to_incremental(C_back)
    R = _pearson_residuals_poisson(X_inc, mu_back, mask_obs_inc)

    rng = np.random.default_rng(42)
    I, J = C_input.shape
    cum_samples = np.empty((int(B), I, J), dtype=np.float64)

    mu_arr   = mu_back.to_numpy(dtype=float, copy=False)
    mask_arr = mask_obs_inc.to_numpy(dtype=bool, copy=False)
    R_arr    = R.to_numpy(dtype=float, copy=False)
    pool = R_arr[np.isfinite(R_arr)]
    sqrt_mu = np.sqrt(np.clip(mu_arr, 1e-12, np.inf))
    obs_idx = np.argwhere(mask_arr)

    def _one_iteration() -> np.ndarray:
        if pool.size:
            draws = rng.choice(pool, size=obs_idx.shape[0], replace=True)
        else:
            draws = np.zeros(obs_idx.shape[0], dtype=float)
        Xstar = np.full_like(mu_arr, np.nan, dtype=float)
        base_vals = mu_arr[obs_idx[:, 0], obs_idx[:, 1]]
        root_vals = sqrt_mu[obs_idx[:, 0], obs_idx[:, 1]]
        Xvals = base_vals + draws * root_vals
        Xvals = np.maximum(Xvals, 1e-8)
        Xstar[obs_idx[:, 0], obs_idx[:, 1]] = Xvals
        Cstar_obs = _to_cumulative_preserve_nans(pd.DataFrame(Xstar, index=C_input.index, columns=C_input.columns))
        Cstar_full = _complete_with_frozen_f(Cstar_obs, f_pos)
        return Cstar_full.to_numpy(dtype=float, copy=False)

    sample_list = []
    for b in range(int(B)):
        Cb = _one_iteration()
        cum_samples[b, :, :] = Cb
        if b < 5:
            sample_list.append(pd.DataFrame(Cb, index=C_input.index, columns=C_input.columns))

    mean_cum = cum_samples.mean(axis=0)
    p50_cum  = np.percentile(cum_samples, 50, axis=0)
    p75_cum  = np.percentile(cum_samples, 75, axis=0)
    p95_cum  = np.percentile(cum_samples, 95, axis=0)

    def _mk(arr):
        return pd.DataFrame(arr, index=C_input.index, columns=C_input.columns)

    # -----------------------------
    # Optional Excel Report (SAVE_DIR)
    # -----------------------------
    report_bytes = None
    if Report:
        buffer = io.BytesIO()

        # Robust writer (xlsxwriter preferred, openpyxl fallback)
        try:
            writer = pd.ExcelWriter(
                buffer,
                engine="xlsxwriter",
                engine_kwargs={"options": {"strings_to_numbers": True}}
            )
        except TypeError:  # older pandas
            try:
                writer = pd.ExcelWriter(
                    buffer,
                    engine="xlsxwriter",
                    options={"strings_to_numbers": True}
                )
            except Exception:
                writer = pd.ExcelWriter(buffer, engine="openpyxl")

        with writer as xl:
            # -----------------------------
            # INPUTS & PARAMETERS
            # -----------------------------
            C_input.to_excel(xl, sheet_name="Input_Cumulative")
            pd.Series(
                f_pos,
                index=C_input.columns[:-1],
                name="f_j"
            ).to_excel(xl, sheet_name="Frozen_Factors")

            # -----------------------------
            # BACKTRACKED EXPECTATIONS
            # -----------------------------
            C_back.to_excel(xl, sheet_name="E_Cum_Backtracked")
            mu_back.to_excel(xl, sheet_name="E_Inc_Backtracked")

            # -----------------------------
            # RESIDUALS
            # -----------------------------
            R.to_excel(xl, sheet_name="Residuals")

            # -----------------------------
            # BOOTSTRAP ITERATIONS
            # -----------------------------
            for k, Cstar in enumerate(sample_list):
                Cstar.to_excel(
                    xl,
                    sheet_name=f"Iter{k+1}_CompletedCum"
                )

            # -----------------------------
            # AGGREGATED RESULTS
            # -----------------------------
            _mk(mean_cum).to_excel(xl, sheet_name="Mean_Cumulative")
            _mk(p50_cum).to_excel(xl, sheet_name="P50_Cumulative")
            _mk(p75_cum).to_excel(xl, sheet_name="P75_Cumulative")
            _mk(p95_cum).to_excel(xl, sheet_name="P95_Cumulative")

        buffer.seek(0)
        report_bytes = buffer.getvalue()

    # -----------------------------
    # Return selected triangle
    # -----------------------------
    m = (Mode or "").strip().upper()

    if m in {"MEAN", "MED"}:
        return _mk(mean_cum), report_bytes
    if m in {"P50", "50"}:
        return _mk(p50_cum), report_bytes
    if m in {"P75", "75"}:
        return _mk(p75_cum), report_bytes
    if m in {"P95", "95"}:
        return _mk(p95_cum), report_bytes

    return _mk(p50_cum), report_bytes


# =========================================================
# Last-3
# =========================================================
def last3(
    df: pd.DataFrame,
    default_factor: float = 1.0,
    enforce_monotonic: bool = True,
) -> pd.DataFrame:
    if df.shape[1] < 2:
        return df.copy()

    out = df.copy()
    age_cols = list(out.columns)
    cum = out.loc[:, age_cols].apply(pd.to_numeric, errors="coerce")

    ratio_cols = age_cols[:-1]
    D = pd.DataFrame(index=out.index, columns=ratio_cols, dtype="float64")
    for j, col_from in enumerate(ratio_cols):
        col_to = age_cols[j + 1]
        num = cum[col_to]
        den = cum[col_from]
        with np.errstate(invalid="ignore", divide="ignore"):
            D[col_from] = np.where((den.notna()) & (num.notna()) & (den != 0.0), num / den, np.nan)

    factors: dict = {}
    for col in D.columns:
        s = pd.to_numeric(D[col], errors="coerce").dropna()
        if len(s) > 0:
            f = float(s.tail(3).mean())
            if not np.isfinite(f) or f <= 0:
                f = default_factor
        else:
            f = default_factor
        factors[col] = f
    factors.setdefault(age_cols[-1], default_factor)

    completed_cum = cum.copy()
    for i in range(len(completed_cum)):
        row_vals = completed_cum.iloc[i, :].copy()
        last_idx = row_vals.last_valid_index()
        if last_idx is None:
            continue
        last_pos = age_cols.index(last_idx)
        val = row_vals.iloc[last_pos]
        for j in range(last_pos, len(age_cols) - 1):
            from_col = age_cols[j]
            f = factors.get(from_col, default_factor)
            val = val * f if pd.notna(val) else np.nan
            completed_cum.iat[i, j + 1] = val

        if enforce_monotonic:
            rv = completed_cum.iloc[i, :].values.astype(float)
            for k in range(1, len(rv)):
                if np.isfinite(rv[k - 1]) and np.isfinite(rv[k]) and rv[k] < rv[k - 1]:
                    rv[k] = rv[k - 1]
            completed_cum.iloc[i, :] = rv

    return completed_cum


# =========================================================
# Dernier facteur (average/last)
# =========================================================
def dernier_fact(
    df: pd.DataFrame,
    method: Literal["average", "last"] = "last",
    tail_factor: float | None = None,
    enforce_monotonic: bool = True,
) -> pd.DataFrame:
    ages_map = {}
    for c in df.columns:
        try:
            ages_map[c] = int(str(c).strip())
        except Exception:
            pass
    if not ages_map:
        raise ValueError("No integer-like development-age columns detected in df.")
    cols_old = [c for c, _ in sorted(ages_map.items(), key=lambda kv: kv[1])]
    ages = [ages_map[c] for c in cols_old]

    tri_raw = df.loc[:, cols_old].copy()
    tri_raw.columns = ages
    tri_raw = tri_raw.sort_index()

    def _is_cumulative(row: pd.Series) -> bool:
        vals = row.values.astype(float)
        ok = np.isfinite(vals)
        vals = vals[ok]
        if vals.size <= 1:
            return True
        return np.all(np.diff(vals) >= -1e-12)

    is_cum = tri_raw.apply(_is_cumulative, axis=1).all()
    tri_cum = tri_raw.cumsum(axis=1) if not is_cum else tri_raw.copy()

    obs = tri_cum.notna()
    if len(ages) < 2:
        return tri_cum

    factors = {}
    for j_idx in range(len(ages) - 1):
        j, j1 = ages[j_idx], ages[j_idx + 1]
        mask_rows = obs[j] & obs[j1]
        c_j = tri_cum.loc[mask_rows, j]
        c_j1 = tri_cum.loc[mask_rows, j1]
        valid = (c_j > 0) & c_j1.notna()
        c_j = c_j[valid]
        c_j1 = c_j1.loc[c_j.index]
        if c_j.empty:
            f = 1.0
        else:
            if method == "average":
                denom = c_j.sum()
                numer = c_j1.sum()
                f = (numer / denom) if denom > 0 else 1.0
            elif method == "last":
                last_idx = c_j.index[-1]
                denom = tri_cum.at[last_idx, j]
                numer = tri_cum.at[last_idx, j1]
                f = (numer / denom) if (pd.notna(denom) and denom not in (0.0,)) else 1.0
            else:
                raise ValueError("method must be 'average' or 'last'.")
        if not np.isfinite(f) or f <= 0:
            f = 1.0
        factors[j] = f

    if tail_factor is not None:
        tail_factor = float(tail_factor)
        if not np.isfinite(tail_factor) or tail_factor <= 0:
            raise ValueError("tail_factor must be positive and finite.")

    tri_completed = tri_cum.copy()
    last_age = ages[-1]
    for r in tri_completed.index:
        row_obs = obs.loc[r, ages]
        if not row_obs.any():
            continue
        last_obs_pos = np.where(row_obs.values)[0].max()
        last_obs_age = ages[last_obs_pos]
        running = tri_completed.at[r, last_obs_age]
        if pd.isna(running):
            continue
        for j_idx in range(last_obs_pos, len(ages) - 1):
            j = ages[j_idx]
            running = running * factors.get(j, 1.0)
            tri_completed.at[r, ages[j_idx + 1]] = running
        if tail_factor is not None:
            tri_completed.at[r, last_age] = tri_completed.at[r, last_age] * tail_factor

        if enforce_monotonic:
            row_vals = tri_completed.loc[r, ages].values.astype(float)
            for k in range(1, len(row_vals)):
                if np.isfinite(row_vals[k - 1]) and np.isfinite(row_vals[k]) and row_vals[k] < row_vals[k - 1]:
                    row_vals[k] = row_vals[k - 1]
            tri_completed.loc[r, ages] = row_vals

    return tri_completed


# =========================================================
# Poisson / ODP Bootstrap (incremental input expected)
# =========================================================
from pathlib import Path


# --- put these near the top of your methods file ---
from pathlib import Path
import re
import numpy as np
import pandas as pd



def _sanitize_name(name: str, default: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", (name or "").strip())
    cleaned = cleaned.rstrip(" .")
    return cleaned or default

def _looks_cumulative_row(row: pd.Series) -> bool:
    v = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(v)
    v = v[m]
    if v.size <= 1:
        return True
    return np.all(np.diff(v) >= -1e-12)

def _triangle_to_incremental(df: pd.DataFrame) -> pd.DataFrame:
    """Convert to incremental if the triangle *looks* cumulative (majority monotone rows)."""
    is_cum = bool(df.apply(_looks_cumulative_row, axis=1).mean() >= 0.5)
    if not is_cum:
        return df.copy()
    cum = df.apply(pd.to_numeric, errors="coerce")
    inc = cum.copy()
    if inc.shape[1] >= 2:
        inc.iloc[:, 1:] = cum.iloc[:, 1:].to_numpy(dtype=float) - cum.iloc[:, :-1].to_numpy(dtype=float)
        na = cum.isna()
        inc = inc.mask(na | na.shift(axis=1, fill_value=True))
    return inc
# --- end tiny helpers ---


# =========================================================
# Poisson / ODP Bootstrap — FIXED (no missing row/col)
# =========================================================
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings

import streamlit as st




# =========================================================
# Poisson / ODP Bootstrap — FINAL FIXED VERSION
# =========================================================
from pathlib import Path
import re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings



def Poissonbootstrap(
    df: pd.DataFrame,
    Report: bool = True,
    Mode: str = "Med",
    B: int = 200,
    seed: int | None = None,
    report_name: str | None = None,
    zero_future_tol: float = 1e-6,
    lam_cap: float | None = None,
    debug: bool = False,
) -> pd.DataFrame:
    """
    ODP / Poisson bootstrap operating on *incremental* triangles.
    Automatically converts cumulative triangles when needed.
    Returns a completed *cumulative* triangle (mean or percentile).

    Parameters
    ----------
    df : pd.DataFrame
        Incremental or cumulative triangle (AY x Dev)
    Report : bool
        Write Excel report if True
    Mode : {"Med","Mean","P50","P75","P99"}
        Output mode (percentile)
    B : int
        Number of bootstrap simulations
    seed : int | None
        RNG seed
    report_name : str | None
        Custom Excel filename (without extension)
    zero_future_tol : float
        Tolerance to detect observed/future boundary
    lam_cap : float | None
        Cap for Poisson mean
    debug : bool
        Verbose output for first few iterations
    """
    # -----------------------------
    # Constants
    # -----------------------------
    MU_FLOOR = 1e-6
    RIDGE_ALPHA = 1e-8
    MAXITER = 400
  
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # -----------------------------
    # Helpers
    # -----------------------------
    def _sanitize_name(name: str, default: str) -> str:
        cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", (name or "").strip())
        cleaned = cleaned.rstrip(" .")
        return cleaned or default

    def _coerce_numeric(df_in: pd.DataFrame) -> pd.DataFrame:
        """Coerce columns to numeric, preserving original index and columns."""
        out = df_in.copy()
        # Convert only the data, not index/columns
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # --- cumulative detector (safe)
    def _looks_cumulative_row(row: pd.Series) -> bool:
        v = pd.to_numeric(row, errors="coerce").to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size <= 1:
            return False
        return np.all(np.diff(v) >= -1e-12)

    # --- robust incremental converter
    def _force_incremental_once(df_in: pd.DataFrame) -> pd.DataFrame:
        """Convert to incremental only if truly cumulative (avoid false positives)."""
        arr = df_in.to_numpy(dtype=float)
        first_col = arr[:, 0]

        # If first dev col is basically zeros, assume already incremental
        if np.nanmean(np.abs(first_col)) < 1e-9:
            return df_in.copy()

        is_monotone = []
        for row in arr:
            v = np.array([x for x in row if np.isfinite(x)], dtype=float)
            if v.size <= 1:
                is_monotone.append(False)
                continue
            is_monotone.append(np.all(np.diff(v) >= -1e-12))
        
        # Require strong monotonicity
        if np.mean(is_monotone) >= 0.7:
            diff = arr[:, 1:] - arr[:, :-1]
            bad = ~np.isfinite(arr[:, 1:]) | ~np.isfinite(arr[:, :-1])
            inc = arr.copy()
            inc[:, 1:] = np.where(bad, np.nan, diff)
            return pd.DataFrame(inc, index=df_in.index, columns=df_in.columns)
        return df_in.copy()

    # --- safe frontier mask (FIXED)
    def _mask_frontier(inc: pd.DataFrame, tol: float):
        n, m = inc.shape
        obs = np.zeros((n, m), dtype=bool)
        fut = np.zeros((n, m), dtype=bool)
        arr = inc.to_numpy(dtype=float)
        
        for i in range(n):
            row = arr[i, :]
            finite = np.isfinite(row)
            if not finite.any():
                # If no finite values, mark all as observed (they're NaN/empty)
                obs[i, :] = True
                continue
            
            # Find the last finite value in the row
            finite_indices = np.where(finite)[0]
            if len(finite_indices) == 0:
                obs[i, :] = True  # All are NaN
                continue
            
            last_finite_idx = finite_indices[-1]
            
            # Mark everything up to and including the last finite as observed
            obs[i, :last_finite_idx + 1] = True
            
            # Mark everything after the last finite as future
            if last_finite_idx + 1 < m:
                fut[i, last_finite_idx + 1:] = True
        
        # Special handling for last accident year
        # In typical triangles, the last AY has only the first dev period observed
        # So we should ensure at least the first column is marked as observed
        last_row_idx = n - 1
        last_row_finite = np.isfinite(arr[last_row_idx, :])
        if last_row_finite.any():
            # Find last finite value in last row
            last_finite_idx = np.where(last_row_finite)[0][-1]
            obs[last_row_idx, :last_finite_idx + 1] = True
            if last_finite_idx + 1 < m:
                fut[last_row_idx, last_finite_idx + 1:] = True
        else:
            # If no finite values in last row, mark first column as observed (typical triangle)
            obs[last_row_idx, 0] = True
        
        return obs, fut

    # --- Pearson residuals
    def _pearson_resid(y, mu, p):
        mu = np.clip(mu, MU_FLOOR, np.inf)
        r = (y - mu) / np.sqrt(mu)
        n = len(r)
        df_scale = np.sqrt(n / max(n - p, 1))
        r_bc = df_scale * r
        phi = float(np.mean(r_bc**2))
        return r_bc, phi

    # --- GLM fit
    def _fit_poisson(y, X):
        model = sm.GLM(y, X, family=sm.families.Poisson())
        try:
            res = model.fit(maxiter=MAXITER)
            mu = np.clip(res.predict(X), MU_FLOOR, np.inf)
            if not np.isfinite(mu).all():
                raise ValueError
            return res
        except Exception:
            return model.fit_regularized(alpha=RIDGE_ALPHA, L1_wt=0.0, maxiter=MAXITER)

    # -----------------------------
    # Prepare triangle
    # -----------------------------
    tri0 = _coerce_numeric(df)
    tri_inc = _force_incremental_once(tri0)
    idx_ay, cols_dev = tri_inc.index, tri_inc.columns
    n, m = tri_inc.shape

    # Masks (FIXED: using the updated _mask_frontier)
    obs_mask, fut_mask = _mask_frontier(tri_inc, zero_future_tol)
    if not obs_mask.any():
        raise ValueError("No observed cells detected.")
    
    # Debug: print mask info
    if debug:
        print(f"Triangle shape: {n}x{m}")
        print(f"Observed cells: {obs_mask.sum()}")
        print(f"Future cells: {fut_mask.sum()}")
        print(f"Last row obs mask: {obs_mask[-1, :]}")
        print(f"Last row fut mask: {fut_mask[-1, :]}")

    # Observed values
    inc_vals = tri_inc.to_numpy(dtype=float)
    
    # Get observed values for GLM fitting
    obs_positions = np.where(obs_mask)
    y_obs = inc_vals[obs_positions]
    
    # Remove NaN values from observed (in case some observed cells are NaN)
    valid_mask = np.isfinite(y_obs)
    y_obs = y_obs[valid_mask]
    obs_positions = (obs_positions[0][valid_mask], obs_positions[1][valid_mask])
    
    if len(y_obs) == 0:
        raise ValueError("No valid numeric values in observed cells.")

    # Prepare design matrices
    AY_obs = pd.Categorical(idx_ay[obs_positions[0]], categories=idx_ay, ordered=True)
    DV_obs = pd.Categorical(cols_dev[obs_positions[1]], categories=cols_dev, ordered=True)

    X_obs = pd.concat(
        [pd.get_dummies(AY_obs, prefix="AY", drop_first=True),
         pd.get_dummies(DV_obs, prefix="Dev", drop_first=True)], axis=1)
    X_obs = sm.add_constant(X_obs, has_constant="add").astype(float)
    X_cols = X_obs.columns
    X_obs_np = X_obs.to_numpy(dtype=float)

    # Prepare full design matrix
    AY_full = np.repeat(idx_ay, m)
    DV_full = np.tile(cols_dev, n)
    X_full = pd.concat(
        [pd.get_dummies(pd.Categorical(AY_full, categories=idx_ay, ordered=True),
                        prefix="AY", drop_first=True),
         pd.get_dummies(pd.Categorical(DV_full, categories=cols_dev, ordered=True),
                        prefix="Dev", drop_first=True)], axis=1)
    X_full = sm.add_constant(X_full, has_constant="add").astype(float)
    X_full = X_full.reindex(columns=X_cols, fill_value=0.0)
    X_full_np = X_full.to_numpy(dtype=float)

    # -----------------------------
    # Base GLM fit
    # -----------------------------
    res_base = _fit_poisson(y_obs, X_obs_np)
    mu_obs = np.clip(res_base.predict(X_obs_np), MU_FLOOR, np.inf)
    r_pool, phi_base = _pearson_resid(y_obs, mu_obs, len(res_base.params))

    rng = np.random.default_rng(seed)
    
    # Initialize arrays for simulation
    inc_obs_vals = np.zeros_like(inc_vals, dtype=float)
    inc_obs_vals[obs_mask] = inc_vals[obs_mask]  # Use original observed values
    inc_obs_vals = np.nan_to_num(inc_obs_vals, nan=0.0)
    
    fut_mask_arr = fut_mask.astype(bool)
    lam_cap_val = 1e6 if lam_cap is None else float(lam_cap)
    eps = MU_FLOOR

    cum_samples = np.full((B, n, m), np.nan, dtype=float)
    sample_list = []

    # CRITICAL FIX: Don't blindly set first column to 0
    # Instead, preserve the observed values in first column
    # Only use zeros for cells that are truly unobserved
    first_col_mask = obs_mask[:, 0]
    first_col_vals = np.where(first_col_mask, inc_vals[:, 0], 0.0)
    first_col_vals = np.nan_to_num(first_col_vals, nan=0.0)

    # -----------------------------
    # Bootstrap loop
    # -----------------------------
    for b in range(B):
        r_star = rng.choice(r_pool, size=r_pool.shape[0], replace=True)
        y_star = np.clip(mu_obs + r_star * np.sqrt(mu_obs), 0.0, np.inf)
        res_star = _fit_poisson(y_star, X_obs_np)

        mu_full = np.clip(res_star.predict(X_full_np), MU_FLOOR, np.inf).reshape(n, m)
        mu_obs_star = np.clip(res_star.predict(X_obs_np), MU_FLOOR, np.inf)
        _, phi_star = _pearson_resid(y_star, mu_obs_star, len(res_star.params))
        phi_eff = max(float(phi_star), 1e-6)

        mu_future = np.where(fut_mask_arr, mu_full, np.nan)
        mask_fut = np.isfinite(mu_future)
        inc_repl = inc_obs_vals.copy()
        
        if mask_fut.any():
            lam = np.minimum(np.maximum(mu_future[mask_fut], eps) / phi_eff, lam_cap_val)
            # Simulate future increments
            simulated = phi_eff * rng.poisson(lam)
            inc_repl[mask_fut] = simulated
        
        # Preserve observed values (including first column)
        # This is important: we should never overwrite observed values
        inc_repl[obs_mask] = inc_obs_vals[obs_mask]
        
        # Special handling: if first column has zeros in original data, keep them
        # Don't force zeros if they're not in the original data
        for i in range(n):
            if not first_col_mask[i]:
                # If first column is not observed, it might be 0 or simulated
                # Check if it was simulated in fut_mask
                if fut_mask[i, 0]:
                    # It was simulated, keep the simulation
                    pass
                else:
                    # It should be 0 (before first observed dev period)
                    inc_repl[i, 0] = 0.0
        
        cum_repl = np.cumsum(inc_repl, axis=1)
        cum_samples[b, :, :] = cum_repl
        if b < 5:
            sample_list.append(pd.DataFrame(cum_repl, index=idx_ay, columns=cols_dev))
        if debug and b < 3:
            print(f"[DEBUG] iter={b}, phi={phi_eff:.6g}, simulated={mask_fut.sum()}")
            if b == 0:
                print(f"  First col values: {inc_repl[:, 0]}")
                print(f"  Last row values: {inc_repl[-1, :]}")

    # -----------------------------
    # Aggregate results
    # -----------------------------
    mean_cum = np.nanmean(cum_samples, axis=0)
    p50_cum = np.nanpercentile(cum_samples, 50, axis=0)
    p75_cum = np.nanpercentile(cum_samples, 75, axis=0)
    p99_cum = np.nanpercentile(cum_samples, 99, axis=0)

    mk = lambda arr: pd.DataFrame(arr, index=idx_ay, columns=cols_dev)

    # -----------------------------
    # Excel Report
    # -----------------------------
    report_bytes = None

    if Report:
        buffer = io.BytesIO()
        try:
            writer = pd.ExcelWriter(
                buffer,
                engine="xlsxwriter",
                engine_kwargs={"options": {"strings_to_numbers": True}}
            )
        except Exception:
            writer = pd.ExcelWriter(buffer, engine="openpyxl")

        with writer as xl:
            meta = pd.DataFrame({
                "B": [B], "seed": [seed], "phi_base": [phi_base],
                "zero_future_tol": [zero_future_tol], "lam_cap": [lam_cap],
                "mu_floor": [MU_FLOOR], "ridge_alpha": [RIDGE_ALPHA]
            })
            meta.to_excel(xl, sheet_name="Meta", index=False)

            tri_inc_full = pd.DataFrame(inc_vals, index=idx_ay, columns=cols_dev)
            tri_inc_full.to_excel(xl, sheet_name="Input_Incremental")

            # Also show masks for debugging
            obs_df = pd.DataFrame(obs_mask.astype(int), index=idx_ay, columns=cols_dev)
            fut_df = pd.DataFrame(fut_mask.astype(int), index=idx_ay, columns=cols_dev)
            obs_df.to_excel(xl, sheet_name="Observed_Mask")
            fut_df.to_excel(xl, sheet_name="Future_Mask")

            for i, C in enumerate(sample_list):
                C.to_excel(xl, sheet_name=f"Iter{i+1}_Cum")

            mk(mean_cum).to_excel(xl, sheet_name="Mean_Cumulative")
            mk(p50_cum).to_excel(xl, sheet_name="P50_Cumulative")
            mk(p75_cum).to_excel(xl, sheet_name="P75_Cumulative")
            mk(p99_cum).to_excel(xl, sheet_name="P99_Cumulative")

        buffer.seek(0)
        report_bytes = buffer.getvalue()
    
    # Return Selected Mode
    mode = (Mode or "").upper()

    if mode in {"MEAN", "MED"}:
        out = mk(mean_cum)
    elif mode in {"P50", "50"}:
        out = mk(p50_cum)
    elif mode in {"P75", "75"}:
        out = mk(p75_cum)
    elif mode in {"P99", "99"}:
        out = mk(p99_cum)
    else:
        out = mk(p50_cum)

    return out, report_bytes





# =========================================================
# Minimal Chain Ladder (auto type)
# =========================================================
def chainladder(df: pd.DataFrame) -> pd.DataFrame:
    def _coerce_numeric(df_: pd.DataFrame) -> pd.DataFrame:
        out = df_.copy()
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    def _incremental_to_cumulative(inc_: pd.DataFrame) -> pd.DataFrame:
        out = inc_.copy()
        all_nan_rows = out.isna().all(axis=1)
        out = out.fillna(0).cumsum(axis=1)
        out.loc[all_nan_rows] = np.nan
        return out

    def _detect_triangle_is_cumulative(tri_: pd.DataFrame) -> bool:
        if tri_.shape[1] <= 1:
            return True
        def nondec(s):
            v = s.dropna().values
            return np.all(np.diff(v) >= -1e-12) if len(v) else True
        return tri_.apply(nondec, axis=1).mean() >= 0.6

    raw = df.copy()
    df_num = _coerce_numeric(raw)
    age_nums = pd.to_numeric(df_num.columns, errors="coerce")
    age_cols = list(pd.Index(df_num.columns)[~age_nums.isna()])
    tri = df_num[age_cols]

    def _to_num(name):
        try:
            v = float(pd.to_numeric(name, errors="coerce"))
            return v
        except Exception:
            return np.inf
    tri = tri.reindex(sorted(tri.columns, key=_to_num), axis=1)

    new_cols = []
    for c in tri.columns:
        try:
            f = float(pd.to_numeric(c, errors="coerce"))
            new_cols.append(int(f) if f.is_integer() else f)
        except Exception:
            new_cols.append(c)
    tri.columns = new_cols

    is_cum = _detect_triangle_is_cumulative(tri)
    cum_in = _incremental_to_cumulative(tri) if not is_cum else tri.copy()

    cols = list(cum_in.columns)
    facs = []
    for j in range(len(cols) - 1):
        x = cum_in[cols[j]]
        y = cum_in[cols[j + 1]]
        mask = x.notna() & y.notna() & (x > 0)
        den = x[mask].sum()
        num = y[mask].sum()
        f = (num / den) if den > 0 else 1.0
        facs.append(max(float(f), 1.0))
    f = pd.Series(facs, index=cols[:-1], name="f_age_to_age")

    cum_completed = cum_in.copy()
    for i in range(cum_completed.shape[0]):
        row = cum_completed.iloc[i, :]
        obs = ~row.isna()
        if not obs.any():
            continue
        last_obs = np.where(obs.values)[0][-1]
        for j in range(last_obs, len(cols) - 1):
            prev = cum_completed.iat[i, j]
            if pd.isna(prev):
                break
            if pd.isna(cum_completed.iat[i, j + 1]):
                cum_completed.iat[i, j + 1] = prev * f.iloc[j]

    return cum_completed


# =========================================================
# Moyenne des facteurs
# =========================================================
def moyenne_facteurs(df: pd.DataFrame, eps_denom: float = 0.0) -> pd.DataFrame:
    keep = {}
    for c in df.columns:
        try:
            k = int(str(c).strip())
            keep[c] = k
        except Exception:
            pass
    if not keep:
        raise ValueError("Aucune colonne d'âge (entière) détectée dans `df`.")

    ages_sorted = sorted(keep.items(), key=lambda x: x[1])
    cols_old = [x[0] for x in ages_sorted]
    cols_new = [int(x[1]) for x in ages_sorted]

    cum = df.loc[:, cols_old].copy()
    cum.columns = cols_new
    cum = cum.sort_index(axis=1)
    cum = cum.apply(pd.to_numeric, errors="coerce")

    ages = list(cum.columns)
    if len(ages) < 2:
        return cum

    F = pd.DataFrame(index=cum.index, columns=ages[:-1], dtype=float)
    for j, j1 in zip(ages[:-1], ages[1:]):
        base = cum[j]
        nxt  = cum[j1]
        valid = base.notna() & nxt.notna() & (base > eps_denom)
        F[j] = np.where(valid, nxt / base, np.nan)

    mu = F.mean(axis=0, skipna=True)
    mu.index.name = "age"
    mu.name = "mu"

    completed = cum.copy()
    mu_aligned = mu.reindex(ages[:-1])
    mu_vals = mu_aligned.values.astype(float)
    mu_prod_from_start = np.cumprod(mu_vals)

    def product_mu_between(a0_idx: int, t_idx: int) -> float:
        if t_idx <= a0_idx:
            return 1.0
        num = mu_prod_from_start[t_idx - 1]
        den = mu_prod_from_start[a0_idx - 1] if a0_idx > 0 else 1.0
        return float(num / den)

    for e in completed.index:
        row = completed.loc[e]
        obs_mask = row.notna()
        if not obs_mask.any():
            continue
        a0_index = np.where(obs_mask.values)[0][-1]
        a0_age = ages[a0_index]
        base = completed.at[e, a0_age]
        for t_idx in range(a0_index + 1, len(ages)):
            t_age = ages[t_idx]
            mult = product_mu_between(a0_index, t_idx)
            pred = base * mult if np.isfinite(base) and np.isfinite(mult) else np.nan
            prev_val = completed.at[e, ages[t_idx - 1]]
            if np.isfinite(prev_val):
                if not np.isfinite(pred) or pred < prev_val:
                    pred = prev_val
            completed.at[e, t_age] = pred

    return completed


# =========================================================
# De Vylder (ALS)
# =========================================================
def vylder(
    df: pd.DataFrame,
    tol: float = 1e-8,
    max_iter: int = 5000,
    eps: float = 1e-9,
    ridge: float = 1e-12,
    enforce_monotone_p: bool = False,
    treat_zero_as_missing: bool = False,
    age_upper_bound: int = 200
) -> pd.DataFrame:
    keep = {}
    for c in df.columns:
        s = str(c).strip()
        try:
            v = int(float(s))
        except Exception:
            continue
        if 0 <= v <= age_upper_bound and v < 1900:
            keep[c] = v
    if not keep:
        raise ValueError("No valid integer-named development-age columns found (0..age_upper_bound).")

    cols_old = [k for k, _ in sorted(keep.items(), key=lambda x: x[1])]
    cols_new = [int(v) for _, v in sorted(keep.items(), key=lambda x: x[1])]

    tri = df.loc[:, cols_old].copy()
    tri.columns = cols_new
    tri = tri.sort_index(axis=1).apply(pd.to_numeric, errors="coerce")

    def _likely_cumulative(row: pd.Series) -> bool:
        x = row.values.astype(float)
        m = np.isfinite(x)
        if m.sum() < 2:
            return False
        x = x[m]
        return np.all(np.diff(x) >= -1e-12)

    is_cum_mask = tri.apply(_likely_cumulative, axis=1)
    looks_cumulative = bool(is_cum_mask.mean() >= 0.5)
    observed_mask = tri.notna().values

    if looks_cumulative:
        tri_inc = tri.copy()
        tri_inc.iloc[:, 1:] = tri_inc.iloc[:, 1:].values - tri_inc.iloc[:, :-1].values
        inc = tri_inc.values.astype(float)
        obs_cum = observed_mask.copy()
        obs_inc = np.zeros_like(obs_cum, dtype=bool)
        if inc.shape[1] > 0:
            obs_inc[:, 0] = obs_cum[:, 0]
        if inc.shape[1] > 1:
            obs_inc[:, 1:] = obs_cum[:, 1:] & obs_cum[:, :-1]
    else:
        inc = tri.values.astype(float)
        obs_inc = observed_mask.copy()

    if treat_zero_as_missing:
        obs_inc = obs_inc & (inc != 0.0)
        inc = np.where(obs_inc, inc, np.nan)

    mask = np.isfinite(inc)
    n_rows, n_cols = inc.shape

    col_sums = np.nansum(np.where(mask, inc, 0.0), axis=0)
    p = col_sums + eps
    s = p.sum()
    p = p / s if s > 0 else np.full(n_cols, 1.0 / max(n_cols, 1))
    U = np.zeros(n_rows, dtype=float)

    def _project_monotone_decreasing(vec: np.ndarray) -> np.ndarray:
        v = -vec.copy()
        n = len(v)
        y = v.copy()
        w = np.ones(n, dtype=float)
        i = 0
        while i < n - 1:
            if y[i] > y[i + 1] + 1e-18:
                j = i
                while j >= 0 and y[j] > y[j + 1] + 1e-18:
                    newv = (w[j] * y[j] + w[j + 1] * y[j + 1]) / (w[j] + w[j + 1])
                    y[j] = y[j + 1] = newv
                    w[j] = w[j] + w[j + 1]
                    k = j - 1
                    while k >= 0 and y[k] > y[k + 1] + 1e-18:
                        newv = (w[k] * y[k] + w[k + 1] * y[k + 1]) / (w[k] + w[k + 1])
                        y[k] = y[k + 1] = newv
                        w[k] = w[k] + w[k + 1]
                        k -= 1
                    j -= 1
            i += 1
        res = -y
        res = np.clip(res, 0.0, None)
        tot = res.sum()
        return res / tot if tot > 0 else np.full_like(res, 1.0 / len(res))

    prev_loss = np.inf
    for _ in range(int(max_iter)):
        denom_i = (mask * (p ** 2)).sum(axis=1) + ridge
        num_i   = np.where(mask, inc * p, 0.0).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            U = np.divide(num_i, denom_i, out=np.zeros_like(num_i), where=denom_i > 0)
        U = np.clip(U, 0.0, None)

        UU = U[:, None]
        denom_j = (mask * (UU ** 2)).sum(axis=0) + ridge
        num_j   = np.where(mask, inc * UU, 0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            p = np.divide(num_j, denom_j, out=p, where=denom_j > 0)

        p = np.clip(p, eps, None)
        if enforce_monotone_p:
            p = _project_monotone_decreasing(p)
        else:
            p = p / p.sum()

        base = U[:, None] * p[None, :]
        resid = np.where(mask, inc - base, 0.0)
        loss = float(np.sum(resid * resid))
        if not np.isfinite(loss):
            break
        if abs(loss - prev_loss) <= tol * max(prev_loss, 1.0):
            break
        prev_loss = loss

    base = U[:, None] * p[None, :]
    completed_inc = np.where(mask, inc, base)

    if looks_cumulative:
        completed_cum = np.nancumsum(completed_inc, axis=1)
        completed_cum = np.where(observed_mask, tri.values, completed_cum)
    else:
        completed_cum = np.nancumsum(completed_inc, axis=1)

    return pd.DataFrame(completed_cum, index=tri.index, columns=tri.columns)


# =========================================================
# Combinaison
# =========================================================
def combinaison(df: pd.DataFrame) -> pd.DataFrame:
    EPS_DENOM       = 0.0
    MAX_COMB_EXACT  = 10000
    USE_MONTE_CARLO = True
    MC_SAMPLES      = 10000
    RANDOM_SEED     = 1234

    keep = {}
    for c in df.columns:
        try:
            k = int(str(c).strip())
            keep[c] = k
        except Exception:
            pass
    if not keep:
        raise ValueError("Aucune colonne d'âge (entière) détectée dans `df`.")

    ages_sorted = sorted(keep.items(), key=lambda x: x[1])
    cols_old = [x[0] for x in ages_sorted]
    cols_new = [int(x[1]) for x in ages_sorted]

    cum = df.loc[:, cols_old].copy()
    cum.columns = cols_new
    cum = cum.sort_index(axis=1)
    cum = cum.apply(pd.to_numeric, errors="coerce")

    ages = list(cum.columns)
    if len(ages) < 2:
        return cum.copy()

    F = pd.DataFrame(index=cum.index, columns=ages[:-1], dtype=float)
    for j, j1 in zip(ages[:-1], ages[1:]):
        base = cum[j]
        nxt  = cum[j1]
        valid = base.notna() & nxt.notna() & (base > EPS_DENOM)
        F[j] = np.where(valid, nxt / base, np.nan)

    def _exact_mean_std_of_products(lists_of_values: list[list[float]]) -> tuple[float, float, int]:
        n_comb = 1
        for ls in lists_of_values:
            n_comb *= max(len(ls), 1)
        if any(len(ls) == 0 for ls in lists_of_values):
            return (np.nan, np.nan, 0)
        prods = [float(np.prod(combo)) for combo in itertools.product(*lists_of_values)]
        arr = np.asarray(prods, dtype=float)
        mean = float(arr.mean())
        std  = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        return (mean, std, n_comb)

    def _analytic_mean_std_of_products(means: np.ndarray, stds: np.ndarray) -> tuple[float, float]:
        if np.any(~np.isfinite(means)):
            return (np.nan, np.nan)
        prod_means = float(np.prod(means))
        second_mom = stds**2 + means**2
        var_prod = float(np.prod(second_mom)) - prod_means**2
        std_prod = math.sqrt(var_prod) if var_prod >= 0 else np.nan
        return prod_means, std_prod

    def _mc_mean_std_of_products(lists_of_values: list[np.ndarray], n: int, seed: int) -> tuple[float, float]:
        rng = np.random.default_rng(seed)
        pools = [np.asarray(v, dtype=float) for v in lists_of_values]
        if any(p.size == 0 for p in pools):
            return (np.nan, np.nan)
        out = np.empty(n, dtype=float)
        for i in range(n):
            prod = 1.0
            for p in pools:
                prod *= p[rng.integers(0, p.size)]
            out[i] = prod
        return float(out.mean()), float(out.std(ddof=1))

    def _compute_multiplier_between_indices(a0_idx: int, t_idx: int) -> tuple[float, float]:
        if t_idx <= a0_idx:
            return (1.0, 0.0)
        lists, means, stds = [], [], []
        for j_idx in range(a0_idx, t_idx):
            j_age = ages[j_idx]
            col = F[j_age].dropna().values
            col = col[np.isfinite(col)]
            col = col[col > 0]
            lists.append(col.tolist())
            means.append(col.mean() if col.size else np.nan)
            stds.append(col.std(ddof=1) if col.size > 1 else (0.0 if col.size == 1 else np.nan))
        means = np.asarray(means, dtype=float)
        stds  = np.asarray(stds, dtype=float)

        n_comb = 1
        for v in lists:
            n_comb *= len(v)
        if n_comb == 0:
            return (np.nan, np.nan)
        if n_comb <= MAX_COMB_EXACT:
            return _exact_mean_std_of_products(lists)[:2]
        if USE_MONTE_CARLO:
            return _mc_mean_std_of_products([np.asarray(v) for v in lists], MC_SAMPLES, RANDOM_SEED)
        return _analytic_mean_std_of_products(means, stds)

    completed = cum.copy()
    for e in completed.index:
        row = completed.loc[e]
        obs_mask = row.notna()
        if not obs_mask.any():
            continue
        a0_idx = np.where(obs_mask.values)[0][-1]
        a0_age = ages[a0_idx]
        base   = completed.at[e, a0_age]

        for t_idx in range(a0_idx + 1, len(ages)):
            t_age = ages[t_idx]
            mean_mult, _ = _compute_multiplier_between_indices(a0_idx, t_idx)
            pred = base * mean_mult if (np.isfinite(base) and np.isfinite(mean_mult)) else np.nan
            prev_val = completed.at[e, ages[t_idx - 1]]
            if np.isfinite(prev_val) and (not np.isfinite(pred) or pred < prev_val):
                pred = prev_val
            completed.at[e, t_age] = pred

    return completed
