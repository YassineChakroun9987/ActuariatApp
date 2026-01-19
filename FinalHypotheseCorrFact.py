from __future__ import annotations
import numpy as np
import pandas as pd
import math

INPUT_PATH = "TESTHYP.xlsx"
INPUT_SHEET = "Sheet1"
INDEX_COL = 0

def read_cumulative_triangle(path: str, sheet: str, index_col=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if index_col is not None:
        if isinstance(index_col, (int, np.integer)) or (isinstance(index_col, str) and str(index_col).isdigit()):
            pos = int(index_col)
            idx_name = df.columns[pos]
            df = df.set_index(idx_name)
            if pos > 0:
                df = df.drop(columns=df.columns[:pos], errors="ignore")
        else:
            if index_col not in df.columns:
                raise KeyError(f"index_col '{index_col}' not found in columns: {list(df.columns)}")
            df = df.set_index(index_col)
    return df

def AssumptionFactors(df: pd.DataFrame, output_path: str = "Correlation_Factors_Output2.xlsx", r2_threshold: float = 0.95, tail_factor: float = 1.0, eps_denom: float = 0.0) -> bool:
    def _coerce_age_columns(df_: pd.DataFrame) -> pd.DataFrame:
        keep = {}
        for c in df_.columns:
            try:
                keep[c] = int(str(c).strip())
            except Exception:
                pass
        if not keep:
            raise ValueError("No integer-like development-age columns found.")
        cols_sorted = [k for k,_ in sorted([(c, keep[c]) for c in keep], key=lambda x: x[1])]
        sub = df_.loc[:, cols_sorted].copy()
        sub.columns = [int(str(c).strip()) for c in sub.columns]
        sub = sub.sort_index(axis=1)
        return sub.apply(pd.to_numeric, errors="coerce")

    def origin_through_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        x, y = x[mask], y[mask]
        if x.size == 0:
            return (np.nan, np.nan)
        xx = float(np.dot(x, x))
        xy = float(np.dot(x, y))
        yy = float(np.dot(y, y))
        beta = xy / xx if xx > 0 else np.nan
        r2 = (xy * xy) / (xx * yy) if (xx > 0 and yy > 0) else np.nan
        return beta, r2

    def regression_diagnostics(cum: pd.DataFrame, eps: float=0.0) -> pd.DataFrame:
        ages = list(cum.columns)
        out = []
        for j, j1 in zip(ages[:-1], ages[1:]):
            x = cum[j].to_numpy(dtype=float)
            y = cum[j1].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y) & (x > eps)
            beta, r2 = origin_through_regression(x[mask], y[mask])
            out.append({"age": j, "beta_reg": beta, "R2": r2, "n": int(mask.sum())})
        return pd.DataFrame(out).set_index("age")

    def chainladder_factors(cum: pd.DataFrame, eps: float=0.0) -> pd.Series:
        ages = list(cum.columns)
        facs = {}
        for j, j1 in zip(ages[:-1], ages[1:]):
            base = cum[j]; nxt = cum[j1]
            mask = base.notna() & nxt.notna() & (base > eps)
            num = nxt[mask].sum(); den = base[mask].sum()
            facs[j] = float(num/den) if den > 0 else np.nan
        return pd.Series(facs)

    def cadence_from_factors(factors: pd.Series, tail_factor_: float = 1.0) -> pd.Series:
        facs = factors.dropna().astype(float).tolist()
        facs_full = facs + [float(tail_factor_)]
        T = len(facs_full)
        p = np.zeros(T + 1, dtype=float)
        p[T] = 1.0
        for k in range(T-1, 0, -1):
            p[k] = p[k+1] / facs_full[k-1]
        w = p[1:] - p[:-1]
        ages = list(factors.index)
        cadence = pd.Series(w[:-1], index=ages, name="Cadence")
        cadence.loc["Ultime"] = 0.0
        return cadence

    def build_facteurs_table(cum: pd.DataFrame, tail_factor_: float) -> pd.DataFrame:
        facs = chainladder_factors(cum, eps=eps_denom)
        fac_row = facs.copy()
        fac_row.loc["Ultime"] = float(tail_factor_)
        cad_row = cadence_from_factors(facs, tail_factor_=tail_factor_)
        return pd.DataFrame([fac_row, cad_row], index=["Fac. Rete", "Cadence"])

    cum = _coerce_age_columns(df.copy())
    diag_df = regression_diagnostics(cum, eps=eps_denom)
    facteurs_table = build_facteurs_table(cum, tail_factor_=tail_factor)

    with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
        cum.to_excel(xw, sheet_name="Original_Cumulative")
        diag_df.to_excel(xw, sheet_name="Regression_R2")
        facteurs_table.to_excel(xw, sheet_name="Facteurs_Cadence")

    ok = False if diag_df.empty else not (diag_df["R2"] < float(r2_threshold)).any()
    return bool(ok)

def main():
    df = read_cumulative_triangle(INPUT_PATH, INPUT_SHEET, index_col=INDEX_COL)
    valid = AssumptionFactors(df)
    print("VALID" if valid else "NOT VALID")

if __name__ == "__main__":
    main()
