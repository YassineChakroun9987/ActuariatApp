from __future__ import annotations
import math
import numpy as np
import pandas as pd

INPUT_PATH= "TESTHYP.xlsx"
INPUT_SHEET= "Sheet1"
INDEX_COL= 0


def main():
    df = read_cumulative_triangle(INPUT_PATH, INPUT_SHEET, index_col=INDEX_COL)
    AssumptionCalendar(df)
    if(AssumptionCalendar(df)==True):
        print('VALID')
    else: print("NOT VALID")

def read_cumulative_triangle(path: str, sheet: str, index_col=0) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    if index_col is not None:
        idx = df.columns[index_col]
        df = df.set_index(idx)
        df = df.drop(columns=[c for c in df.columns[:index_col]], errors="ignore")

    return df


def AssumptionCalendar(df: pd.DataFrame, output_path: str = "Calendar_Effect_Test2.xlsx", z_alpha_2: float = 1.96, eps_denom: float = 0.0, tie_tol: float = 0.0) -> bool:
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

    def d_triangle_from_cumulative(cum: pd.DataFrame, eps: float=0.0) -> pd.DataFrame:
        ages = list(cum.columns)
        if len(ages) < 2:
            raise ValueError("Need at least 2 age columns.")
        D = pd.DataFrame(index=cum.index, columns=ages[:-1], dtype=float)
        for j, j1 in zip(ages[:-1], ages[1:]):
            base = cum[j]; nxt = cum[j1]
            valid = base.notna() & nxt.notna() & (base > eps)
            D[j] = np.where(valid, nxt / base, np.nan)
        return D

    def binary_triangle_by_column(D: pd.DataFrame, tie_tol_: float) -> tuple[pd.DataFrame, pd.Series]:
        Dn = D.apply(pd.to_numeric, errors="coerce")
        med = Dn.median(axis=0, skipna=True)
        diff = Dn.subtract(med, axis=1)
        if tie_tol_ > 0:
            pos = diff.gt(tie_tol_)
            neg = diff.lt(-tie_tol_)
        else:
            pos = diff.gt(0)
            neg = diff.lt(0)
        B = pd.DataFrame(np.where(pos, 1.0, np.where(neg, 0.0, np.nan)), index=Dn.index, columns=Dn.columns)
        return B, med

    def diagonals_indices(df_: pd.DataFrame):
        rows = list(df_.index); cols = list(df_.columns)
        rpos = {r:i for i,r in enumerate(rows)}
        cpos = {c:i for i,c in enumerate(cols)}
        buckets = {}
        for r in rows:
            for c in cols:
                v = df_.at[r, c]
                if pd.isna(v):
                    continue
                k = rpos[r] + cpos[c]
                buckets.setdefault(k, []).append((r, c))
        for k in buckets:
            buckets[k].sort(key=lambda rc: cpos[rc[1]])
        for k in sorted(buckets.keys()):
            yield k, buckets[k]

    def diag_stats_from_binary(B: pd.DataFrame) -> pd.DataFrame:
        out = []
        for k, cells in diagonals_indices(B):
            vals = [B.at[r, c] for (r, c) in cells]
            vals = [v for v in vals if pd.notna(v)]
            if not vals:
                continue
            P = sum(1 for v in vals if v == 1.0)
            Q = sum(1 for v in vals if v == 0.0)
            N = P + Q
            S = min(P, Q)
            Z = S
            m = int(N // 2)
            Comb = math.comb(N - 1, m) if N >= 1 else 0
            EZ = (N / 2.0) - Comb * (N / (2.0 ** N)) if N > 0 else 0.0
            VZ = (N * (N - 1) / 4.0) - Comb * ((N * (N - 1)) / (2.0 ** N)) + EZ - EZ ** 2 if N > 1 else 0.0
            out.append({"CalendarDiag": k, "P": P, "Q": Q, "N": N, "S": S, "Z": Z, "m": m, "Comb": Comb, "E(Z)": EZ, "V(Z)": VZ})
        return pd.DataFrame(out)

    def totals_and_ci(diag_tbl: pd.DataFrame, z_: float):
        Z_total = diag_tbl["Z"].sum()
        EZ_sum  = diag_tbl["E(Z)"].sum()
        VZ_sum  = diag_tbl["V(Z)"].sum()
        sd = math.sqrt(VZ_sum)
        return {"Z_total": Z_total, "E(Z)_sum": EZ_sum, "V(Z)_sum": VZ_sum, "CI_low": EZ_sum - z_*sd, "CI_high": EZ_sum + z_*sd}

    cum = _coerce_age_columns(df.copy())
    D = d_triangle_from_cumulative(cum, eps=eps_denom)
    B, med = binary_triangle_by_column(D, tie_tol_=tie_tol)
    diag_tbl = diag_stats_from_binary(B)
    if diag_tbl.empty:
        with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
            cum.to_excel(xw, sheet_name="Original_Cumulative")
            D.to_excel(xw, sheet_name="D_Triangle_AgeToAge")
            med.to_frame("Median_by_Age").to_excel(xw, sheet_name="Column_Medians")
            B.to_excel(xw, sheet_name="Binary_01_by_Age")
            pd.DataFrame([], columns=["CalendarDiag","P","Q","N","S","Z","m","Comb","E(Z)","V(Z)"]).to_excel(xw, sheet_name="Diagonal_Stats", index=False)
            pd.DataFrame([{"Z_total": np.nan, "E(Z)_sum": np.nan, "V(Z)_sum": np.nan, "CI_low": np.nan, "CI_high": np.nan}]).to_excel(xw, sheet_name="Totals_and_CI", index=False)
        return True
    summary = totals_and_ci(diag_tbl, z_alpha_2)
    with pd.ExcelWriter(output_path, engine="openpyxl") as xw:
        cum.to_excel(xw, sheet_name="Original_Cumulative")
        D.to_excel(xw, sheet_name="D_Triangle_AgeToAge")
        med.to_frame("Median_by_Age").to_excel(xw, sheet_name="Column_Medians")
        B.to_excel(xw, sheet_name="Binary_01_by_Age")
        diag_tbl.to_excel(xw, sheet_name="Diagonal_Stats", index=False)
        pd.DataFrame([summary]).to_excel(xw, sheet_name="Totals_and_CI", index=False)
    ok =not (summary['Z_total'] < summary['CI_low'] or summary['Z_total'] > summary['CI_high'])
    return bool(ok)


if __name__ == "__main__":
    main()
