
def BornhuetterFerguson(
    df: pd.DataFrame,
    primes_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Bornhuetter-Ferguson / Chain Ladder hybrid (robust version).

    - Ultimates and IBNR are BF-based (ELR × Premium)
    - Future Cij are filled using CL timing
    - Missing or invalid premiums fall back gracefully (no crash)
    """

    # --------------------------------------------------
    # 0. Prepare triangle (same spirit as old function)
    # --------------------------------------------------
    tri = df.apply(pd.to_numeric, errors="coerce").copy()

    # Detect cumulative
    def _is_cum(row):
        v = row.dropna().values
        if len(v) < 2:
            return True
        return np.all(np.diff(v) >= -1e-12)

    is_cumulative = tri.apply(_is_cum, axis=1).mean() >= 0.5

    if not is_cumulative:
        tri = tri.fillna(0).cumsum(axis=1).where(tri.notna())

    tri = tri.sort_index()
    tri.index = tri.index.astype(int)

    ages = list(tri.columns)
    n_dev = len(ages)

    # --------------------------------------------------
    # 1. Chain Ladder development (timing only)
    # --------------------------------------------------
    dev_factors = []
    for j in range(n_dev - 1):
        num = tri.iloc[:, j + 1].sum(skipna=True)
        den = tri.iloc[:, j].sum(skipna=True)
        dev_factors.append(num / den if den > 0 else 1.0)

    dev_factors = np.array(dev_factors)

    cdfs = np.flip(np.cumprod(np.flip(dev_factors)))
    cdfs = np.append(cdfs, 1.0)
    percent_reported = 1.0 / cdfs

    # --------------------------------------------------
    # 2. Expected Loss Ratio (robust, old behavior)
    # --------------------------------------------------
    loss_ratio = None

    if primes_df is not None and not primes_df.empty and "Premium" in primes_df.columns:
        try:
            first_ay = tri.index[0]
            first_ult = tri.loc[first_ay].iloc[-1]
            first_prem = primes_df.loc[int(first_ay), "Premium"]

            if pd.notna(first_prem) and first_prem > 0 and pd.notna(first_ult):
                loss_ratio = float(first_ult) / float(first_prem)
        except Exception:
            pass

    if loss_ratio is None or not np.isfinite(loss_ratio):
        loss_ratio = 1.0  # identical fallback to your old function

    # --------------------------------------------------
    # 3. BF ultimate + CL timing fill
    # --------------------------------------------------
    completed = tri.copy()

    for ay in tri.index:
        row = tri.loc[ay]
        last_dev = row.last_valid_index()
        k = ages.index(last_dev)

        reported = row.iloc[k]
        alpha_k = percent_reported[k]

        # --- Expected ultimate (robust premium access)
        ultimate = reported

        if primes_df is not None and not primes_df.empty and "Premium" in primes_df.columns:
            try:
                prem = primes_df.loc[int(ay), "Premium"]
                if pd.notna(prem) and prem > 0:
                    expected_ultimate = prem * loss_ratio
                    ibnr = expected_ultimate * (1 - alpha_k)
                    ultimate = reported + ibnr
                else:
                    ibnr = 0.0
            except Exception:
                ibnr = 0.0
        else:
            ibnr = 0.0

        # --- Allocate IBNR using CL timing
        if k < n_dev - 1 and ibnr > 0:
            remaining_cdfs = cdfs[k:] / cdfs[k]
            incr_props = remaining_cdfs[:-1] - remaining_cdfs[1:]

            cumulative = reported
            for j, prop in enumerate(incr_props, start=k + 1):
                cumulative += ibnr * prop
                completed.at[ay, ages[j]] = cumulative

            completed.at[ay, ages[-1]] = ultimate
        else:
            completed.at[ay, ages[-1]] = ultimate

    return completed
