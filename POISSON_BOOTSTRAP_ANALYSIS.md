# Poisson Bootstrap Analysis Report

## Issue Summary
The Poisson Bootstrap method is producing triangles with **zeros in the last row and first column**.

---

## Problem Root Cause Analysis

### 1. **The Core Calculation Issue (AppMethods.py)**

#### The Problem Chain:

**Step 1: Input Data Type (Line 651)**
```python
tri_inc = _force_incremental_once(tri0)
```
- The function receives a **CUMULATIVE** triangle from AppvFinal.py
- Line 1217 in AppvFinal.py passes `base` (which is cumulative)
- The function converts it to **incremental** using `_force_incremental_once()`

**Step 2: Frontier Masking (Line 653)**
```python
obs_mask, fut_mask = _mask_frontier(tri_inc, zero_future_tol)
```
- This marks which cells are "observed" (obs_mask) vs "future" (fut_mask)
- The mask is based on the **incremental** triangle where small/near-zero values indicate future cells
- **CRITICAL**: The `zero_future_tol = 1e-6` parameter creates a problem!

**Step 3: The Problem - Zeros in inc_obs_vals (Line 697)**
```python
inc_obs_vals = np.where(obs_mask, np.nan_to_num(inc_vals, nan=0.0), 0.0).astype(float)
```
- This line creates the observed incremental values array
- For cells marked as "future" (obs_mask=False), it sets them to **0.0**
- Later, these zeros are never replaced during bootstrap (only future cells simulated)

**Step 4: First Column Preservation Fix (Line 710)**
```python
first_col_obs = inc_obs_vals[:, 0].copy()
# ... later in loop (Line 723):
inc_repl[:, 0] = first_col_obs
```
- This correctly preserves the first column
- ✅ This part is working correctly

**Step 5: Cumulative Sum (Line 725)**
```python
cum_repl = np.cumsum(inc_repl, axis=1)
```
- If the first column has zeros for the last row, cumsum will propagate zeros
- If any earlier column in the last row has been masked as "future", it will be 0

---

### 2. **The Real Problem: Zero Values in Incomplete Observed Rows**

The issue is in how `inc_obs_vals` is constructed:

```python
inc_obs_vals = np.where(obs_mask, np.nan_to_num(inc_vals, nan=0.0), 0.0).astype(float)
```

**The Problem:**
- `obs_mask` is True only for cells with:
  - Finite values
  - Absolute value > `zero_future_tol` (1e-6)
  - Before the "last valid" non-small value in each row

- For the **last row** (newest AY), if it has a recent value near zero (e.g., 100 in a column), the algorithm may:
  1. Mark it as the last observed
  2. But cells before it with NaN get converted to 0.0
  3. When cumsum is applied, these zeros become the cumulative values

**For the first column:**
- Actually should be OK with the current fix (Line 710)
- Unless the first column itself has zeros in the last row from the source data

---

### 3. **Collision Point with AppvFinal.py**

#### The Issue Flow:

1. **AppvFinal.py Line 1217:** Passes cumulative triangle to Poissonbootstrap
   ```python
   base_for_method = base  # This is CUMULATIVE
   ```

2. **AppMethods.py Line 651:** Tries to detect if cumulative
   ```python
   tri_inc = _force_incremental_once(tri0)
   ```

3. **AppMethods.py Line 572-592:** Cumulative detection check
   ```python
   if np.mean(is_monotone) >= 0.7:  # Requires 70% of rows monotone
       # Convert to incremental
   ```
   - If input is already **incremental** (not monotone), it stays incremental ✅
   - If input is **cumulative** (monotone), it converts correctly ✅

4. **The Frontier Masking Issue:** Lines 603-615
   ```python
   valid = (np.abs(row) > tol) & finite
   last_valid = np.max(np.where(valid)[0]) if valid.any() else -1
   if last_valid >= 0:
       obs[i, : last_valid + 1] = finite[: last_valid + 1]
       if last_valid + 1 < m:
           fut[i, last_valid + 1 :] = True
   ```

   **The Problem Here:**
   - For **incremental** data, very small values (< 1e-6) are treated as "future"
   - This is correct for incremental BUT can cause issues if:
     - First column of last row has real small incremental values
     - These get masked as "future" instead of "observed"

5. **AppvFinal.py Line 1233:** Alignment and filling
   ```python
   mask = completed.isna() & pred_full.notna()
   completed[mask] = pred_full[mask]
   completed = completed.ffill(axis=1)
   ```
   - This tries to forward-fill, but if pred_full has zeros, they propagate

---

## Root Cause: The `zero_future_tol` Parameter

The `zero_future_tol = 1e-6` is **too aggressive** for incremental triangles.

### Why:
- In **cumulative** triangles: 1e-6 makes sense (very small cumulative values = future)
- In **incremental** triangles: Small values (100, 1000) are legitimate observed incremental claims
- But 1e-6 is so small it won't affect real numbers

**Wait - the real issue:**

The frontier mask is being applied to the **incremental** triangle incorrectly:

```python
valid = (np.abs(row) > tol) & finite
```

This correctly identifies non-zero incremental cells. The problem is that when we later do:

```python
inc_obs_vals = np.where(obs_mask, np.nan_to_num(inc_vals, nan=0.0), 0.0).astype(float)
```

We're replacing **all NaN values with 0.0**, which is wrong for future cells.

---

## The Actual Bug

**The bug is NOT in the calculation functions - it's in how `inc_obs_vals` is initialized!**

Line 697 in AppMethods.py:
```python
inc_obs_vals = np.where(obs_mask, np.nan_to_num(inc_vals, nan=0.0), 0.0).astype(float)
```

This line does:
```
Where obs_mask is True:  use inc_vals with NaNs replaced by 0
Where obs_mask is False: use 0
```

**The problem:** We're treating NaNs as 0s, which makes the first column look like zeros if there are NaNs in the observed region.

---

## Solution

Change line 697 to preserve the original structure better:

```python
# Preserve both observed AND the structure - NaNs should stay NaN for unobserved, not become 0
inc_obs_vals = inc_vals.copy().to_numpy(dtype=float)  # Keep structure as-is
```

OR more carefully:

```python
# Only set future cells to 0, leave observed alone
inc_obs_vals = inc_vals.to_numpy(dtype=float)
# Set future cells to 0 for simulation
inc_obs_vals[~obs_mask] = 0.0
# Observed NaNs should be converted only if needed for GLM
inc_obs_vals = np.where(obs_mask, np.nan_to_num(inc_vals.to_numpy(dtype=float), nan=0.0), 0.0).astype(float)
```

**WAIT** - this is already what it does. The real issue is different.

---

## The ACTUAL Root Cause

Looking more carefully at the flow:

1. Input to Poisson is **cumulative**
2. Gets converted to **incremental** ✅
3. First column of incremental = first column of cumulative ✅
4. But when extracting `y_obs` (line 665):
   ```python
   y_all = tri_inc.to_numpy(dtype=float)[obs_mask]
   ```
   Only the cells marked in `obs_mask` are used for GLM fitting

5. The last row issue:
   - If the last row has been incomplete (only a few values), the frontier mask might mark only the observed columns
   - When `inc_obs_vals` is created, unobserved cells become 0
   - When zeros are cumsum'd, they stay as zeros for those incomplete rows
   - **But the last row should be fully observed!** (Line 615: `if not obs[-1, :].any(): obs[-1, :] = np.isfinite(arr[-1, :])`)

6. So the issue is: **Last row might have NaN values that shouldn't exist**
   - If the original cumulative had NaN in last row, it stays NaN when converted to incremental
   - This NaN becomes 0 in inc_obs_vals (line 697)
   - When cumsum is applied, the zeros propagate

---

## Diagnosis Conclusion

| Component | Status | Notes |
|-----------|--------|-------|
| Calculation Functions | ✅ CORRECT | The math is sound |
| First Column Preservation | ✅ CORRECT | Line 710 fixes it properly |
| Report Generation | ✅ CORRECT | Report generation is fine |
| Frontier Masking Logic | ⚠️ ACCEPTABLE | Works but has edge cases |
| **inc_obs_vals Initialization** | ❌ **ISSUE** | NaN → 0 conversion causes zeros to propagate |
| **Last Row Handling** | ⚠️ ISSUE | Last row might have legitimate NaNs that become 0s |

---

## Final Verdict

**No collision with AppvFinal.py** - The UI is calling the method correctly.

**The issue IS in AppMethods.py, in the calculation setup (not the actual calculation):**

The zeros appear because:
1. NaN values in the cumulative triangle become 0 in the incremental conversion
2. These zeros are never simulated (only future cells are)
3. When cumsum is applied, zeros remain as zeros
4. This particularly affects rows with missing data

**Fix needed:** The frontier masking and inc_obs_vals initialization needs to properly handle missing data in the last row (most recent AY).
