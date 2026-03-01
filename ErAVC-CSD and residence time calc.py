import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# ==============================
# USER SETTINGS
# ==============================

FILES = {
    "Host_Plag": "HG13B-PLAG-Results.csv",
    "Host_Amph": "HG13B-AMPH-Results.csv",
    "Enclave_Plag": "HG13E-PLAG-Results.csv",
    "Enclave_Amph": "HG13E-AMPH-Results.csv"
}

FERET_COLUMN = "Feret"
CUTOFF = 50
N_BINS = 20
GROWTH_RATES = [1, 5]      # µm/day
BOOTSTRAP_ITER = 1000
AIC_THRESHOLD = 6

# ==============================
# FUNCTIONS
# ==============================

def compute_csd(lengths):
    log_min = np.log10(lengths.min())
    log_max = np.log10(lengths.max())
    bins = np.logspace(log_min, log_max, N_BINS)

    n, edges = np.histogram(lengths, bins=bins)
    deltaL = np.diff(edges)
    mid = np.sqrt(edges[:-1] * edges[1:])

    mask = n > 0
    n = n[mask]
    deltaL = deltaL[mask]
    mid = mid[mask]

    y = np.log(n / deltaL)
    return mid, y


def linear_fit(x, y):
    if len(np.unique(x)) < 2:
        return 0, 0, 0, np.inf

    slope, intercept, r, p, stderr = stats.linregress(x, y)
    y_pred = slope * x + intercept
    rss = np.sum((y - y_pred)**2)
    return slope, intercept, r, rss


def compute_aic(rss, n, k):
    if rss == np.inf or rss <= 0:
        return np.inf
    return n * np.log(rss/n) + 2*k


def piecewise_fit(x, y):
    if len(x) < 6:
        return None, None, np.inf

    best_rss = np.inf
    best_break = None
    best_params = None

    for i in range(3, len(x)-3):
        x1, y1 = x[:i], y[:i]
        x2, y2 = x[i:], y[i:]

        s1, i1, _, rss1 = linear_fit(x1, y1)
        s2, i2, _, rss2 = linear_fit(x2, y2)

        rss_total = rss1 + rss2

        if rss_total < best_rss:
            best_rss = rss_total
            best_break = x[i]
            best_params = (s1, i1, s2, i2)

    return best_break, best_params, best_rss


def bootstrap_breakpoint(x, y):
    breaks = []

    for _ in range(BOOTSTRAP_ITER):
        idx = np.random.choice(len(x), len(x), replace=True)
        xb, yb = x[idx], y[idx]

        order = np.argsort(xb)
        xb, yb = xb[order], yb[order]

        br, _, rss = piecewise_fit(xb, yb)

        if br is not None and rss != np.inf:
            breaks.append(br)

    if len(breaks) == 0:
        return None, None

    return np.percentile(breaks, [2.5, 97.5])


def residence_time(slope):
    slope = abs(slope)
    if slope == 0:
        return np.inf, np.inf

    taus = [1/(G*slope) for G in GROWTH_RATES]
    return min(taus), max(taus)


# ==============================
# MAIN ANALYSIS
# ==============================

results = []

for name, file in FILES.items():

    if not Path(file).exists():
        print(f"File not found: {file}")
        continue

    df = pd.read_csv(file)

    if FERET_COLUMN not in df.columns:
        print(f"{name}: Column '{FERET_COLUMN}' not found.")
        continue

    lengths = df[FERET_COLUMN].values
    lengths = lengths[lengths >= CUTOFF]

    if len(lengths) < 10:
        print(f"{name}: Not enough data after cutoff.")
        continue

    x, y = compute_csd(lengths)

    slope, intercept, r, rss_lin = linear_fit(x, y)
    aic_lin = compute_aic(rss_lin, len(x), 2)

    brk, params, rss_pw = piecewise_fit(x, y)
    aic_pw = compute_aic(rss_pw, len(x), 4)

    delta_aic = aic_lin - aic_pw
    ci_low, ci_high = bootstrap_breakpoint(x, y)
    tau_min, tau_max = residence_time(slope)

    # ===== Plot =====
    plt.figure(figsize=(7,6))
    plt.scatter(x, y, color='black', label="CSD data")
    plt.plot(x, slope*x + intercept, 'r--', label="Linear fit")

    if delta_aic > AIC_THRESHOLD and params is not None:
        s1, i1, s2, i2 = params
        idx = np.where(x >= brk)[0][0]
        plt.plot(x[:idx], s1*x[:idx]+i1, 'b')
        plt.plot(x[idx:], s2*x[idx:]+i2, 'b', label="Piecewise fit")

    if brk is not None:
        plt.axvline(brk, color='green', linewidth=2,
                    label=f"Breakpoint {brk:.1f} µm")

    if ci_low is not None:
        plt.axvspan(ci_low, ci_high, color='green', alpha=0.1,
                    label=f"95% CI {ci_low:.1f}-{ci_high:.1f}")

    plt.xlabel("Crystal length (µm)")
    plt.ylabel("ln(n/ΔL)")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.show()

    results.append({
        "Dataset": name,
        "Slope": slope,
        "R": r,
        "Delta_AIC": delta_aic,
        "Breakpoint": brk,
        "CI_low": ci_low,
        "CI_high": ci_high,
        "Tau_min_days": tau_min,
        "Tau_max_days": tau_max
    })

results_df = pd.DataFrame(results)
print(results_df)

if __name__ == "__main__":
    print("CSD analysis completed.")
