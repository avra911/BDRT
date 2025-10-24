# Bitcoin Diminished Return Theory (BDRT) Model
# Copyright (c) 2025, Razvan AVRAMESCU
# Licensed under CC BY 4.0 – You may share and adapt, but must credit the original author.
#
# Citation:
# Razvan AVRAMESCU. "Bitcoin Diminished Return Theory - A Log-Linear Model of Cycle-Based Decay."
# https://www.avramescu.net/bitcoins-diminishing-monthly-returns-a-decade-ahead/
#
# Disclaimer:
# This is a research model intended for educational and analytical purposes.
# It is not financial advice and does not guarantee market outcomes.

import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==== Configuration / Defaults ====
BULL_MONTHS = 36
BEAR_MONTHS = 12
BEAR_DROP = 0.77 # total drop during bear (e.g. 77% drop -> remaining factor 23%)
DEFAULT_START_PRICE = 126_200
DEFAULT_START_DATE = "2025-10-19"


def fit_diminishing_returns(method: str = "include_all", floor: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a log-linear model on historical 4-year monthly averages and extrapolate
    monthly 4-year averages (r_avg_4y) for future cycles (cycles 5..11).

    Parameters
    - method: one of "exclude_first", "include_all", "weighted"
      * "exclude_first": use only cycles 2-4
      * "include_all": use cycles 1-4 with equal weight
      * "weighted": use cycles 1-4 but downweight cycle 1 (outlier)
    - floor: minimum allowed value for extrapolated averages (prevents non-positive results)

    Returns
    - future_cycles: np.ndarray of cycle indices (5..11)
    - r_avg_4y: np.ndarray of monthly 4-year averages for those cycles (decimal, e.g. 0.02 = 2%)
    """
    # historical cycles (monthly 4-year averages, decimals)
    cycles_hist = np.array([1, 2, 3, 4])
    r_hist = np.array([0.4439, 0.0753, 0.0445, 0.0276])

    if method == "exclude_first":
        mask = cycles_hist >= 2
        cycles_use = cycles_hist[mask]
        r_use = r_hist[mask]
        weights = None
    elif method == "include_all":
        cycles_use = cycles_hist
        r_use = r_hist
        weights = None
    elif method == "weighted":
        cycles_use = cycles_hist
        r_use = r_hist
        # downweight cycle 1 as an outlier
        weights = np.array([0.2, 1.0, 1.0, 1.0])
    else:
        raise ValueError("method must be 'exclude_first', 'include_all', or 'weighted'")

    # Fit log-linear model: log(r) = a + b * cycle
    X = np.vstack([np.ones_like(cycles_use), cycles_use]).T
    y = np.log(r_use)

    if weights is not None:
        W = np.sqrt(weights)
        Xw = X * W[:, None]
        yw = y * W
        a, b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    else:
        a, b = np.linalg.lstsq(X, y, rcond=None)[0]

    # Extrapolate for future cycles (5..11)
    future_cycles = np.arange(5, 12)
    r_avg_4y = np.exp(a + b * future_cycles)

    # Apply floor to avoid zero/negative values
    if floor is not None:
        r_avg_4y = np.maximum(r_avg_4y, floor)

    return future_cycles, r_avg_4y


def bull_rate_from_avg(r_avg: float, bear_drop: float = BEAR_DROP) -> float:
    """
    Compute monthly bull rate such that the combined 48-month factor
    (36 bull months + 12 bear months) yields the given 4-year average r_avg,
    accounting for the bear drop.
    """
    F_bear = 1 - bear_drop
    target_factor_48 = (1 + r_avg) ** (BULL_MONTHS + BEAR_MONTHS)
    g_bull = (target_factor_48 / F_bear) ** (1 / BULL_MONTHS) - 1
    return g_bull


def build_trajectory(
    cycles: np.ndarray,
    r_avg_4y: np.ndarray,
    start_price: float = DEFAULT_START_PRICE,
    start_date: str = DEFAULT_START_DATE,
    bear_drop: float = BEAR_DROP,
) -> pd.DataFrame:
    """
    Build a monthly price trajectory for given cycles and per-cycle 4y averages.

    Returns a DataFrame with columns:
    date, price_usd, phase, cycle, bull_monthly, bear_drop_total, r_avg_total
    """
    total_months = len(cycles) * (BULL_MONTHS + BEAR_MONTHS)
    dates = pd.date_range(start=start_date, periods=total_months, freq="MS")

    prices, phases, cycles_col = [], [], []
    bull_rates, bear_rates, avg_rates = [], [], []

    price = float(start_price)
    i_date = 0

    for cyc, r_avg in zip(cycles, r_avg_4y):
        g_bull = bull_rate_from_avg(r_avg, bear_drop)
        F_bear = 1 - bear_drop
        m_factor_bear = F_bear ** (1 / BEAR_MONTHS)

        # 1 year bear
        for _ in range(BEAR_MONTHS):
            if i_date >= len(dates):
                break
            price *= m_factor_bear
            prices.append(price)
            phases.append("Bear")
            cycles_col.append(int(cyc))
            bull_rates.append(g_bull)
            bear_rates.append(bear_drop)
            avg_rates.append(r_avg)
            i_date += 1

        # 3 years bull
        for _ in range(BULL_MONTHS):
            if i_date >= len(dates):
                break
            price *= (1 + g_bull)
            prices.append(price)
            phases.append("Bull")
            cycles_col.append(int(cyc))
            bull_rates.append(g_bull)
            bear_rates.append(bear_drop)
            avg_rates.append(r_avg)
            i_date += 1

    df = pd.DataFrame({
        "date": dates[:i_date],
        "price_usd": np.round(prices[:i_date], 2),
        "phase": phases[:i_date],
        "cycle": cycles_col[:i_date],
        "bull_monthly": bull_rates[:i_date],
        "bear_drop_total": bear_rates[:i_date],
        "r_avg_total": avg_rates[:i_date],
    })

    return df


def plot_trajectory(df: pd.DataFrame, save_fig: str | None = None) -> None:
    """Plot trajectory with bull/bear coloring, mid-cycle labels, and peaks/bottoms."""
    plt.style.use("dark_background")
    plt.figure(figsize=(13, 7))
    plt.ticklabel_format(style="plain", axis="y")
    ax = plt.gca()
    ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # colored line segments
    for i in range(len(df) - 1):
        color = "#00ff66" if df["phase"].iloc[i] == "Bull" else "#ff4444"
        plt.plot(df["date"].iloc[i:i+2], df["price_usd"].iloc[i:i+2], color=color, linewidth=2)

    # cycle labels (midpoint)
    for cyc in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == cyc]
        if sub.empty:
            continue
        mid_idx = len(sub) // 2
        mid_date = sub["date"].iloc[mid_idx]
        mid_price = sub["price_usd"].iloc[mid_idx]
        r_avg_display = sub["r_avg_total"].iloc[0] * 100
        bull_r_display = sub["bull_monthly"].iloc[0] * 100
        plt.text(
            mid_date, mid_price,
            f"≈{r_avg_display:.2f}% avg (4y)\n+{bull_r_display:.2f}% bull",
            color="#66ccff", fontsize=9, ha="center", va="bottom",
            bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=1)
        )

    # mid-to-mid dashed trendline
    cycle_mids, cycle_mid_prices = [], []
    for cyc in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == cyc]
        if sub.empty:
            continue
        mid_idx = len(sub) // 2
        cycle_mids.append(sub["date"].iloc[mid_idx])
        cycle_mid_prices.append(sub["price_usd"].iloc[mid_idx])
    plt.plot(cycle_mids, cycle_mid_prices, color="white", linewidth=2.5, linestyle="--")
    plt.scatter(cycle_mids, cycle_mid_prices, color="white", s=30, zorder=5)

    # peaks & bottoms
    for cyc in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == cyc]
        if sub.empty:
            continue
        peak_row = sub.loc[sub["price_usd"].idxmax()]
        bottom_row = sub.loc[sub["price_usd"].idxmin()]
        plt.scatter(peak_row["date"], peak_row["price_usd"], color="lime", s=50)
        plt.text(peak_row["date"], peak_row["price_usd"] * 1.05,
                 f"Peak {int(peak_row['price_usd']):,}", color="lime", fontsize=8, ha="center")
        plt.scatter(bottom_row["date"], bottom_row["price_usd"], color="orange", s=50)
        plt.text(bottom_row["date"], bottom_row["price_usd"] * 0.85,
                 f"Bottom {int(bottom_row['price_usd']):,}", color="orange", fontsize=8, ha="center")

    start_year, end_year = df["date"].iloc[0].year, df["date"].iloc[-1].year
    plt.title(
        f"Bitcoin Projection {start_year}-{end_year} - Each 4y Cycle: 1y Bear (−{BEAR_DROP*100:.0f}%) → 3y Bull",
        color="white", fontsize=13
    )
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.ylim(0, df["price_usd"].max() * 1.1)

    yticks = np.linspace(0, df["price_usd"].max(), 20)
    plt.yticks(yticks, [f"{int(y):,}" for y in yticks])
    plt.tight_layout()

    if save_fig:
        plt.savefig(save_fig, dpi=200, bbox_inches="tight")
    plt.show()


def summarize_cycles(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with key metrics per cycle."""
    rows = []
    for cyc in sorted(df["cycle"].unique()):
        sub = df[df["cycle"] == cyc]
        if sub.empty:
            continue
        start_price = float(sub.iloc[0]["price_usd"])
        peak_price = float(sub["price_usd"].max())
        bottom_price = float(sub["price_usd"].min())
        bull_rate = float(sub["bull_monthly"].dropna().iloc[0]) * 100
        avg_4y = float(sub["r_avg_total"].iloc[0]) * 100
        rows.append({
            "Cycle": int(cyc),
            "Start_after_bear_USD": round(start_price),
            "Bull_%_per_month": round(bull_rate, 2),
            "Peak_USD": round(peak_price),
            "Bottom_after_bear_USD": round(bottom_price),
            "Avg_4y_%": round(avg_4y, 2),
            "Bear_drop_%": int(BEAR_DROP * 100),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diminishing-cycle Bitcoin projection (4y cycles).")
    parser.add_argument("--method", choices=["exclude_first", "include_all", "weighted"], default="include_all")
    parser.add_argument("--floor", type=float, default=1e-4, help="Minimum monthly average (decimal)")
    parser.add_argument("--start-price", type=float, default=DEFAULT_START_PRICE)
    parser.add_argument("--start-date", type=str, default=DEFAULT_START_DATE)
    parser.add_argument("--save-csv", type=str, default=None, help="If set, save trajectory to this CSV path")
    parser.add_argument("--save-fig", type=str, default=None, help="If set, save plot image to this path")
    args = parser.parse_args()

    cycles, r_avg_4y = fit_diminishing_returns(method=args.method, floor=args.floor)
    print("Monthly 4y averages per future cycle (%):")
    print(np.round(r_avg_4y * 100, 3))

    df = build_trajectory(cycles, r_avg_4y, start_price=args.start_price, start_date=args.start_date)

    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
    plot_trajectory(df, save_fig=args.save_fig)

    summary_df = summarize_cycles(df)
    print("\n=== Cycle Summary (1y bear → 3y bull) ===")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
