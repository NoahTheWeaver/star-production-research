#!/usr/bin/env python3
"""
Test whether STAR production data fits a straight line or a curve.
Compares linear, quadratic, and piecewise-linear models.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from itertools import combinations

BASE = os.path.dirname(os.path.abspath(__file__))

DATE_CORRECTIONS = {"E309": "2020-05-21"}


def load_data():
    import re
    with open(os.path.join(BASE, "extracted_data.json")) as f:
        raw = json.load(f)

    dates, ordinals, labels = [], [], []
    for r in raw:
        sn = r["serial_number"]
        date_str = DATE_CORRECTIONS.get(sn, r["installation_date"])
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue

        # Convert serial to ordinal
        if re.match(r'^[A-H]\d{3}$', sn):
            ordinal = (ord(sn[0]) - ord('A')) * 1000 + int(sn[1:])
        elif re.match(r'^\d{4}$', sn):
            ordinal = int(sn)
        else:
            continue  # skip variant formats for regression analysis

        dates.append(d)
        ordinals.append(ordinal)
        labels.append(sn)

    return np.array(dates), np.array(ordinals), labels


def fit_and_compare(dates, ordinals, labels):
    """Compare linear, quadratic, and piecewise-linear fits."""
    date_nums = np.array([mdates.date2num(d) for d in dates])

    # Normalize for numerical stability
    x_mean = date_nums.mean()
    x_std = date_nums.std()
    x_norm = (date_nums - x_mean) / x_std

    n = len(date_nums)

    # --- Linear fit ---
    coeffs_lin = np.polyfit(x_norm, ordinals, 1)
    pred_lin = np.polyval(coeffs_lin, x_norm)
    resid_lin = ordinals - pred_lin
    ss_res_lin = np.sum(resid_lin ** 2)
    ss_tot = np.sum((ordinals - ordinals.mean()) ** 2)
    r2_lin = 1 - ss_res_lin / ss_tot
    aic_lin = n * np.log(ss_res_lin / n) + 2 * 2  # 2 params

    # --- Quadratic fit ---
    coeffs_quad = np.polyfit(x_norm, ordinals, 2)
    pred_quad = np.polyval(coeffs_quad, x_norm)
    resid_quad = ordinals - pred_quad
    ss_res_quad = np.sum(resid_quad ** 2)
    r2_quad = 1 - ss_res_quad / ss_tot
    aic_quad = n * np.log(ss_res_quad / n) + 2 * 3  # 3 params

    # --- Piecewise linear (find optimal breakpoint) ---
    best_bp_aic = float('inf')
    best_bp_date = None
    best_bp_r2 = 0
    best_bp_slopes = None
    best_bp_preds = None

    # Try breakpoints at each data point (between 20th and 80th percentile)
    sorted_idx = np.argsort(date_nums)
    lo = int(n * 0.2)
    hi = int(n * 0.8)

    for bp_idx in range(lo, hi):
        bp = date_nums[sorted_idx[bp_idx]]
        mask1 = date_nums <= bp
        mask2 = date_nums > bp

        if mask1.sum() < 3 or mask2.sum() < 3:
            continue

        # Fit two separate lines
        c1 = np.polyfit(date_nums[mask1], ordinals[mask1], 1)
        c2 = np.polyfit(date_nums[mask2], ordinals[mask2], 1)

        pred_pw = np.zeros(n)
        pred_pw[mask1] = np.polyval(c1, date_nums[mask1])
        pred_pw[mask2] = np.polyval(c2, date_nums[mask2])

        ss_res_pw = np.sum((ordinals - pred_pw) ** 2)
        r2_pw = 1 - ss_res_pw / ss_tot
        aic_pw = n * np.log(ss_res_pw / n) + 2 * 5  # 5 params (2 slopes, 2 intercepts, 1 breakpoint)

        if aic_pw < best_bp_aic:
            best_bp_aic = aic_pw
            best_bp_date = mdates.num2date(bp).date()
            best_bp_r2 = r2_pw
            best_bp_slopes = (c1[0] * 365.25, c2[0] * 365.25)  # units/year
            best_bp_preds = pred_pw.copy()
            best_bp_mask1 = mask1.copy()
            best_bp_mask2 = mask2.copy()
            best_bp_c1 = c1
            best_bp_c2 = c2

    # --- Print comparison ---
    print("=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(f"{'Model':<25} {'R²':<10} {'AIC':<12} {'Params':<8}")
    print("-" * 65)
    print(f"{'Linear':<25} {r2_lin:<10.4f} {aic_lin:<12.1f} {'2':<8}")
    print(f"{'Quadratic':<25} {r2_quad:<10.4f} {aic_quad:<12.1f} {'3':<8}")
    print(f"{'Piecewise (best)':<25} {best_bp_r2:<10.4f} {best_bp_aic:<12.1f} {'5':<8}")
    print("-" * 65)
    print()

    # Determine winner
    models = [("Linear", aic_lin, r2_lin), ("Quadratic", aic_quad, r2_quad),
              ("Piecewise", best_bp_aic, best_bp_r2)]
    models.sort(key=lambda m: m[1])
    print(f"Best model by AIC: {models[0][0]} (AIC={models[0][1]:.1f})")
    print()

    slope_lin = coeffs_lin[0] / x_std * 365.25
    print(f"Linear slope: {slope_lin:.0f} units/year")
    print(f"Quadratic leading coeff: {coeffs_quad[0]:.1f} (positive = acceleration)")
    print(f"Piecewise breakpoint: {best_bp_date}")
    print(f"  Before breakpoint: {best_bp_slopes[0]:.0f} units/year")
    print(f"  After breakpoint:  {best_bp_slopes[1]:.0f} units/year")
    print(f"  Rate change: {best_bp_slopes[1] - best_bp_slopes[0]:+.0f} units/year")

    # --- Residual analysis for linear model ---
    print("\n--- Linear model residuals by period ---")
    for year_start in range(2005, 2024, 3):
        year_end = year_start + 3
        mask = np.array([(d.year >= year_start and d.year < year_end) for d in dates])
        if mask.sum() > 0:
            mean_resid = resid_lin[mask].mean()
            print(f"  {year_start}-{year_end}: mean residual = {mean_resid:+.0f} (n={mask.sum()})")

    # --- Generate comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: All three fits
    ax = axes[0, 0]
    ax.scatter(dates, ordinals, c='#4CAF50', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    x_smooth = np.linspace(x_norm.min(), x_norm.max(), 200)
    x_smooth_dates = [mdates.num2date(x * x_std + x_mean) for x in x_smooth]
    ax.plot(x_smooth_dates, np.polyval(coeffs_lin, x_smooth), 'r--', linewidth=2, label=f'Linear (R²={r2_lin:.3f})')
    ax.plot(x_smooth_dates, np.polyval(coeffs_quad, x_smooth), 'b-', linewidth=2, label=f'Quadratic (R²={r2_quad:.3f})')
    ax.set_title('Linear vs. Quadratic Fit', fontsize=13, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Piecewise fit
    ax = axes[0, 1]
    ax.scatter(dates, ordinals, c='#4CAF50', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    # Draw two line segments
    x_line1 = np.linspace(date_nums[best_bp_mask1].min(), date_nums[best_bp_mask1].max(), 50)
    x_line2 = np.linspace(date_nums[best_bp_mask2].min(), date_nums[best_bp_mask2].max(), 50)
    ax.plot([mdates.num2date(x) for x in x_line1], np.polyval(best_bp_c1, x_line1),
            'r-', linewidth=2.5, label=f'Pre-{best_bp_date.year}: {best_bp_slopes[0]:.0f}/yr')
    ax.plot([mdates.num2date(x) for x in x_line2], np.polyval(best_bp_c2, x_line2),
            '#FF6F00', linewidth=2.5, label=f'Post-{best_bp_date.year}: {best_bp_slopes[1]:.0f}/yr')
    ax.axvline(x=best_bp_date, color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'Piecewise Linear (breakpoint: {best_bp_date}) (R²={best_bp_r2:.3f})', fontsize=13, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 3: Residuals from linear fit
    ax = axes[1, 0]
    ax.scatter(dates, resid_lin, c='#2196F3', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Linear Model Residuals', fontsize=13, fontweight='bold')
    ax.set_xlabel('Installation Date')
    ax.set_ylabel('Residual (actual - predicted)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Plot 4: Residuals from quadratic fit
    ax = axes[1, 1]
    ax.scatter(dates, resid_quad, c='#9C27B0', s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Quadratic Model Residuals', fontsize=13, fontweight='bold')
    ax.set_xlabel('Installation Date')
    ax.set_ylabel('Residual (actual - predicted)')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle('Hamilton Microlab STAR — Is Production Linear or Accelerating?',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(BASE, "curve_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {path}")

    return {
        "linear_r2": r2_lin,
        "quad_r2": r2_quad,
        "piecewise_r2": best_bp_r2,
        "linear_aic": aic_lin,
        "quad_aic": aic_quad,
        "piecewise_aic": best_bp_aic,
        "linear_slope": slope_lin,
        "quad_coeff": coeffs_quad[0],
        "breakpoint_date": best_bp_date,
        "pre_slope": best_bp_slopes[0],
        "post_slope": best_bp_slopes[1],
        "winner": models[0][0],
    }


if __name__ == "__main__":
    dates, ordinals, labels = load_data()
    print(f"Analyzing {len(dates)} data points (letter-prefix + numeric serials)\n")
    results = fit_and_compare(dates, ordinals, labels)
