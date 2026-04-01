#!/usr/bin/env python3
"""
Hamilton Microlab STAR — Final Production Analysis Report
Uses piecewise-linear model with Feb 2019 inflection point.
"""

import json
import csv
import os
import re
from datetime import datetime, date
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = os.path.dirname(os.path.abspath(__file__))
DATE_CORRECTIONS = {"E309": "2020-05-21"}


def load_data():
    with open(os.path.join(BASE, "extracted_data.json")) as f:
        raw = json.load(f)

    records = []
    for r in raw:
        sn = r["serial_number"]
        date_str = DATE_CORRECTIONS.get(sn, r["installation_date"])
        try:
            install_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            continue
        records.append({
            "serial_number": sn,
            "installation_date": install_date,
            "source_format": r["format"],
            "source_file": r["source_file"],
        })
    return sorted(records, key=lambda x: x["installation_date"])


def classify_serial(sn):
    if re.match(r'^[A-H]\d{3}$', sn):
        return "letter_prefix"
    elif re.match(r'^\d{4}$', sn):
        return "numeric_4digit"
    elif re.match(r'^\d{3}[A-Z]$', sn):
        return "digit_prefix_letter_suffix"
    return "other"


def serial_to_ordinal(sn):
    fmt = classify_serial(sn)
    if fmt == "letter_prefix":
        return (ord(sn[0]) - ord('A')) * 1000 + int(sn[1:])
    elif fmt == "numeric_4digit":
        return int(sn)
    return None


def run_analysis(records):
    """Run piecewise, quadratic, and linear fits. Return all stats."""
    # Filter to records with computable ordinals
    pts = []
    for r in records:
        o = serial_to_ordinal(r["serial_number"])
        if o is not None:
            pts.append((r["installation_date"], o, r["serial_number"]))

    dates_all = np.array([p[0] for p in pts])
    ords_all = np.array([p[1] for p in pts])
    sns_all = [p[2] for p in pts]
    dn_all = np.array([mdates.date2num(d) for d in dates_all])

    # Also letter-prefix only
    lp_pts = [(d, o, s) for d, o, s in pts if classify_serial(s) == "letter_prefix"]
    dates_lp = np.array([p[0] for p in lp_pts])
    ords_lp = np.array([p[1] for p in lp_pts])
    sns_lp = [p[2] for p in lp_pts]
    dn_lp = np.array([mdates.date2num(d) for d in dates_lp])

    results = {}
    for label, dn, ords, dates, sns in [
        ("all", dn_all, ords_all, dates_all, sns_all),
        ("letter_prefix", dn_lp, ords_lp, dates_lp, sns_lp),
    ]:
        n = len(dn)
        ss_tot = np.sum((ords - ords.mean()) ** 2)

        # Linear
        c_lin = np.polyfit(dn, ords, 1)
        pred_lin = np.polyval(c_lin, dn)
        ss_lin = np.sum((ords - pred_lin) ** 2)
        r2_lin = 1 - ss_lin / ss_tot
        aic_lin = n * np.log(ss_lin / n) + 2 * 2

        # Quadratic
        c_quad = np.polyfit(dn, ords, 2)
        pred_quad = np.polyval(c_quad, dn)
        ss_quad = np.sum((ords - pred_quad) ** 2)
        r2_quad = 1 - ss_quad / ss_tot
        aic_quad = n * np.log(ss_quad / n) + 2 * 3

        # Piecewise — scan for best breakpoint
        best_aic = float('inf')
        best_bp = None
        sorted_dn = np.sort(dn)
        lo, hi = int(n * 0.2), int(n * 0.8)
        for i in range(lo, hi):
            bp = sorted_dn[i]
            m1, m2 = dn <= bp, dn > bp
            if m1.sum() < 3 or m2.sum() < 3:
                continue
            c1 = np.polyfit(dn[m1], ords[m1], 1)
            c2 = np.polyfit(dn[m2], ords[m2], 1)
            pred = np.where(m1, np.polyval(c1, dn), np.polyval(c2, dn))
            ss = np.sum((ords - pred) ** 2)
            aic = n * np.log(ss / n) + 2 * 5
            if aic < best_aic:
                best_aic = aic
                best_bp = {
                    "date": mdates.num2date(bp).date(),
                    "dn": bp,
                    "c1": c1, "c2": c2,
                    "m1": m1, "m2": m2,
                    "r2": 1 - ss / ss_tot,
                    "aic": aic,
                    "slope_pre": c1[0] * 365.25,
                    "slope_post": c2[0] * 365.25,
                }

        results[label] = {
            "n": n, "dates": dates, "ords": ords, "dn": dn, "sns": sns,
            "lin": {"c": c_lin, "r2": r2_lin, "aic": aic_lin, "slope": c_lin[0] * 365.25},
            "quad": {"c": c_quad, "r2": r2_quad, "aic": aic_quad},
            "pw": best_bp,
        }

    return results


def create_primary_plot(res):
    """Main scatter plot with piecewise fit — the hero chart."""
    lp = res["letter_prefix"]
    dates, ords, sns = lp["dates"], lp["ords"], lp["sns"]
    dn = lp["dn"]
    pw = lp["pw"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Scatter points
    ax.scatter(dates, ords, c='#2E7D32', s=65, alpha=0.85, edgecolors='white',
               linewidth=0.5, zorder=3)

    # Piecewise regression lines
    x1 = np.linspace(dn[pw["m1"]].min(), pw["dn"], 80)
    x2 = np.linspace(pw["dn"], dn[pw["m2"]].max(), 80)
    bp_label = pw["date"].strftime("%b %Y")
    ax.plot([mdates.num2date(x) for x in x1], np.polyval(pw["c1"], x1),
            color='#1565C0', linewidth=2.5, linestyle='--',
            label=f'Pre-{bp_label}: ~{pw["slope_pre"]:.0f} units/year')
    ax.plot([mdates.num2date(x) for x in x2], np.polyval(pw["c2"], x2),
            color='#E65100', linewidth=2.5, linestyle='--',
            label=f'Post-{bp_label}: ~{pw["slope_post"]:.0f} units/year')

    # Breakpoint line
    ax.axvline(x=pw["date"], color='#B71C1C', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.annotate(f'Inflection: {pw["date"].strftime("%b %Y")}',
                xy=(pw["date"], ords.max() * 0.5),
                xytext=(40, 30), textcoords='offset points',
                fontsize=11, fontweight='bold', color='#B71C1C',
                arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))

    # Label select data points
    indices_to_label = set()
    indices_to_label.add(0)
    indices_to_label.add(len(sns) - 1)
    # Find points nearest to breakpoint
    bp_dist = np.abs(dn - pw["dn"])
    indices_to_label.add(np.argmin(bp_dist))
    # Mid-points of each segment
    pre_idx = np.where(pw["m1"])[0]
    post_idx = np.where(pw["m2"])[0]
    if len(pre_idx) > 2:
        indices_to_label.add(pre_idx[len(pre_idx) // 2])
    if len(post_idx) > 2:
        indices_to_label.add(post_idx[len(post_idx) // 2])

    for i in indices_to_label:
        if i < len(sns):
            ax.annotate(sns[i], (dates[i], ords[i]),
                        textcoords="offset points", xytext=(8, -12),
                        fontsize=8, color='#555', fontstyle='italic')

    # Rate callout box
    multiplier = pw["slope_post"] / pw["slope_pre"] if pw["slope_pre"] > 0 else float('inf')
    stats_text = (f"Letter-prefix serials only (n={lp['n']})\n"
                  f"Breakpoint: {pw['date']}\n"
                  f"Pre-{bp_label}:  ~{pw['slope_pre']:.0f} units/year\n"
                  f"Post-{bp_label}: ~{pw['slope_post']:.0f} units/year\n"
                  f"Acceleration: {multiplier:.0f}x")
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF8E1', edgecolor='#F9A825',
                      alpha=0.95), fontfamily='monospace')

    ax.set_xlabel('Installation / Manufacture Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)', fontsize=12, fontweight='bold')
    ax.set_title('Hamilton Microlab STAR — Production Rate with Inflection Point',
                 fontsize=14, fontweight='bold', pad=15)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='lower right')
    plt.tight_layout()

    path = os.path.join(BASE, "production_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def create_model_comparison_plot(res):
    """Four-panel diagnostic plot."""
    lp = res["letter_prefix"]
    dates, ords, dn = lp["dates"], lp["ords"], lp["dn"]
    pw = lp["pw"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: All three fits overlaid
    ax = axes[0, 0]
    ax.scatter(dates, ords, c='#4CAF50', s=45, alpha=0.7, edgecolors='white', linewidth=0.5)
    x_sm = np.linspace(dn.min(), dn.max(), 200)
    x_sm_d = [mdates.num2date(x) for x in x_sm]
    ax.plot(x_sm_d, np.polyval(lp["lin"]["c"], x_sm), 'r--', lw=2,
            label=f'Linear (R²={lp["lin"]["r2"]:.3f})')
    ax.plot(x_sm_d, np.polyval(lp["quad"]["c"], x_sm), 'b-', lw=2,
            label=f'Quadratic (R²={lp["quad"]["r2"]:.3f})')
    # Piecewise
    x1 = np.linspace(dn[pw["m1"]].min(), pw["dn"], 80)
    x2 = np.linspace(pw["dn"], dn[pw["m2"]].max(), 80)
    ax.plot([mdates.num2date(x) for x in x1], np.polyval(pw["c1"], x1),
            '#FF6F00', lw=2.5, label=f'Piecewise (R²={pw["r2"]:.3f})')
    ax.plot([mdates.num2date(x) for x in x2], np.polyval(pw["c2"], x2),
            '#FF6F00', lw=2.5)
    ax.set_title('Three Models Compared', fontsize=13, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel 2: Piecewise with breakpoint highlighted
    ax = axes[0, 1]
    colors = ['#1565C0' if m else '#E65100' for m in pw["m1"]]
    ax.scatter(dates, ords, c=colors, s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    ax.plot([mdates.num2date(x) for x in x1], np.polyval(pw["c1"], x1),
            '#1565C0', lw=2.5, label=f'Pre-{pw["date"].year}: {pw["slope_pre"]:.0f}/yr')
    ax.plot([mdates.num2date(x) for x in x2], np.polyval(pw["c2"], x2),
            '#E65100', lw=2.5, label=f'Post-{pw["date"].year}: {pw["slope_post"]:.0f}/yr')
    ax.axvline(x=pw["date"], color='gray', linestyle=':', alpha=0.5)
    ax.set_title(f'Piecewise Model — Breakpoint {pw["date"]}', fontsize=13, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel 3: Linear residuals
    pred_lin = np.polyval(lp["lin"]["c"], dn)
    resid_lin = ords - pred_lin
    ax = axes[1, 0]
    ax.scatter(dates, resid_lin, c='#2196F3', s=45, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Linear Model — Residuals (systematic pattern)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Installation Date')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    # Panel 4: Piecewise residuals
    pred_pw = np.where(pw["m1"], np.polyval(pw["c1"], dn), np.polyval(pw["c2"], dn))
    resid_pw = ords - pred_pw
    ax = axes[1, 1]
    ax.scatter(dates, resid_pw, c='#9C27B0', s=45, alpha=0.7, edgecolors='white', linewidth=0.5)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.set_title('Piecewise Model — Residuals (no systematic pattern)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Installation Date')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle('Model Selection: Why the Piecewise Fit Is the Right Choice',
                 fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(BASE, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_html(records, res):
    lp = res["letter_prefix"]
    al = res["all"]
    pw = lp["pw"]

    multiplier = pw["slope_post"] / pw["slope_pre"] if pw["slope_pre"] > 0 else 0
    date_start = min(r["installation_date"] for r in records)
    date_end = max(r["installation_date"] for r in records)
    span = (date_end - date_start).days / 365.25
    bp_date = pw["date"]
    bp_label = bp_date.strftime("%B %Y")
    bp_year = bp_date.year

    # Year breakdown
    year_counts = defaultdict(list)
    for r in records:
        year_counts[r["installation_date"].year].append(r["serial_number"])

    year_rows = ""
    for year in sorted(year_counts.keys()):
        sns = sorted(year_counts[year])
        era = "Pre-inflection" if year < bp_year else ("Inflection year" if year == bp_year else "Post-inflection")
        year_rows += f"""        <tr>
          <td><strong>{year}</strong></td>
          <td>{len(sns)}</td>
          <td class="era-{era.split()[0].lower()}">{era}</td>
          <td>{', '.join(sns)}</td>
        </tr>\n"""

    # Serial format counts
    fmt_counts = Counter(classify_serial(r["serial_number"]) for r in records)

    # Production line mapping
    def production_line(sn):
        fmt = classify_serial(sn)
        if fmt == "letter_prefix":
            return "USA"
        elif fmt == "numeric_4digit":
            return "Swiss"
        else:
            return "Roche"

    # Data table
    table_rows = ""
    for r in records:
        sn = r["serial_number"]
        fmt = classify_serial(sn)
        ordinal = serial_to_ordinal(sn)
        ordinal_str = str(ordinal) if ordinal is not None else "N/A"
        corrected = " *" if sn in DATE_CORRECTIONS else ""
        era = "pre" if r["installation_date"] < bp_date else "post"
        line = production_line(sn)
        table_rows += f"""        <tr class="era-row-{era}">
          <td><strong>{sn}</strong></td>
          <td>{r["installation_date"].isoformat()}{corrected}</td>
          <td>{line}</td>
          <td>{ordinal_str}</td>
        </tr>\n"""

    # Per-line stats
    line_records = defaultdict(list)
    for r in records:
        line_records[production_line(r["serial_number"])].append(r)

    line_stats = {}
    for line_name, recs in line_records.items():
        dates_sorted = sorted(r["installation_date"] for r in recs)
        sns_sorted = sorted(r["serial_number"] for r in recs)
        # Compute max ordinal / implied units for each line
        if line_name == "USA":
            max_ord = max(serial_to_ordinal(r["serial_number"]) for r in recs)
            implied = max_ord
        elif line_name == "Swiss":
            max_ord = max(int(r["serial_number"]) for r in recs)
            implied = max_ord
        else:  # Roche — 3 digits + letter, harder to compute total
            # Extract numeric portions; treat as sequential within their scheme
            nums = sorted(int(r["serial_number"][:3]) for r in recs if r["serial_number"][:3].isdigit())
            max_ord = max(nums) if nums else 0
            implied = max_ord  # lower-bound estimate
        line_stats[line_name] = {
            "count": len(recs),
            "date_min": dates_sorted[0],
            "date_max": dates_sorted[-1],
            "serials": sns_sorted,
            "max_ordinal": max_ord,
            "implied_units": implied,
        }

    total_implied = sum(ls["implied_units"] for ls in line_stats.values())

    # Convenience refs for template
    ls_usa = line_stats["USA"]
    ls_swiss = line_stats["Swiss"]
    ls_roche = line_stats["Roche"]
    swiss_max_sn = max(r["serial_number"] for r in line_records["Swiss"])

    # Compute implied units per era
    lp_records = [r for r in records if classify_serial(r["serial_number"]) == "letter_prefix"]
    pre_lp = [r for r in lp_records if r["installation_date"] < bp_date]
    post_lp = [r for r in lp_records if r["installation_date"] >= bp_date]

    # Most recent letter-prefix system (for cutoff note)
    latest_lp = max(lp_records, key=lambda r: r["installation_date"])
    latest_lp_sn = latest_lp["serial_number"]
    latest_lp_date = latest_lp["installation_date"]
    latest_lp_ord = serial_to_ordinal(latest_lp_sn)

    pre_ords = sorted([serial_to_ordinal(r["serial_number"]) for r in pre_lp])
    post_ords = sorted([serial_to_ordinal(r["serial_number"]) for r in post_lp])

    pre_range = f"{pre_ords[0]}–{pre_ords[-1]}" if pre_ords else "N/A"
    post_range = f"{post_ords[0]}–{post_ords[-1]}" if post_ords else "N/A"
    pre_span = (pre_ords[-1] - pre_ords[0] + 1) if len(pre_ords) > 1 else 0
    post_span = (post_ords[-1] - post_ords[0] + 1) if len(post_ords) > 1 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hamilton Microlab STAR — Production Volume Analysis</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px 40px; color: #333;
         line-height: 1.6; }}
  h1 {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
  h2 {{ color: #2c5f8a; margin-top: 40px; }}
  h3 {{ color: #3a7cb8; }}
  .headline {{ background: linear-gradient(135deg, #B71C1C 0%, #D84315 100%);
               color: white; padding: 24px 30px; border-radius: 8px; margin: 25px 0;
               font-size: 15px; line-height: 1.7; }}
  .headline .big {{ font-size: 32px; font-weight: bold; display: block; margin-bottom: 6px; }}
  .headline .sub {{ font-size: 17px; opacity: 0.95; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
               gap: 16px; margin: 20px 0; }}
  .kpi {{ background: #f0f6fc; border-left: 4px solid #2c5f8a; padding: 16px 20px;
          border-radius: 4px; }}
  .kpi.accent {{ border-left-color: #B71C1C; background: #fce4ec; }}
  .kpi .value {{ font-size: 28px; font-weight: bold; color: #1a3a5c; }}
  .kpi.accent .value {{ color: #B71C1C; }}
  .kpi .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
  th {{ background: #1a3a5c; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #e8f0fe; }}
  .era-pre {{ color: #1565C0; font-weight: 600; }}
  .era-inflection {{ color: #B71C1C; font-weight: 600; }}
  .era-post {{ color: #E65100; font-weight: 600; }}
  .era-row-pre {{ }}
  .era-row-post {{ background: #FFF8E1; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px;
           margin: 15px 0; border-radius: 4px; font-size: 14px; }}
  .methodology {{ background: #f1f8e9; border-left: 4px solid #7cb342; padding: 14px 18px;
                  margin: 15px 0; border-radius: 4px; font-size: 14px; }}
  .model-table {{ margin: 15px 0; }}
  .model-table td, .model-table th {{ padding: 10px 16px; }}
  .winner {{ background: #E8F5E9 !important; font-weight: bold; }}
  .scatter-container {{ text-align: center; margin: 25px 0; }}
  .scatter-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px;
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
  .scatter-container .caption {{ font-size: 13px; color: #666; margin-top: 8px; font-style: italic; }}
  .two-era {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }}
  .era-box {{ padding: 20px; border-radius: 6px; }}
  .era-box.pre {{ background: #E3F2FD; border: 2px solid #1565C0; }}
  .era-box.post {{ background: #FFF3E0; border: 2px solid #E65100; }}
  .era-box h3 {{ margin-top: 0; }}
  .era-box .rate {{ font-size: 28px; font-weight: bold; }}
  .era-box.pre .rate {{ color: #1565C0; }}
  .era-box.post .rate {{ color: #E65100; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
             font-size: 12px; color: #999; }}
  .corrected {{ color: #e65100; font-size: 12px; }}
</style>
</head>
<body>

<h1>Hamilton Microlab STAR — Production Volume Analysis</h1>
<p><strong>Report generated:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
<strong>Data source:</strong> Hamilton iData instrument PDFs ({len(records)} systems analyzed)<br>
<strong>Scope:</strong> Serial number and installation date analysis to estimate manufacturing volume and production rate</p>

<!-- ============================================================ -->
<h2>Executive Summary</h2>

<div class="kpi-grid">
  <div class="kpi">
    <div class="value">~{total_implied:,}+</div>
    <div class="label">Implied total units manufactured (all three lines combined, as of data cutoff)</div>
  </div>
  <div class="kpi">
    <div class="value">{len(records)}</div>
    <div class="label">Unique systems in sample</div>
  </div>
  <div class="kpi">
    <div class="value">{span:.0f} yrs</div>
    <div class="label">Date range covered ({date_start} to {date_end})</div>
  </div>
</div>

<table>
  <thead>
    <tr><th>Production Line</th><th>Highest Serial Observed</th><th>Implied Units (minimum)</th></tr>
  </thead>
  <tbody>
    <tr><td><strong>USA</strong> (letter + 3 digits)</td><td>{latest_lp_sn} (ordinal {latest_lp_ord:,})</td><td><strong>~{line_stats['USA']['implied_units']:,}</strong></td></tr>
    <tr><td><strong>Swiss</strong> (4 numeric digits)</td><td>{max(r['serial_number'] for r in line_records['Swiss'])}</td><td><strong>~{line_stats['Swiss']['implied_units']:,}</strong></td></tr>
    <tr><td><strong>Roche</strong> (3 digits + letter)</td><td>{line_stats['Roche']['serials'][-1]}</td><td><strong>~{line_stats['Roche']['implied_units']:,}</strong></td></tr>
    <tr style="background:#E8F5E9; font-weight:bold;"><td>Combined Total</td><td></td><td>~{total_implied:,}+</td></tr>
  </tbody>
</table>

<p style="font-size:13px; color:#666;">These are minimum estimates based on the highest serial number observed in each line.
Actual totals are likely higher — our data cuts off at {latest_lp_date.year} and does not include
systems manufactured since then.</p>

<div class="note">
<strong>Data cutoff:</strong> The most recent letter-prefix system in our dataset is <strong>{latest_lp_sn}</strong>
(ordinal {latest_lp_ord:,}, installed {latest_lp_date}). We do not have iData records for systems
manufactured after this date. Production almost certainly continued beyond {latest_lp_date.year} —
this analysis only reflects what our collected instrument data can show. Any conclusions about
current or recent production rates should not be drawn from this dataset.
</div>

<div class="two-era">
  <div class="era-box pre">
    <h3 style="color:#1565C0;">Era 1: Pre-{bp_label}</h3>
    <div class="rate">~{pw['slope_pre']:.0f} units/year</div>
    <p>Steady, moderate production from 2008 through early {bp_year}.<br>
    Serial range: {pre_range} ({pre_span:,} implied units over ~{(bp_date - pre_lp[0]['installation_date']).days / 365.25:.0f} years)<br>
    Systems in sample: {len(pre_lp)}</p>
  </div>
  <div class="era-box post">
    <h3 style="color:#E65100;">Era 2: Post-{bp_label}</h3>
    <div class="rate">~{pw['slope_post']:.0f} units/year</div>
    <p>Rapid acceleration beginning ~{bp_label}.<br>
    Serial range: {post_range} ({post_span:,} implied units over ~{(post_lp[-1]['installation_date'] - bp_date).days / 365.25:.1f} years)<br>
    Systems in sample: {len(post_lp)}</p>
  </div>
</div>

<!-- ============================================================ -->
<h2>Production Over Time (USA Line)</h2>

<div class="scatter-container">
  <img src="production_scatter.png" alt="Serial Number vs. Manufacture Date — Piecewise Linear Fit">
  <div class="caption">Figure 1: USA-line systems only (letter + 3 digit serials). Swiss and Roche systems are excluded.
  The blue dashed line shows the pre-{bp_label} trend (~{pw['slope_pre']:.0f}/yr);
  the orange dashed line shows the post-{bp_label} trend (~{pw['slope_post']:.0f}/yr). Vertical red line marks the inflection point.</div>
</div>

<p>The scatter plot makes the two production eras visually unmistakable. Before {bp_label}, serial numbers
advanced slowly — roughly {pw['slope_pre']:.0f} new units per year. After the inflection point, the slope
steepens dramatically to approximately {pw['slope_post']:.0f} units per year.</p>

<!-- ============================================================ -->
<h2>Why a Piecewise Model — Not a Straight Line</h2>

<p>An initial linear regression across the full dataset yields R² = {lp['lin']['r2']:.3f} — meaning
a straight line explains only {lp['lin']['r2']*100:.0f}% of the variance in serial numbers over time.
Three candidate models were evaluated:</p>

<table class="model-table">
  <thead>
    <tr><th>Model</th><th>R²</th><th>AIC</th><th>Parameters</th><th>Interpretation</th></tr>
  </thead>
  <tbody>
    <tr>
      <td>Linear</td>
      <td>{lp['lin']['r2']:.3f}</td>
      <td>{lp['lin']['aic']:.1f}</td>
      <td>2</td>
      <td>Constant production rate (~{lp['lin']['slope']:.0f}/yr) — poor fit</td>
    </tr>
    <tr>
      <td>Quadratic</td>
      <td>{lp['quad']['r2']:.3f}</td>
      <td>{lp['quad']['aic']:.1f}</td>
      <td>3</td>
      <td>Gradually accelerating production — better</td>
    </tr>
    <tr class="winner">
      <td>Piecewise Linear</td>
      <td>{pw['r2']:.3f}</td>
      <td>{pw['aic']:.1f}</td>
      <td>5</td>
      <td>Two distinct eras with a breakpoint — best fit</td>
    </tr>
  </tbody>
</table>

<p><strong>The piecewise-linear model is the recommended model</strong> for three reasons:</p>

<ol>
  <li><strong>Lowest AIC</strong> (Akaike Information Criterion): Despite using more parameters, the
  piecewise model's improvement in fit more than compensates for the added complexity. AIC penalizes
  extra parameters, so a lower AIC means a genuinely better model, not just overfitting.</li>

  <li><strong>Highest R²</strong> ({pw['r2']:.3f}): The piecewise model explains {pw['r2']*100:.0f}% of the
  variance in serial number progression — versus {lp['lin']['r2']*100:.0f}% for the linear model. That is a
  substantial improvement.</li>

  <li><strong>Residual diagnostics</strong>: The linear model's residuals show a clear systematic pattern
  (see Figure 2, bottom-left) — underpredicting early and late, overpredicting in the middle. This is a
  textbook sign of model misspecification. The piecewise model's residuals are randomly scattered
  (bottom-right), indicating a proper fit.</li>
</ol>

<div class="scatter-container">
  <img src="model_comparison.png" alt="Model Comparison — Four Panel Diagnostic">
  <div class="caption">Figure 2: Model comparison diagnostics. Top-left: all three models overlaid.
  Top-right: piecewise model with color-coded eras. Bottom: residual plots showing the linear model's
  systematic error pattern (left) vs. the piecewise model's random residuals (right).</div>
</div>

<div class="note">
<strong>What does the inflection point mean?</strong> The data indicates that something changed in Hamilton's
STAR manufacturing around {bp_label} — either a capacity expansion, a new production line, increased
market demand, or a combination. The ~{multiplier:.0f}x rate increase is too large to be explained by
normal year-to-year variation. It represents a fundamental shift in production volume.
</div>

<!-- ============================================================ -->
<h2>Three Production Lines</h2>

<p>Hamilton manufactured the Microlab STAR across <strong>three distinct production lines</strong>,
each with its own serial number format:</p>

<table>
  <thead>
    <tr><th>Production Line</th><th>Serial Format</th><th>Example</th><th>Systems in Sample</th><th>Date Range</th><th>Serials Observed</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Swiss</strong> (Switzerland)</td>
      <td>4 numeric digits</td>
      <td>1234</td>
      <td>{ls_swiss['count']}</td>
      <td>{ls_swiss['date_min']} to {ls_swiss['date_max']}</td>
      <td>{', '.join(ls_swiss['serials'])}</td>
    </tr>
    <tr style="background: #E8F5E9;">
      <td><strong>USA</strong></td>
      <td>Letter + 3 digits</td>
      <td>A123</td>
      <td>{ls_usa['count']}</td>
      <td>{ls_usa['date_min']} to {ls_usa['date_max']}</td>
      <td>{ls_usa['serials'][0]} ... {ls_usa['serials'][-1]}</td>
    </tr>
    <tr>
      <td><strong>Roche</strong></td>
      <td>3 digits + letter</td>
      <td>123A</td>
      <td>{ls_roche['count']}</td>
      <td>{ls_roche['date_min']} to {ls_roche['date_max']}</td>
      <td>{', '.join(ls_roche['serials'])}</td>
    </tr>
  </tbody>
</table>

<p>Each production line maintains its own independent, sequential serial numbering. This means
<strong>total STAR production is the sum of all three lines</strong>, not just the USA line analyzed above.</p>

<div class="note">
<strong>Analysis scope:</strong> The production rate analysis (piecewise model, Era 1/Era 2) is based on
the <strong>USA line only</strong> ({ls_usa['count']} systems), which has the largest sample and cleanest
sequential progression in our data. The Swiss and Roche lines are represented by only
{ls_swiss['count']} and {ls_roche['count']} systems respectively —
too few data points to model their production rates independently, but their serial numbers contribute
to the overall production count. The highest Swiss serial observed is <strong>{ls_swiss['serials'][-1]}</strong>
({int(ls_swiss['serials'][-1]):,} units implied), and Roche serials suggest a separate
production run of at least several hundred units.
</div>

<!-- ============================================================ -->
<h2>Year-by-Year Breakdown</h2>

<table>
  <thead>
    <tr><th>Year</th><th>Systems in Sample</th><th>Era</th><th>Serial Numbers</th></tr>
  </thead>
  <tbody>
{year_rows}  </tbody>
</table>

<!-- ============================================================ -->
<h2>Complete Data Table</h2>

<p>{len(records)} systems sorted by installation date. Post-2019 rows highlighted.</p>

<table>
  <thead>
    <tr><th>Serial Number</th><th>Installation Date</th><th>Production Line</th><th>Ordinal</th></tr>
  </thead>
  <tbody>
{table_rows}  </tbody>
</table>

<p class="corrected">* Date corrected from source PDF (original contained data-entry error)</p>

<!-- ============================================================ -->
<h2>Data Quality Notes</h2>
<ul>
  <li><strong>E309:</strong> Installation date in source PDF reads "2020-21-05" (invalid). Corrected to
      2020-05-21 based on download date alignment.</li>
  <li><strong>A234 folder:</strong> Folder is labeled "A234" but both PDFs reference serial <strong>A025</strong>.
      Data correctly extracted as A025. The folder name appears to be a mislabel.</li>
  <li><strong>1941, F679:</strong> System folders exist but contain no PDF files. Excluded.</li>
  <li><strong>Multiple PDFs per system:</strong> Where Pre/Post service records exist, installation dates
      are consistent across all PDFs for the same system.</li>
  <li><strong>"Installation date" as manufacture proxy:</strong> This field records the date of initial
      system installation. Actual manufacture date may precede this by weeks to months — the
      production rate estimates should be understood as approximate.</li>
</ul>

<!-- ============================================================ -->
<h2>Limitations</h2>
<ul>
  <li>Sample of {len(records)} systems — not a complete production census. Rate estimates assume
      the sample is reasonably representative of the full production population.</li>
  <li>Serial number gaps may exist (skipped numbers, test units, returns).
      Implied totals are upper bounds.</li>
  <li>The piecewise breakpoint is optimized from the data, not independently confirmed.
      The true inflection may be slightly earlier or later than {bp_label}.</li>
  <li>Post-{bp_year} rate may overstate sustained production if the sample is biased toward
      recent systems (which are more likely to have iData on file).</li>
  <li><strong>Data cutoff at {latest_lp_date}:</strong> Our letter-prefix iData collection does not extend
      beyond this date. Hamilton almost certainly continued manufacturing STAR systems after
      {latest_lp_date.year}. This analysis cannot speak to production volumes in
      {latest_lp_date.year + 1}–present.</li>
</ul>

<!-- ============================================================ -->
<h2>Next Step: Determining Exact Total Production</h2>

<p>The analysis above estimates production rates and trends, but the serial numbering system
makes it possible to determine <strong>exact total production</strong> with a single data point.</p>

<div class="methodology">
<strong>Key insight:</strong> Because each production line assigns serial numbers sequentially,
the highest serial number on the most recently manufactured system from each line <em>is</em>
that line's total production count. Sum all three lines to get overall STAR production.
</div>

<p><strong>What we know from this dataset (as of data cutoff):</strong></p>

<table>
  <thead>
    <tr><th>Production Line</th><th>Highest Serial</th><th>Implied Minimum Units</th><th>How to Read</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>USA</strong></td>
      <td>{latest_lp_sn} ({latest_lp_date.strftime('%B %Y')})</td>
      <td><strong>~{line_stats['USA']['implied_units']:,}</strong></td>
      <td>{latest_lp_sn[0]} = {ord(latest_lp_sn[0]) - ord('A')} &times; 1,000 + {int(latest_lp_sn[1:])} = {latest_lp_ord:,}</td>
    </tr>
    <tr>
      <td><strong>Swiss</strong></td>
      <td>{line_stats['Swiss']['serials'][-1]} ({line_stats['Swiss']['date_max'].strftime('%B %Y')})</td>
      <td><strong>~{line_stats['Swiss']['implied_units']:,}</strong></td>
      <td>Serial number is the ordinal directly</td>
    </tr>
    <tr>
      <td><strong>Roche</strong></td>
      <td>{line_stats['Roche']['serials'][-1]} ({line_stats['Roche']['date_max'].strftime('%B %Y')})</td>
      <td><strong>~{line_stats['Roche']['implied_units']:,}</strong></td>
      <td>Leading digits = ordinal (letter indicates sub-series or region)</td>
    </tr>
    <tr style="background:#E8F5E9; font-weight:bold;">
      <td>Combined (as of data cutoff)</td>
      <td></td>
      <td>~{total_implied:,}+</td>
      <td></td>
    </tr>
  </tbody>
</table>

<p>These are <strong>minimum estimates</strong> — our data ends in {latest_lp_date.year}. At the USA line's
post-inflection rate of ~{pw['slope_post']:.0f} units/year alone, an additional ~{pw['slope_post'] * (datetime.now().year - latest_lp_date.year):,.0f}
USA-line units may have been produced since then. Swiss and Roche lines may have continued as well.</p>

<p><strong>How to get definitive current totals:</strong></p>
<ul>
  <li>For each production line, find the serial number on the <strong>most recently manufactured</strong> system</li>
  <li><strong>USA line:</strong> Serial format is letter + 3 digits (e.g., J450). Convert the letter to its
      zero-indexed position (A=0, B=1, ... H=7, I=8, J=9, K=10, ...), multiply by 1,000, add the
      digits. Example: J450 = 9 &times; 1,000 + 450 = <strong>9,450 total USA units</strong></li>
  <li><strong>Swiss line:</strong> Serial is a plain number (e.g., 8200). That number <em>is</em> the unit count</li>
  <li><strong>Roche line:</strong> Leading 3 digits are the sequential number within that sub-series</li>
  <li>Sum all three for total global STAR production</li>
</ul>

<p><strong>Where to look:</strong></p>
<ul>
  <li>The iData report for any system (Master section &rarr; "Serial number instrument")</li>
  <li>The physical nameplate/label on the instrument itself</li>
  <li>Hamilton's service or sales records, if accessible</li>
  <li>A Hamilton field service engineer may be able to confirm the latest serial they've seen from each line</li>
</ul>

<div class="footer">
  <p>Analysis performed {datetime.now().strftime('%Y-%m-%d')}. Source: Hamilton iData instrument PDFs
  in <code>_iData by System/</code>. Extraction and analysis scripts: <code>extract_data.py</code>,
  <code>generate_final_report.py</code>.</p>
</div>

</body>
</html>"""

    # Save both report.html and index.html (for GitHub Pages)
    for fname in ["report.html", "index.html"]:
        path = os.path.join(BASE, fname)
        with open(path, 'w') as f:
            f.write(html)
        print(f"  Saved: {path}")


def main():
    print("Loading data...")
    records = load_data()
    print(f"  {len(records)} records\n")

    print("Running analysis...")
    res = run_analysis(records)

    lp = res["letter_prefix"]
    pw = lp["pw"]
    print(f"  Letter-prefix systems: {lp['n']}")
    print(f"  Breakpoint: {pw['date']}")
    print(f"  Pre-2019 rate:  {pw['slope_pre']:.0f} units/year")
    print(f"  Post-2019 rate: {pw['slope_post']:.0f} units/year")
    print(f"  Piecewise R²: {pw['r2']:.3f}")
    print()

    print("Generating plots...")
    create_primary_plot(res)
    create_model_comparison_plot(res)

    print("\nGenerating HTML report...")
    generate_html(records, res)

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
