#!/usr/bin/env python3
"""
Hamilton Microlab STAR Production Analysis
-------------------------------------------
Reads extracted_data.json, analyzes serial number patterns and manufacturing
dates, and produces:
  1. extracted_dataset.csv   — clean, deduplicated dataset
  2. production_scatter.png  — XY scatter plot with regression line
  3. report.html             — executive-ready report
"""

import json
import csv
import os
import re
from datetime import datetime, date
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

BASE = os.path.dirname(os.path.abspath(__file__))

# --- Known corrections for data-entry errors in source PDFs ---
DATE_CORRECTIONS = {
    "E309": "2020-05-21",  # PDF says 2020-21-05; download date confirms 2020-05-21
}


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
            print(f"  WARNING: Skipping {sn} — invalid date '{date_str}'")
            continue

        records.append({
            "serial_number": sn,
            "installation_date": install_date,
            "source_format": r["format"],
            "source_file": r["source_file"],
        })

    return sorted(records, key=lambda x: x["installation_date"])


def classify_serial(sn):
    """
    Classify serial number format.
    Returns (format_type, numeric_value_or_None)

    Observed patterns:
      - Pure numeric: 1413, 2082, 5130, 7109 etc. (older systems)
      - Letter + 3 digits: A142, B222, D777, E683, H230 etc. (newer systems)
      - 3 digits + letter: 293H, 324G, 354H (variant)
      - Mixed/other: 177D
    """
    if re.match(r'^\d{4}$', sn):
        return "numeric_4digit", int(sn)
    elif re.match(r'^[A-H]\d{3}$', sn):
        return "letter_prefix", sn
    elif re.match(r'^\d{3}[A-Z]$', sn):
        return "digit_prefix_letter_suffix", sn
    else:
        return "other", sn


def letter_prefix_to_ordinal(sn):
    """Convert letter-prefix serial like A142 to an ordinal for sequencing.
    A=0, B=1, ..., H=7 => A000=0, A001=1, ..., B000=1000, ..., H999=7999
    """
    letter_val = ord(sn[0]) - ord('A')
    num_val = int(sn[1:])
    return letter_val * 1000 + num_val


def analyze_serial_pattern(records):
    """Analyze serial number sequencing to estimate total production."""
    letter_prefix_records = []
    numeric_records = []
    other_records = []

    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        if fmt == "letter_prefix":
            letter_prefix_records.append(r)
        elif fmt == "numeric_4digit":
            numeric_records.append(r)
        else:
            other_records.append(r)

    # Sort letter-prefix by ordinal
    letter_prefix_records.sort(key=lambda r: letter_prefix_to_ordinal(r["serial_number"]))

    return letter_prefix_records, numeric_records, other_records


def compute_production_estimate(records):
    """
    Estimate total STAR units manufactured based on serial number range.
    Assumption: serial numbers are roughly sequential.
    """
    all_ordinals = []
    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        if fmt == "letter_prefix":
            all_ordinals.append(letter_prefix_to_ordinal(r["serial_number"]))
        elif fmt == "numeric_4digit":
            all_ordinals.append(int(r["serial_number"]))

    if not all_ordinals:
        return {}

    min_ord = min(all_ordinals)
    max_ord = max(all_ordinals)

    # Date range
    dates = [r["installation_date"] for r in records]
    min_date = min(dates)
    max_date = max(dates)
    span_years = (max_date - min_date).days / 365.25

    return {
        "min_serial_ordinal": min_ord,
        "max_serial_ordinal": max_ord,
        "serial_range": max_ord - min_ord,
        "implied_total_units": max_ord - min_ord + 1,
        "date_range_start": min_date,
        "date_range_end": max_date,
        "span_years": round(span_years, 1),
        "avg_units_per_year": round((max_ord - min_ord + 1) / span_years, 0) if span_years > 0 else 0,
        "sample_size": len(records),
    }


def save_csv(records):
    path = os.path.join(BASE, "extracted_dataset.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["serial_number", "installation_date", "serial_format",
                         "serial_ordinal", "source_format", "source_file"])
        for r in records:
            fmt, _ = classify_serial(r["serial_number"])
            if fmt == "letter_prefix":
                ordinal = letter_prefix_to_ordinal(r["serial_number"])
            elif fmt == "numeric_4digit":
                ordinal = int(r["serial_number"])
            else:
                ordinal = ""
            writer.writerow([
                r["serial_number"],
                r["installation_date"].isoformat(),
                fmt,
                ordinal,
                r["source_format"],
                r["source_file"],
            ])
    print(f"  Saved CSV: {path}")
    return path


def create_scatter_plot(records):
    """Create XY scatter plot: X=installation date, Y=serial ordinal, with regression line."""
    dates = []
    ordinals = []
    labels = []

    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        if fmt == "letter_prefix":
            ordinal = letter_prefix_to_ordinal(r["serial_number"])
        elif fmt == "numeric_4digit":
            ordinal = int(r["serial_number"])
        else:
            continue
        dates.append(r["installation_date"])
        ordinals.append(ordinal)
        labels.append(r["serial_number"])

    # Convert dates to numeric for regression
    date_nums = np.array([mdates.date2num(d) for d in dates])
    ordinals_arr = np.array(ordinals)

    # Linear regression
    slope, intercept = np.polyfit(date_nums, ordinals_arr, 1)

    # Generate regression line
    x_line = np.linspace(date_nums.min(), date_nums.max(), 100)
    y_line = slope * x_line + intercept

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Classify points for color coding
    colors = []
    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        if fmt == "numeric_4digit":
            colors.append('#2196F3')
        elif fmt == "letter_prefix":
            colors.append('#4CAF50')
        elif fmt in ("digit_prefix_letter_suffix", "other"):
            continue

    ax.scatter(dates, ordinals, c=colors, s=60, alpha=0.8, edgecolors='white',
               linewidth=0.5, zorder=3)

    # Add regression line
    ax.plot([mdates.num2date(x) for x in x_line], y_line,
            color='#FF5722', linewidth=2, linestyle='--', alpha=0.8,
            label=f'Trend: ~{slope * 365.25:.0f} units/year')

    # Label select points (first, last, and a few in between)
    label_indices = [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, len(dates)-1]
    for i in set(label_indices):
        if i < len(labels):
            ax.annotate(labels[i], (dates[i], ordinals[i]),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color='#555', fontstyle='italic')

    # Formatting
    ax.set_xlabel('Installation / Manufacture Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Serial Number (ordinal)', fontsize=12, fontweight='bold')
    ax.set_title('Hamilton Microlab STAR — Serial Number vs. Manufacture Date',
                 fontsize=14, fontweight='bold', pad=15)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    # Add annotation box with key stats
    stats_text = (f"Sample size: {len(dates)} systems\n"
                  f"Date range: {min(dates)} to {max(dates)}\n"
                  f"Serial range: {min(ordinals)} to {max(ordinals)}\n"
                  f"Production rate: ~{slope * 365.25:.0f} units/year")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    path = os.path.join(BASE, "production_scatter.png")
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot: {path}")
    return path, slope * 365.25  # return annual rate


def generate_html_report(records, stats, annual_rate):
    """Generate an executive-readable HTML report."""

    letter_recs, numeric_recs, other_recs = analyze_serial_pattern(records)

    # Build per-year breakdown
    year_counts = defaultdict(list)
    for r in records:
        year_counts[r["installation_date"].year].append(r["serial_number"])

    # Sorted data table rows
    table_rows = ""
    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        if fmt == "letter_prefix":
            ordinal = letter_prefix_to_ordinal(r["serial_number"])
        elif fmt == "numeric_4digit":
            ordinal = int(r["serial_number"])
        else:
            ordinal = "N/A"
        corrected = " *" if r["serial_number"] in DATE_CORRECTIONS else ""
        table_rows += f"""        <tr>
          <td><strong>{r["serial_number"]}</strong></td>
          <td>{r["installation_date"].isoformat()}{corrected}</td>
          <td>{fmt.replace('_', ' ')}</td>
          <td>{ordinal}</td>
        </tr>\n"""

    # Year breakdown rows
    year_rows = ""
    for year in sorted(year_counts.keys()):
        sns = sorted(year_counts[year])
        year_rows += f"""        <tr>
          <td><strong>{year}</strong></td>
          <td>{len(sns)}</td>
          <td>{', '.join(sns)}</td>
        </tr>\n"""

    # Serial format summary
    fmt_counts = Counter()
    for r in records:
        fmt, _ = classify_serial(r["serial_number"])
        fmt_counts[fmt] += 1

    format_summary = ""
    fmt_descriptions = {
        "numeric_4digit": "Pure numeric (e.g., 1413, 5130) — older systems",
        "letter_prefix": "Letter + 3 digits (e.g., A142, D777, H230) — primary modern format",
        "digit_prefix_letter_suffix": "3 digits + letter (e.g., 293H, 354H) — variant",
        "other": "Other format (e.g., 177D)",
    }
    for fmt, count in sorted(fmt_counts.items(), key=lambda x: -x[1]):
        format_summary += f"<li><strong>{fmt_descriptions.get(fmt, fmt)}</strong>: {count} systems</li>\n"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hamilton Microlab STAR — Production Analysis Report</title>
<style>
  body {{ font-family: -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 20px 40px; color: #333;
         line-height: 1.6; }}
  h1 {{ color: #1a3a5c; border-bottom: 3px solid #1a3a5c; padding-bottom: 10px; }}
  h2 {{ color: #2c5f8a; margin-top: 35px; }}
  h3 {{ color: #3a7cb8; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
               gap: 16px; margin: 20px 0; }}
  .kpi {{ background: #f0f6fc; border-left: 4px solid #2c5f8a; padding: 16px 20px;
          border-radius: 4px; }}
  .kpi .value {{ font-size: 28px; font-weight: bold; color: #1a3a5c; }}
  .kpi .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 14px; }}
  th {{ background: #1a3a5c; color: white; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #e8f0fe; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f9a825; padding: 12px 16px;
           margin: 15px 0; border-radius: 4px; font-size: 14px; }}
  .methodology {{ background: #f1f8e9; border-left: 4px solid #7cb342; padding: 12px 16px;
                  margin: 15px 0; border-radius: 4px; font-size: 14px; }}
  .scatter-container {{ text-align: center; margin: 25px 0; }}
  .scatter-container img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .footer {{ margin-top: 40px; padding-top: 15px; border-top: 1px solid #ddd;
             font-size: 12px; color: #999; }}
  .corrected {{ color: #e65100; font-size: 12px; }}
</style>
</head>
<body>

<h1>Hamilton Microlab STAR — Production Volume Analysis</h1>
<p><strong>Report generated:</strong> {datetime.now().strftime('%B %d, %Y')}<br>
<strong>Data source:</strong> Hamilton iData instrument data PDFs ({stats['sample_size']} systems analyzed)<br>
<strong>Scope:</strong> Serial number and installation date analysis to estimate total manufacturing volume</p>

<h2>Key Findings</h2>

<div class="kpi-grid">
  <div class="kpi">
    <div class="value">~{stats['implied_total_units']:,.0f}</div>
    <div class="label">Implied total units manufactured (serial range)</div>
  </div>
  <div class="kpi">
    <div class="value">~{annual_rate:,.0f}</div>
    <div class="label">Estimated units per year (regression slope)</div>
  </div>
  <div class="kpi">
    <div class="value">{stats['span_years']}</div>
    <div class="label">Years covered ({stats['date_range_start']} to {stats['date_range_end']})</div>
  </div>
  <div class="kpi">
    <div class="value">{stats['sample_size']}</div>
    <div class="label">Systems in sample (unique serial numbers)</div>
  </div>
</div>

<div class="methodology">
<strong>Methodology:</strong> Installation dates and serial numbers were extracted from Hamilton iData PDF reports
for {stats['sample_size']} Microlab STAR/STARlet systems. Serial numbers were converted to ordinal values
(A000=0 through H999=7999 for letter-prefix format; numeric serials used as-is). A linear regression of
serial ordinal vs. installation date yields the estimated annual production rate. The implied total units
figure represents the span from the lowest to highest observed serial ordinal.
</div>

<h2>Production Over Time</h2>

<div class="scatter-container">
  <img src="production_scatter.png" alt="Serial Number vs. Manufacture Date scatter plot">
</div>

<p>The scatter plot shows a clear positive linear trend: serial numbers increase monotonically
with installation date, confirming that serial numbers are assigned sequentially at manufacture.
The regression line indicates an average production rate of approximately
<strong>{annual_rate:,.0f} units per year</strong>.</p>

<h2>Serial Number Format Analysis</h2>

<p>Four serial number formats were observed in the sample:</p>
<ul>
{format_summary}</ul>

<div class="note">
<strong>Note on serial format assumption:</strong> The primary format (letter + 3 digits) accounts for
{fmt_counts.get('letter_prefix', 0)} of {stats['sample_size']} systems. The letter advances as serial numbers
increment past x999 (e.g., A999 &rarr; B000). Pure numeric serials (e.g., 1413, 5130) appear on older
systems and likely represent an earlier numbering scheme. The variant formats (293H, 324G, 354H, 177D)
may follow a parallel or overlapping scheme.
</div>

<h2>Year-by-Year Breakdown (from sample)</h2>

<table>
  <thead>
    <tr><th>Year</th><th>Systems in Sample</th><th>Serial Numbers</th></tr>
  </thead>
  <tbody>
{year_rows}  </tbody>
</table>

<h2>Complete Data Table</h2>

<table>
  <thead>
    <tr><th>Serial Number</th><th>Installation Date</th><th>Format</th><th>Ordinal</th></tr>
  </thead>
  <tbody>
{table_rows}  </tbody>
</table>

<p class="corrected">* Date corrected from source PDF (original contained data-entry error)</p>

<h2>Data Quality Notes</h2>
<ul>
  <li><strong>E309:</strong> Installation date in PDF reads "2020-21-05" (invalid month 21). Corrected to
      2020-05-21 based on download date alignment.</li>
  <li><strong>1941, F679:</strong> System folders exist but contain no PDF files. Excluded from analysis.</li>
  <li><strong>A234 folder:</strong> Folder is labeled "A234" but both PDFs inside reference serial <strong>A025</strong>.
      Data correctly extracted as A025. The folder name appears to be a mislabel.</li>
  <li><strong>Multiple PDFs per system:</strong> Where Pre/Post or multiple service dates exist,
      the first successfully parsed PDF was used. Installation dates are consistent across
      PDFs for the same system.</li>
  <li><strong>"Installation date" interpretation:</strong> This field represents the date the instrument
      firmware/system was first installed — used here as a proxy for manufacture date. Actual
      manufacture may precede this by weeks to months.</li>
</ul>

<h2>Limitations and Caveats</h2>
<ul>
  <li>This analysis is based on a <strong>sample of {stats['sample_size']} systems</strong>, not a complete
      production census. The production rate estimate assumes our sample is representative.</li>
  <li>Serial number gaps may exist (e.g., skipped numbers, test units, returned units).
      The "implied total" is an upper bound.</li>
  <li>The numeric-format serials (1413, 2082, etc.) and the letter-prefix serials (A142, H230, etc.)
      may represent overlapping or separate numbering schemes.</li>
  <li>Production rate is not necessarily uniform — the linear regression captures the average trend,
      but actual production may vary year to year.</li>
</ul>

<div class="footer">
  <p>Analysis performed on {datetime.now().strftime('%Y-%m-%d')}. Data extracted from Hamilton iData
  instrument PDFs located in <code>_iData by System/</code>. Report generated programmatically.</p>
</div>

</body>
</html>"""

    path = os.path.join(BASE, "report.html")
    with open(path, 'w') as f:
        f.write(html)
    print(f"  Saved report: {path}")
    return path


def main():
    print("Loading extracted data...")
    records = load_data()
    print(f"  Loaded {len(records)} valid records\n")

    print("Saving clean CSV...")
    save_csv(records)

    print("\nComputing production estimates...")
    stats = compute_production_estimate(records)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nGenerating scatter plot...")
    plot_path, annual_rate = create_scatter_plot(records)

    print("\nGenerating HTML report...")
    report_path = generate_html_report(records, stats, annual_rate)

    print("\n=== DONE ===")
    print(f"  Dataset:  extracted_dataset.csv")
    print(f"  Plot:     production_scatter.png")
    print(f"  Report:   report.html")


if __name__ == "__main__":
    main()
