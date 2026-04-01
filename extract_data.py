#!/usr/bin/env python3
"""
Extract serial numbers and installation dates from Hamilton Microlab STAR iData PDFs.

Two PDF formats are handled:
1. "Microlab STAR Instrument Data" — Master table on page 1 with labeled rows
   (Serial number instrument, Installation date)
2. "Hamilton Instrument Data" — Index on page 1, Master table on page 2 with
   coded fields (si = system installation date, serial from title/folder)
"""

import os
import re
import json
import pymupdf  # fitz

DATA_DIR = os.path.join(os.path.dirname(__file__), "_iData by System")


def extract_text_from_page(pdf_path, page_num):
    """Extract text from a specific page of a PDF."""
    try:
        doc = pymupdf.open(pdf_path)
        if page_num < len(doc):
            text = doc[page_num].get_text()
            doc.close()
            return text
        doc.close()
    except Exception as e:
        print(f"  Error reading {pdf_path} page {page_num}: {e}")
    return ""


def parse_microlab_format(text):
    """
    Parse Microlab STAR Instrument Data format.
    Looks for 'Serial number instrument' and 'Installation date' in the text.
    """
    serial = None
    install_date = None

    # Serial number instrument
    m = re.search(r'Serial\s+number\s+instrument\s+(\S+)', text, re.IGNORECASE)
    if m:
        serial = m.group(1).strip()

    # Installation date
    m = re.search(r'Installation\s+date\s+(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    if m:
        install_date = m.group(1)

    return serial, install_date


def parse_hamilton_format(text):
    """
    Parse Hamilton Instrument Data format (page 2).
    Fields are in a table with code columns like 'si' for system installation date.
    """
    serial = None
    install_date = None

    # Look for system installation date — pattern: "system installation date si YYYY-MM-DD"
    m = re.search(r'system\s+installation\s+date\s+si\s+(\d{4}-\d{2}-\d{2})', text, re.IGNORECASE)
    if m:
        install_date = m.group(1)

    return serial, install_date


def extract_from_pdf(pdf_path, folder_name):
    """
    Extract serial number and installation date from a single PDF.
    Returns (serial, install_date, format_type) or (None, None, None).
    """
    # Try page 1 first (Microlab format has data on page 0)
    text_p0 = extract_text_from_page(pdf_path, 0)

    if "Microlab STAR Instrument Data" in text_p0 or "Serial number instrument" in text_p0:
        serial, install_date = parse_microlab_format(text_p0)
        if serial and install_date:
            return serial, install_date, "microlab"

    # Try Hamilton format (data on page 1, i.e. second page)
    if "Instrument Data" in text_p0 or "Index" in text_p0:
        text_p1 = extract_text_from_page(pdf_path, 1)
        if text_p1:
            _, install_date = parse_hamilton_format(text_p1)
            if install_date:
                # Serial comes from the folder name for this format
                return folder_name, install_date, "hamilton"

    # Fallback: try all first 3 pages for any date pattern
    for page_num in range(min(3, 10)):
        text = extract_text_from_page(pdf_path, page_num) if page_num > 0 else text_p0
        if not text:
            continue

        # Try microlab patterns
        serial, install_date = parse_microlab_format(text)
        if serial and install_date:
            return serial, install_date, "microlab"

        # Try hamilton patterns
        _, install_date = parse_hamilton_format(text)
        if install_date:
            return folder_name, install_date, "hamilton"

    return None, None, None


def main():
    results = {}
    errors = []

    folders = sorted(os.listdir(DATA_DIR))
    print(f"Found {len(folders)} system folders\n")

    for folder in folders:
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        pdfs = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
        if not pdfs:
            errors.append(f"{folder}: No PDFs found")
            continue

        # Try each PDF — prefer Microlab format (has embedded serial) over Hamilton
        pdfs.sort()
        extracted = False
        candidates = []
        for pdf_name in pdfs:
            pdf_path = os.path.join(folder_path, pdf_name)
            serial, install_date, fmt = extract_from_pdf(pdf_path, folder)
            if serial and install_date:
                candidates.append((serial, install_date, fmt, pdf_name))

        # Prefer microlab format (has real serial number), then hamilton
        candidates.sort(key=lambda c: (0 if c[2] == "microlab" else 1))
        if candidates:
            serial, install_date, fmt, pdf_name = candidates[0]
            if folder not in results:
                results[folder] = {
                    "folder": folder,
                    "serial_number": serial,
                    "installation_date": install_date,
                    "format": fmt,
                    "source_file": pdf_name
                }
                print(f"  {folder}: SN={serial}, Date={install_date} ({fmt}) from {pdf_name}")
                extracted = True

        if not extracted:
            errors.append(f"{folder}: Could not extract data from any PDF")
            # Print what we found for debugging
            for pdf_name in pdfs[:1]:
                pdf_path = os.path.join(folder_path, pdf_name)
                text = extract_text_from_page(pdf_path, 0)
                print(f"  {folder}: FAILED — page0 preview: {text[:200]!r}")

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "extracted_data.json")
    with open(output_path, 'w') as f:
        json.dump(list(results.values()), f, indent=2)

    print(f"\n--- Summary ---")
    print(f"Successfully extracted: {len(results)} systems")
    print(f"Failed: {len(errors)} systems")
    for err in errors:
        print(f"  ERROR: {err}")
    print(f"\nData saved to: {output_path}")


if __name__ == "__main__":
    main()
