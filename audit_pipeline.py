#!/usr/bin/env python3
"""
Audit script to verify OCR pipeline logic.
"""

import os
import re
from collections import defaultdict

TARGET_DIR = "/Users/juanceresa/Desktop/cs/translations"

# Current keywords from analyze_docs_claude.py (UPDATED)
KEYWORDS = [
    "Last Will", "Testament", "Purchase", "Sale", "Buy", "Sell",
    "Deed", "Declaration", "Power", "Attorney", "Atty", "Heir",
    "Partition", "Partioning", "Boundaries", "Boundary", "Boundries",
    "Transfer", "Mortgage", "Holdings", "Agreement", "Agmt",
    "Property", "Prop", "Finca", "Hacienda", "Dominio",
    "Compraventa", "Compra", "Venta", "Escritura", "Declaración",
    "Hipoteca", "Cancelacion", "Testamento", "Herencia", "Heredero",
    "Partición", "Poder", "Linderos", "Expediente",
    # Additional keywords to capture missed files
    "Master Plan", "Livestock", "Anex", "Annex", "Letter", "Lawyer",
    "Millage", "Tax", "Credit", "Report", "Employee", "Buildings",
    "Aurelia", "Aguaras", "Ceresa", "Rodriguez", "Queral",
]


def get_base_document_name(filename):
    """
    Extract base document name from multi-page scan filenames.
    Handles two patterns:
      1. "Document.2pdf.pdf" -> "Document"
      2. "Document 1.pdf", "Document 2.pdf" -> "Document"
    """
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]

    # Pattern 1: Remove .Npdf suffix (e.g., .2pdf, .10pdf)
    name = re.sub(r'\.\d+pdf$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\d+pdf$', '', name, flags=re.IGNORECASE)

    # Pattern 2: Remove trailing " N" where N is a number
    name = re.sub(r'\s+\d{1,2}$', '', name)

    return name.strip()


def audit_file_discovery():
    """Audit 1: Check which files are captured vs missed."""
    print("=" * 70)
    print("AUDIT 1: FILE DISCOVERY")
    print("=" * 70)

    all_pdfs = []
    matched_pdfs = []
    missed_pdfs = []

    for root, dirs, files in os.walk(TARGET_DIR):
        if 'critical_files' in root or '.venv' in root:
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                full_path = os.path.join(root, file)
                all_pdfs.append(full_path)

                if any(k.lower() in file.lower() for k in KEYWORDS):
                    matched_pdfs.append(full_path)
                else:
                    missed_pdfs.append(full_path)

    print(f"\nTotal PDFs found: {len(all_pdfs)}")
    print(f"Matched by keywords: {len(matched_pdfs)} ({100*len(matched_pdfs)/len(all_pdfs):.1f}%)")
    print(f"Missed (no keyword match): {len(missed_pdfs)}")

    if missed_pdfs:
        print(f"\n--- MISSED FILES (not matching any keyword) ---")
        for f in missed_pdfs[:20]:
            print(f"  - {os.path.basename(f)}")
        if len(missed_pdfs) > 20:
            print(f"  ... and {len(missed_pdfs) - 20} more")

    return matched_pdfs, missed_pdfs


def audit_grouping(matched_pdfs):
    """Audit 2: Check multi-page document grouping."""
    print("\n" + "=" * 70)
    print("AUDIT 2: MULTI-PAGE DOCUMENT GROUPING")
    print("=" * 70)

    grouped = defaultdict(list)
    for path in matched_pdfs:
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        base_name = get_base_document_name(filename)
        key = os.path.join(directory, base_name)
        grouped[key].append(path)

    # Sort pages within each group
    def sort_key(path):
        filename = os.path.basename(path)

        # Pattern 1: .Npdf.pdf suffix
        match = re.search(r'\.?(\d+)pdf\.pdf$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Pattern 2: "Name N.pdf" suffix
        match = re.search(r'\s+(\d{1,2})\.pdf$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))

        # Base file (no number) comes first
        if not re.search(r'(\d+pdf\.pdf$|\s+\d{1,2}\.pdf$)', filename, re.IGNORECASE):
            return 0

        return 999

    for key in grouped:
        grouped[key].sort(key=sort_key)

    # Stats
    single_page = [k for k, v in grouped.items() if len(v) == 1]
    multi_page = [(k, v) for k, v in grouped.items() if len(v) > 1]

    print(f"\nUnique documents: {len(grouped)}")
    print(f"Single-page documents: {len(single_page)}")
    print(f"Multi-page documents: {len(multi_page)}")

    # Show multi-page documents
    print(f"\n--- MULTI-PAGE DOCUMENTS ---")
    for key, files in sorted(multi_page, key=lambda x: -len(x[1]))[:15]:
        doc_name = os.path.basename(key)
        print(f"\n  [{len(files)} pages] {doc_name}")
        for f in files[:5]:
            print(f"      - {os.path.basename(f)}")
        if len(files) > 5:
            print(f"      ... and {len(files) - 5} more pages")

    # Check for potential grouping issues
    print(f"\n--- POTENTIAL GROUPING ISSUES ---")
    issues = []

    for key, files in grouped.items():
        doc_name = os.path.basename(key)

        # Check for numbered pages that might indicate separate documents
        # e.g., "Parada 1.pdf", "Parada 2.pdf" should be separate from ".1pdf.pdf", ".2pdf.pdf"
        numbered_pattern = re.search(r'\s+\d+$', doc_name)
        if numbered_pattern and len(files) == 1:
            # This might be one of a series, check if siblings exist
            base_without_num = re.sub(r'\s+\d+$', '', doc_name)
            dir_path = os.path.dirname(key)
            related = [k for k in grouped.keys() if os.path.dirname(k) == dir_path
                       and re.sub(r'\s+\d+$', '', os.path.basename(k)) == base_without_num]
            if len(related) > 1:
                issues.append((doc_name, f"Part of numbered series with {len(related)} parts"))

    if issues:
        for name, issue in issues[:10]:
            print(f"  - {name}: {issue}")
    else:
        print("  No issues detected")

    return grouped


def audit_naming_patterns(matched_pdfs):
    """Audit 3: Analyze file naming patterns."""
    print("\n" + "=" * 70)
    print("AUDIT 3: FILE NAMING PATTERNS")
    print("=" * 70)

    patterns = {
        "Multi-page (.Npdf.pdf)": [],
        "Numbered series (Name 1.pdf)": [],
        "Date prefix (YYYY M D)": [],
        "Standard single": [],
    }

    for path in matched_pdfs:
        filename = os.path.basename(path)

        if re.search(r'\.\d+pdf\.pdf$', filename, re.IGNORECASE):
            patterns["Multi-page (.Npdf.pdf)"].append(filename)
        elif re.search(r'\s+\d+\.pdf$', filename, re.IGNORECASE):
            patterns["Numbered series (Name 1.pdf)"].append(filename)
        elif re.search(r'^\d{4}\s+\d+\s+\d+', filename):
            patterns["Date prefix (YYYY M D)"].append(filename)
        else:
            patterns["Standard single"].append(filename)

    for pattern, files in patterns.items():
        print(f"\n{pattern}: {len(files)} files")
        for f in files[:3]:
            print(f"    Example: {f}")


def audit_keyword_coverage():
    """Audit 4: Check which keywords are actually matching."""
    print("\n" + "=" * 70)
    print("AUDIT 4: KEYWORD EFFECTIVENESS")
    print("=" * 70)

    keyword_hits = defaultdict(list)

    for root, dirs, files in os.walk(TARGET_DIR):
        if 'critical_files' in root or '.venv' in root:
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                for kw in KEYWORDS:
                    if kw.lower() in file.lower():
                        keyword_hits[kw].append(file)

    print("\nKeywords sorted by match count:")
    for kw, files in sorted(keyword_hits.items(), key=lambda x: -len(x[1])):
        print(f"  {kw:20} : {len(files):3} files")

    # Keywords with zero matches
    zero_matches = [kw for kw in KEYWORDS if kw not in keyword_hits]
    if zero_matches:
        print(f"\nKeywords with NO matches: {zero_matches}")


def main():
    print("CUBAN LAND DOCUMENT PIPELINE - FULL AUDIT")
    print("=" * 70)

    matched, missed = audit_file_discovery()
    grouped = audit_grouping(matched)
    audit_naming_patterns(matched)
    audit_keyword_coverage()

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"Total files: 260")
    print(f"Matched by keywords: {len(matched)} ({100*len(matched)/260:.1f}%)")
    print(f"Unique documents after grouping: {len(grouped)}")
    print(f"Files not captured: {len(missed)}")

    if missed:
        print("\n>>> ACTION REQUIRED: Review missed files above")
        print("    Some may need additional keywords to capture")


if __name__ == "__main__":
    main()
