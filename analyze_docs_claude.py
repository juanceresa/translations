#!/usr/bin/env python3
"""
Cuban Land Document Analysis Pipeline - Claude API Version (Optimized)

Uses Claude 3.5 Sonnet for comprehensive legal analysis, OCR correction,
and translation for land claim documentation.

Target: Villa Aurelia and Hacienda Aguaras estates
Budget: $10-20 for ~78 documents (~$0.04-0.05 per document)
"""

import os
import easyocr
from pdf2image import convert_from_path
import numpy as np
import re
import json
from collections import defaultdict
from anthropic import Anthropic
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()

# Configuration
TARGET_DIR = "/Users/juanceresa/Desktop/cs/translations"
OUTPUT_DIR = os.path.join(TARGET_DIR, "critical_files_claude")

# Expanded keywords to capture all legally relevant document types
KEYWORDS = [
    # English legal terms
    "Last Will", "Testament", "Purchase", "Sale", "Buy", "Sell",
    "Deed", "Declaration", "Power", "Attorney", "Atty", "Heir",
    "Partition", "Partioning", "Boundaries", "Boundary", "Boundries",
    "Transfer", "Mortgage", "Holdings", "Agreement", "Agmt",
    "Property", "Prop", "Finca", "Hacienda", "Dominio",
    # Spanish legal terms
    "Compraventa", "Compra", "Venta", "Escritura", "Declaración",
    "Hipoteca", "Cancelacion", "Testamento", "Herencia", "Heredero",
    "Partición", "Poder", "Linderos", "Expediente",
    # Additional keywords to capture all files
    "Master Plan", "Livestock", "Anex", "Annex", "Letter", "Lawyer",
    "Millage", "Tax", "Credit", "Report", "Employee", "Buildings",
    "Aurelia", "Aguaras", "Ceresa", "Rodriguez", "Queral",
    # Final additions
    "Memorandum", "Carta", "Ipoteca", "Consejo", "Reforma", "Urbana", "Sanz",
]

# Optimal prompt for comprehensive legal analysis
OPTIMAL_PROMPT = """You are a Cuban legal document expert specializing in pre-revolutionary property records (1900-1960). Analyze this OCR-extracted document for a potential land restitution claim on Villa Aurelia and Hacienda Aguaras estates.

DOCUMENT NAME: {doc_name}

RAW OCR TEXT (contains errors - reconstruct carefully):
```
{ocr_text}
```

TASK: Extract all legally relevant information for establishing chain of title.

Respond in this exact JSON structure:

{{
  "document_identification": {{
    "type": "Specific type (e.g., 'Escritura de Compraventa', 'Testamento', 'Partición de Bienes')",
    "date": "YYYY-MM-DD format if possible, otherwise as written",
    "location": "City/Municipality, Province"
  }},

  "chain_of_title": {{
    "transfer_type": "Sale | Inheritance | Partition | Gift | Mortgage | Power of Attorney | Declaration | Other",
    "grantor": {{
      "name": "Full name of person transferring rights",
      "role": "Seller/Testator/Donor/etc.",
      "relationship_to_property": "Owner/Co-owner/Heir/etc."
    }},
    "grantee": {{
      "name": "Full name of person receiving rights",
      "role": "Buyer/Heir/Donee/etc.",
      "relationship_to_grantor": "If mentioned (spouse, child, etc.)"
    }},
    "consideration": "Sale price with currency, or 'N/A' for inheritance/gift"
  }},

  "property_description": {{
    "names": ["All property names mentioned (Finca X, Villa Y, etc.)"],
    "area": "Size with units (caballerías, hectares, m², etc.)",
    "location": "Municipality, Province, or descriptive location",
    "boundaries": {{
      "north": "What/who borders to the north",
      "south": "What/who borders to the south",
      "east": "What/who borders to the east",
      "west": "What/who borders to the west"
    }}
  }},

  "legal_instrument": {{
    "notary": "Name of notary public",
    "notary_location": "City where notary practiced",
    "protocol_number": "Number in notary's protocol book",
    "execution_date": "Date deed was executed",
    "registry": {{
      "office": "Property Registry office location",
      "tomo": "Volume number",
      "folio": "Page number",
      "finca": "Property registry number"
    }},
    "witnesses": ["Names of witnesses if mentioned"]
  }},

  "for_inheritance_docs": {{
    "testator": "Person who made the will (if applicable)",
    "heirs": [
      {{"name": "Heir name", "inheritance": "What they inherit", "relationship": "Relationship to testator"}}
    ],
    "executor": "Estate executor if named"
  }},

  "family_relationships": [
    "List all family relationships mentioned (e.g., 'Elena Rodriguez Queral - daughter of Aurelia Queral')"
  ],

  "cross_references": {{
    "prior_deeds": ["References to earlier documents"],
    "related_properties": ["Other properties mentioned in same family"],
    "registry_references": ["Any registry inscriptions cited"]
  }},

  "claim_relevance": {{
    "level": "CRITICAL | HIGH | MEDIUM | LOW",
    "target_properties_mentioned": ["Villa Aurelia", "Hacienda Aguaras", or others from target list],
    "key_families_mentioned": ["Ceresa", "Rodriguez", "Queral", or others],
    "chain_of_title_position": "Describe where this fits (e.g., 'Original acquisition by F. Rodriguez', 'Transfer from Queral to Ceresa', etc.)",
    "reasoning": "Why this document matters for the claim"
  }},

  "document_text": {{
    "corrected_spanish": "Full corrected Spanish text",
    "english_translation": "Full English translation",
    "executive_summary": "2-3 sentence summary of the document's legal significance"
  }},

  "quality_notes": {{
    "ocr_quality": "Good | Moderate | Poor | Very Poor",
    "missing_information": ["List any critical fields that couldn't be extracted"],
    "verification_needed": ["Things that should be manually verified"]
  }}
}}

CRITICAL INSTRUCTIONS:
1. Fix ALL OCR errors using context (1→l, 0→o, common Spanish legal terminology)
2. For Cuban legal docs: caballería = ~13.4 hectares, common notary format is "Número X del Protocolo"
3. Be thorough - this will be used for legal proceedings
4. If information is not present, use null, don't guess
5. Pay special attention to: Rodriguez, Queral, Ceresa family names and Villa Aurelia, Hacienda Aguaras, Parada, Finca Aguaras properties"""


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

    # Pattern 2: Remove trailing " N" where N is a number (e.g., "Parada 1" -> "Parada")
    name = re.sub(r'\s+\d{1,2}$', '', name)

    return name.strip()


def find_and_group_files(directory):
    """Find all PDF files matching keywords and group multi-page scans together."""
    all_pdfs = []
    print(f"Scanning {directory} for prioritized files...")

    for root, dirs, files in os.walk(directory):
        if 'critical_files' in root or '.venv' in root:
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                if any(k.lower() in file.lower() for k in KEYWORDS):
                    full_path = os.path.join(root, file)
                    all_pdfs.append(full_path)

    grouped = defaultdict(list)
    for path in all_pdfs:
        dir_path = os.path.dirname(path)
        filename = os.path.basename(path)
        base_name = get_base_document_name(filename)
        key = os.path.join(dir_path, base_name)
        grouped[key].append(path)

    def sort_key(path):
        """Sort pages in order: base file first, then numbered pages."""
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

    return grouped


def process_with_claude(client, ocr_text, doc_name):
    """Use Claude to perform comprehensive legal analysis."""
    prompt = OPTIMAL_PROMPT.format(doc_name=doc_name, ocr_text=ocr_text)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4500,
            messages=[{"role": "user", "content": prompt}]
        )

        # Track token usage
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        response_text = response.content[0].text

        # Parse JSON
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0]
        else:
            json_str = response_text

        result = json.loads(json_str.strip())
        result['_token_usage'] = {'input': input_tokens, 'output': output_tokens}
        return result

    except Exception as e:
        print(f"    Claude API error: {str(e)[:100]}")
        return None


def generate_report(doc_name, file_paths, analysis, combined_ocr):
    """Generate comprehensive markdown report from analysis."""

    doc_id = analysis.get("document_identification", {})
    cot = analysis.get("chain_of_title", {})
    prop = analysis.get("property_description", {})
    legal = analysis.get("legal_instrument", {})
    inherit = analysis.get("for_inheritance_docs", {})
    family = analysis.get("family_relationships", [])
    cross_refs = analysis.get("cross_references", {})
    relevance = analysis.get("claim_relevance", {})
    doc_text = analysis.get("document_text", {})
    quality = analysis.get("quality_notes", {})

    # Format source files
    if len(file_paths) > 1:
        files_list = "\n".join([f"  - `{os.path.basename(f)}`" for f in file_paths])
        file_section = f"**Source Files** ({len(file_paths)} pages):\n{files_list}"
    else:
        file_section = f"**Source File**: `{os.path.basename(file_paths[0])}`"

    # Format grantor/grantee
    grantor = cot.get("grantor", {}) or {}
    grantee = cot.get("grantee", {}) or {}

    # Format boundaries
    boundaries = prop.get("boundaries", {}) or {}
    boundaries_text = ""
    if any(boundaries.values()):
        boundaries_text = f"""
| Direction | Boundary |
|-----------|----------|
| North | {boundaries.get('north', 'N/A')} |
| South | {boundaries.get('south', 'N/A')} |
| East | {boundaries.get('east', 'N/A')} |
| West | {boundaries.get('west', 'N/A')} |
"""

    # Format registry info
    registry = legal.get("registry", {}) or {}
    registry_text = ""
    if any(registry.values()):
        registry_text = f"Tomo: {registry.get('tomo', 'N/A')} | Folio: {registry.get('folio', 'N/A')} | Finca: {registry.get('finca', 'N/A')}"

    # Format heirs (for wills)
    heirs_section = ""
    heirs = inherit.get("heirs", [])
    if heirs and isinstance(heirs, list) and len(heirs) > 0:
        heirs_list = []
        for h in heirs:
            if isinstance(h, dict):
                heirs_list.append(f"- **{h.get('name', 'Unknown')}** ({h.get('relationship', 'N/A')}): {h.get('inheritance', 'N/A')}")
        if heirs_list:
            heirs_section = "\n".join(heirs_list)

    # Format family relationships
    family_section = "\n".join([f"- {r}" for r in family]) if family else "- None identified"

    # Format cross-references
    prior_deeds = cross_refs.get("prior_deeds", [])
    prior_deeds_section = "\n".join([f"- {d}" for d in prior_deeds]) if prior_deeds else "- None"

    # Format missing info
    missing = quality.get("missing_information", [])
    missing_section = "\n".join([f"- {m}" for m in missing]) if missing else "- None"

    verification = quality.get("verification_needed", [])
    verification_section = "\n".join([f"- {v}" for v in verification]) if verification else "- None"

    report = f"""# LEGAL DOCUMENT ANALYSIS REPORT

## {doc_name}

---

## Quick Reference

| Field | Value |
|-------|-------|
| **Document Type** | {doc_id.get('type', 'Unknown')} |
| **Date** | {doc_id.get('date', 'Unknown')} |
| **Location** | {doc_id.get('location', 'Unknown')} |
| **Relevance** | **{relevance.get('level', 'Unknown')}** |
| **Pages** | {len(file_paths)} |
| **OCR Quality** | {quality.get('ocr_quality', 'Unknown')} |

{file_section}

---

## Executive Summary

{doc_text.get('executive_summary', 'No summary available.')}

**Chain of Title Position:** {relevance.get('chain_of_title_position', 'Unknown')}

**Claim Relevance:** {relevance.get('reasoning', 'No reasoning provided.')}

---

## Chain of Title

| Role | Name | Details |
|------|------|---------|
| **Grantor** (Transferring) | {grantor.get('name', 'N/A')} | {grantor.get('role', '')} - {grantor.get('relationship_to_property', '')} |
| **Grantee** (Receiving) | {grantee.get('name', 'N/A')} | {grantee.get('role', '')} - {grantee.get('relationship_to_grantor', '')} |

**Transfer Type:** {cot.get('transfer_type', 'Unknown')}

**Consideration:** {cot.get('consideration', 'N/A')}

---

## Property Description

**Property Names:** {', '.join(prop.get('names', ['N/A'])) if prop.get('names') else 'N/A'}

**Area:** {prop.get('area', 'Not specified')}

**Location:** {prop.get('location', 'Not specified')}

### Boundaries
{boundaries_text if boundaries_text else '*No boundaries extracted*'}

---

## Legal Instrument Details

| Field | Value |
|-------|-------|
| **Notary** | {legal.get('notary', 'N/A')} |
| **Notary Location** | {legal.get('notary_location', 'N/A')} |
| **Protocol Number** | {legal.get('protocol_number', 'N/A')} |
| **Execution Date** | {legal.get('execution_date', 'N/A')} |
| **Registry** | {registry_text if registry_text else 'N/A'} |

**Witnesses:** {', '.join(legal.get('witnesses', [])) if legal.get('witnesses') else 'None listed'}

---

## Family Relationships

{family_section}

---

## Cross-References

### Prior Deeds Referenced
{prior_deeds_section}

### Related Properties
{', '.join(cross_refs.get('related_properties', ['None'])) if cross_refs.get('related_properties') else 'None'}

---

## Target Properties & Families

**Target Properties Mentioned:** {', '.join(relevance.get('target_properties_mentioned', ['None'])) if relevance.get('target_properties_mentioned') else 'None'}

**Key Families Mentioned:** {', '.join(relevance.get('key_families_mentioned', ['None'])) if relevance.get('key_families_mentioned') else 'None'}

---
"""

    # Add inheritance section if applicable
    if inherit.get('testator') or heirs_section:
        report += f"""
## Inheritance Details

**Testator:** {inherit.get('testator', 'N/A')}

**Executor:** {inherit.get('executor', 'N/A')}

### Heirs
{heirs_section if heirs_section else '- None specified'}

---
"""

    # Add document text
    report += f"""
## Full Document Text

### Corrected Spanish

{doc_text.get('corrected_spanish', '*Text not available*')}

### English Translation

{doc_text.get('english_translation', '*Translation not available*')}

---

## Quality Notes

**OCR Quality:** {quality.get('ocr_quality', 'Unknown')}

### Missing Information
{missing_section}

### Verification Needed
{verification_section}

---

## Raw OCR Reference

<details>
<summary>Click to expand raw OCR text</summary>

```
{combined_ocr[:5000]}{'...[truncated]' if len(combined_ocr) > 5000 else ''}
```

</details>

---

*Analysis performed by Claude 3.5 Sonnet | Manual verification recommended for legal proceedings*
"""

    return report


def process_document_group(doc_name, file_paths, reader, client):
    """Process a group of files representing a single multi-page document."""
    all_ocr_text = []

    print(f"  Extracting text from {len(file_paths)} page(s)...")

    for page_num, file_path in enumerate(file_paths, 1):
        try:
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                continue

            image_np = np.array(images[0])
            ocr_result = reader.readtext(image_np, detail=0, paragraph=True)
            page_text = "\n".join(ocr_result)

            if len(page_text) > 20:
                all_ocr_text.append(f"[Page {page_num}]\n{page_text}")
                print(f"    Page {page_num}: {len(page_text)} chars")

        except Exception as e:
            print(f"    Page {page_num}: Error - {str(e)[:50]}")

    if not all_ocr_text:
        return None, None

    combined_ocr = "\n\n".join(all_ocr_text)

    # Truncate if extremely long (save tokens, but allow more than before)
    if len(combined_ocr) > 20000:
        combined_ocr = combined_ocr[:20000] + "\n\n[...truncated due to length...]"

    print(f"  Sending to Claude API ({len(combined_ocr)} chars)...")
    analysis = process_with_claude(client, combined_ocr, doc_name)

    if not analysis:
        return None, None

    # Generate report
    print(f"  Generating report...")
    report = generate_report(doc_name, file_paths, analysis, combined_ocr)

    return report, analysis


def main():
    # Setup
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("=" * 70)
    print("CUBAN LAND DOCUMENT ANALYSIS PIPELINE")
    print("Claude API Version - Optimized for Legal Claims")
    print("=" * 70)

    # Initialize Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not found in .env file")
        return

    client = Anthropic(api_key=api_key)
    print("\n[1/3] Claude API initialized")

    # Initialize OCR
    print("[2/3] Initializing EasyOCR (Spanish)...")
    reader = easyocr.Reader(['es'], gpu=False)

    # Find and group files
    print("\n[3/3] Scanning for documents...")
    grouped_docs = find_and_group_files(TARGET_DIR)

    total_docs = len(grouped_docs)
    total_files = sum(len(files) for files in grouped_docs.values())
    print(f"\nFound {total_docs} unique documents ({total_files} total files)")
    print("=" * 70)

    # Process each document
    success_count = 0
    fail_count = 0
    total_input_tokens = 0
    total_output_tokens = 0

    # Store all analyses for summary
    all_analyses = []

    for i, (doc_key, file_paths) in enumerate(grouped_docs.items(), 1):
        doc_name = os.path.basename(doc_key)
        print(f"\n[{i}/{total_docs}] {doc_name}")

        # Check if already processed - skip if output exists
        output_filename = f"{doc_name}.md"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        json_filename = f"{doc_name}.json"
        json_path = os.path.join(OUTPUT_DIR, json_filename)

        if os.path.exists(output_path) and os.path.exists(json_path):
            print(f"  -> SKIPPED (already processed)")
            # Load existing analysis for index
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_analysis = json.load(f)
                all_analyses.append({
                    'name': doc_name,
                    'relevance': existing_analysis.get('claim_relevance', {}).get('level', 'Unknown'),
                    'type': existing_analysis.get('document_identification', {}).get('type', 'Unknown'),
                    'date': existing_analysis.get('document_identification', {}).get('date', 'Unknown'),
                    'properties': existing_analysis.get('claim_relevance', {}).get('target_properties_mentioned', []),
                    'families': existing_analysis.get('claim_relevance', {}).get('key_families_mentioned', [])
                })
                success_count += 1
            except:
                pass
            continue

        try:
            report, analysis = process_document_group(doc_name, file_paths, reader, client)

            if report and analysis:
                # Track tokens
                tokens = analysis.get('_token_usage', {})
                total_input_tokens += tokens.get('input', 0)
                total_output_tokens += tokens.get('output', 0)

                # Save report
                output_filename = f"{doc_name}.md"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)

                # Save JSON for data processing
                json_filename = f"{doc_name}.json"
                json_path = os.path.join(OUTPUT_DIR, json_filename)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(analysis, f, indent=2, ensure_ascii=False)

                # Track for summary
                all_analyses.append({
                    'name': doc_name,
                    'relevance': analysis.get('claim_relevance', {}).get('level', 'Unknown'),
                    'type': analysis.get('document_identification', {}).get('type', 'Unknown'),
                    'date': analysis.get('document_identification', {}).get('date', 'Unknown'),
                    'properties': analysis.get('claim_relevance', {}).get('target_properties_mentioned', []),
                    'families': analysis.get('claim_relevance', {}).get('key_families_mentioned', [])
                })

                relevance = analysis.get('claim_relevance', {}).get('level', '?')
                print(f"  -> SUCCESS [{relevance}]: {output_filename}")
                success_count += 1
            else:
                print(f"  -> SKIPPED: Processing failed")
                fail_count += 1

        except Exception as e:
            print(f"  -> FAILED: {str(e)}")
            fail_count += 1

    # Calculate cost
    cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000

    # Generate index/summary file
    print("\nGenerating index...")
    index_content = f"""# Document Analysis Index

## Summary

| Metric | Value |
|--------|-------|
| Documents Processed | {success_count} |
| Documents Failed | {fail_count} |
| Total Input Tokens | {total_input_tokens:,} |
| Total Output Tokens | {total_output_tokens:,} |
| **Total Cost** | **${cost:.2f}** |

## Documents by Relevance

### CRITICAL
"""
    for a in sorted([x for x in all_analyses if x['relevance'] == 'CRITICAL'], key=lambda x: x['date'] or ''):
        index_content += f"- [{a['name']}]({a['name']}.md) ({a['date']}) - {a['type']}\n"

    index_content += "\n### HIGH\n"
    for a in sorted([x for x in all_analyses if x['relevance'] == 'HIGH'], key=lambda x: x['date'] or ''):
        index_content += f"- [{a['name']}]({a['name']}.md) ({a['date']}) - {a['type']}\n"

    index_content += "\n### MEDIUM\n"
    for a in sorted([x for x in all_analyses if x['relevance'] == 'MEDIUM'], key=lambda x: x['date'] or ''):
        index_content += f"- [{a['name']}]({a['name']}.md) ({a['date']}) - {a['type']}\n"

    index_content += "\n### LOW\n"
    for a in sorted([x for x in all_analyses if x['relevance'] == 'LOW'], key=lambda x: x['date'] or ''):
        index_content += f"- [{a['name']}]({a['name']}.md) ({a['date']}) - {a['type']}\n"

    with open(os.path.join(OUTPUT_DIR, "INDEX.md"), "w") as f:
        f.write(index_content)

    # Summary
    print("\n" + "=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Documents processed: {success_count}")
    print(f"Documents failed/skipped: {fail_count}")
    print(f"Total tokens: {total_input_tokens:,} in / {total_output_tokens:,} out")
    print(f"Total cost: ${cost:.2f}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("\nREMINDER: Rotate your API key for security!")


if __name__ == "__main__":
    main()
