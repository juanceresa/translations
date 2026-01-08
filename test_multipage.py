#!/usr/bin/env python3
"""Test multi-page document processing."""

import os
import re
import easyocr
from pdf2image import convert_from_path
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

# Test document: 8-page Parada sale
TEST_DIR = "/Users/juanceresa/Desktop/cs/translations/1. Rodriguez Queral Properties and holdings from Cuba/Villa Aurelia (Parada)/1933 Sale of property parcel known as Parada"


def get_base_document_name(filename):
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    name = re.sub(r'\.\d+pdf$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\d+pdf$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+\d{1,2}$', '', name)
    return name.strip()


def main():
    print("=" * 60)
    print("MULTI-PAGE DOCUMENT TEST")
    print("=" * 60)

    # Find all PDFs in test directory
    pdfs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.pdf')])
    print(f"\nFound {len(pdfs)} PDF files:")
    for p in pdfs:
        print(f"  - {p}")

    # Group by base name
    print(f"\nBase name extraction test:")
    for p in pdfs[:3]:
        base = get_base_document_name(p)
        print(f"  {p} -> '{base}'")

    # Initialize OCR
    print("\nInitializing OCR...")
    reader = easyocr.Reader(['es'], gpu=False)

    # Extract text from first 3 pages
    print("\nExtracting OCR from first 3 pages:")
    all_text = []

    for i, pdf_file in enumerate(pdfs[:3], 1):
        pdf_path = os.path.join(TEST_DIR, pdf_file)
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if images:
                image_np = np.array(images[0])
                ocr_result = reader.readtext(image_np, detail=0, paragraph=True)
                page_text = "\n".join(ocr_result)
                all_text.append(f"[Page {i}]\n{page_text}")
                print(f"  Page {i}: {len(page_text)} chars")
                print(f"    Preview: {page_text[:100]}...")
        except Exception as e:
            print(f"  Page {i}: Error - {e}")

    # Combine and send to Claude
    combined_text = "\n\n".join(all_text)
    print(f"\nCombined text length: {len(combined_text)} chars")

    print("\nSending to Claude API...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""This is a multi-page Cuban legal document (pages 1-3 of 8).

OCR TEXT:
```
{combined_text}
```

Please:
1. Identify what type of document this is
2. Extract the date and parties involved
3. Briefly summarize what this document is about

Respond in JSON:
{{"document_type": "...", "date": "...", "parties": [...], "summary": "..."}}"""
        }]
    )

    print(f"\nTokens: Input={response.usage.input_tokens}, Output={response.usage.output_tokens}")
    print(f"Cost: ${(response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000:.4f}")

    result = response.content[0].text
    print(f"\n--- CLAUDE RESPONSE ---")
    print(result)


if __name__ == "__main__":
    main()
