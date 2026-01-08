#!/usr/bin/env python3
"""Quick test of Claude API on one document."""

import os
import easyocr
from pdf2image import convert_from_path
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

# Test on the 1933 document
TEST_FILE = "/Users/juanceresa/Desktop/cs/translations/1. Rodriguez Queral Properties and holdings from Cuba/1933 7 8 F. Rodriguez Purchase Agreement.pdf"

def main():
    print("Initializing...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    reader = easyocr.Reader(['es'], gpu=False)

    print(f"\nProcessing: {os.path.basename(TEST_FILE)}")

    # OCR
    images = convert_from_path(TEST_FILE, first_page=1, last_page=1)
    image_np = np.array(images[0])
    ocr_result = reader.readtext(image_np, detail=0, paragraph=True)
    ocr_text = "\n".join(ocr_result)

    print(f"OCR extracted {len(ocr_text)} chars")
    print("\n--- RAW OCR ---")
    print(ocr_text[:500])
    print("...")

    # Claude API
    print("\n--- SENDING TO CLAUDE ---")

    prompt = f"""You are an expert in Cuban legal documents from the 1920s-1960s.

RAW OCR TEXT (contains errors):
```
{ocr_text}
```

Please:
1. Correct the OCR errors and provide the CORRECTED SPANISH TEXT
2. Provide an accurate ENGLISH TRANSLATION
3. Identify the document type, date, parties, and properties mentioned

Respond in JSON format:
{{
  "corrected_spanish": "...",
  "english_translation": "...",
  "document_type": "...",
  "date": "...",
  "parties": [...],
  "properties": [...],
  "summary": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nTokens used - Input: {response.usage.input_tokens}, Output: {response.usage.output_tokens}")
    print(f"Estimated cost: ${(response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000:.4f}")

    result_text = response.content[0].text

    # Parse JSON
    if "```json" in result_text:
        json_str = result_text.split("```json")[1].split("```")[0]
    elif "```" in result_text:
        json_str = result_text.split("```")[1].split("```")[0]
    else:
        json_str = result_text

    result = json.loads(json_str.strip())

    print("\n--- CORRECTED SPANISH ---")
    print(result.get("corrected_spanish", "N/A")[:1000])

    print("\n--- ENGLISH TRANSLATION ---")
    print(result.get("english_translation", "N/A")[:1000])

    print("\n--- EXTRACTED INFO ---")
    print(f"Type: {result.get('document_type')}")
    print(f"Date: {result.get('date')}")
    print(f"Parties: {result.get('parties')}")
    print(f"Properties: {result.get('properties')}")
    print(f"Summary: {result.get('summary')}")

if __name__ == "__main__":
    main()
