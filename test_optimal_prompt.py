#!/usr/bin/env python3
"""
Test optimal comprehensive prompt for legal land claim analysis.
Goal: Maximum legal value within $10-20 budget for ~78 documents.
"""

import os
import re
import easyocr
from pdf2image import convert_from_path
import numpy as np
from anthropic import Anthropic
from dotenv import load_dotenv
import json

load_dotenv()

# Test on the Parada sale document (multi-page)
TEST_DIR = "/Users/juanceresa/Desktop/cs/translations/1. Rodriguez Queral Properties and holdings from Cuba/Villa Aurelia (Parada)/1933 Sale of property parcel known as Parada"

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
    "target_properties_mentioned": ["Villa Aurelia", "Hacienda Aguaras", or others from target list"],
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


def main():
    print("=" * 70)
    print("OPTIMAL PROMPT COST TEST")
    print("=" * 70)

    # Get first 3 pages of Parada document
    pdfs = sorted([f for f in os.listdir(TEST_DIR) if f.endswith('.pdf') and 'Parada' in f])[:3]

    print(f"\nTesting on: {pdfs[0].rsplit(' ', 1)[0]}... ({len(pdfs)} pages)")

    # OCR
    print("\nInitializing OCR...")
    reader = easyocr.Reader(['es'], gpu=False)

    all_text = []
    for i, pdf_file in enumerate(pdfs, 1):
        pdf_path = os.path.join(TEST_DIR, pdf_file)
        try:
            images = convert_from_path(pdf_path, first_page=1, last_page=1)
            if images:
                image_np = np.array(images[0])
                ocr_result = reader.readtext(image_np, detail=0, paragraph=True)
                page_text = "\n".join(ocr_result)
                all_text.append(f"[Page {i}]\n{page_text}")
                print(f"  Page {i}: {len(page_text)} chars")
        except Exception as e:
            print(f"  Page {i}: Error - {e}")

    combined_text = "\n\n".join(all_text)

    # Format prompt
    doc_name = "5 3 1933 Sale of property parcel known as Parada"
    prompt = OPTIMAL_PROMPT.format(doc_name=doc_name, ocr_text=combined_text)

    print(f"\nPrompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

    # Send to Claude
    print("\nSending to Claude API...")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000

    print(f"\n{'=' * 50}")
    print("COST ANALYSIS")
    print(f"{'=' * 50}")
    print(f"Input tokens:  {input_tokens:,}")
    print(f"Output tokens: {output_tokens:,}")
    print(f"This document: ${cost:.4f}")
    print(f"\nProjected for 78 documents: ${cost * 78:.2f}")
    print(f"{'=' * 50}")

    # Parse and display result
    result_text = response.content[0].text

    # Extract JSON
    if "```json" in result_text:
        json_str = result_text.split("```json")[1].split("```")[0]
    elif "```" in result_text:
        json_str = result_text.split("```")[1].split("```")[0]
    else:
        json_str = result_text

    try:
        result = json.loads(json_str.strip())

        print("\n" + "=" * 70)
        print("EXTRACTED DATA PREVIEW")
        print("=" * 70)

        print(f"\n--- DOCUMENT IDENTIFICATION ---")
        doc_id = result.get("document_identification", {})
        print(f"Type: {doc_id.get('type')}")
        print(f"Date: {doc_id.get('date')}")
        print(f"Location: {doc_id.get('location')}")

        print(f"\n--- CHAIN OF TITLE ---")
        cot = result.get("chain_of_title", {})
        print(f"Transfer Type: {cot.get('transfer_type')}")
        grantor = cot.get("grantor", {})
        grantee = cot.get("grantee", {})
        print(f"Grantor: {grantor.get('name')} ({grantor.get('role')})")
        print(f"Grantee: {grantee.get('name')} ({grantee.get('role')})")
        print(f"Consideration: {cot.get('consideration')}")

        print(f"\n--- PROPERTY ---")
        prop = result.get("property_description", {})
        print(f"Names: {prop.get('names')}")
        print(f"Area: {prop.get('area')}")
        boundaries = prop.get("boundaries", {})
        print(f"Boundaries: N={boundaries.get('north')}, S={boundaries.get('south')}")

        print(f"\n--- LEGAL INSTRUMENT ---")
        legal = result.get("legal_instrument", {})
        print(f"Notary: {legal.get('notary')} ({legal.get('notary_location')})")
        print(f"Protocol #: {legal.get('protocol_number')}")
        registry = legal.get("registry", {})
        print(f"Registry: Tomo {registry.get('tomo')}, Folio {registry.get('folio')}")

        print(f"\n--- CLAIM RELEVANCE ---")
        rel = result.get("claim_relevance", {})
        print(f"Level: {rel.get('level')}")
        print(f"Target Properties: {rel.get('target_properties_mentioned')}")
        print(f"Key Families: {rel.get('key_families_mentioned')}")
        print(f"Chain Position: {rel.get('chain_of_title_position')}")

        print(f"\n--- EXECUTIVE SUMMARY ---")
        doc_text = result.get("document_text", {})
        print(doc_text.get("executive_summary"))

        print(f"\n--- QUALITY ---")
        quality = result.get("quality_notes", {})
        print(f"OCR Quality: {quality.get('ocr_quality')}")
        print(f"Missing: {quality.get('missing_information')}")

        # Save full result
        with open("/Users/juanceresa/Desktop/cs/translations/sample_output.json", "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nFull JSON saved to: sample_output.json")

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"\nRaw response:\n{result_text[:2000]}")


if __name__ == "__main__":
    main()
