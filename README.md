# Cuban Land Document Analysis Pipeline

Analysis pipeline for scanned Cuban land documents (PDFs) related to pre-revolutionary property ownership. Designed to identify legally significant documents for potential land claims.

## Features

- **Multi-page document grouping** - Automatically groups related scanned pages into unified documents
- **OCR post-processing** - Fixes common OCR errors (1/l, 0/o substitutions)
- **Spanish to English translation** - Chunked translation for long documents using MarianMT
- **Legal entity extraction** - Identifies parties, properties, and legal references
- **Document classification** - Categorizes documents (wills, deeds, purchase agreements, etc.)
- **Relevance scoring** - Automatic CRITICAL/HIGH/MEDIUM/LOW relevance assessment

## Scripts

| Script | Description |
|--------|-------------|
| `analyze_docs.py` | Main analysis pipeline |
| `analyze_docs_claude.py` | Alternative pipeline with Claude integration |
| `audit_pipeline.py` | Audit and validation utilities |
| `convert_to_pdf.py` | PDF conversion utilities |
| `test_*.py` | Test scripts for pipeline components |

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python analyze_docs.py
```

Output reports are saved to `critical_files/` directory.

## Requirements

- Python 3.12+
- PyTorch
- Transformers (MarianMT for translation)
- EasyOCR
- pdf2image

## Document Types Detected

- Last Will and Testament
- Purchase & Sale Agreement
- Property Deed
- Power of Attorney
- Property Partition
- Mortgage Document
- Property Declaration
- Land Boundary Survey
- Property Transfer
- Property Holdings Record
- Heir Documentation
- Tax/Assessment Document
- Land Registry Document
- Administrative Form

## License

Private - All rights reserved.
