#!/usr/bin/env python3
import os
import easyocr
from pdf2image import convert_from_path
import numpy as np
import sys
import ssl
import re
from collections import defaultdict
from transformers import pipeline
from spellchecker import SpellChecker

# Bypass SSL for model downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
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
    "Partición", "Poder", "Linderos", "Expediente"
]
TARGET_DIR = "/Users/juanceresa/Desktop/cs/translations"
OUTPUT_DIR = os.path.join(TARGET_DIR, "critical_files")

# OCR error corrections (common substitutions in scanned Spanish documents)
OCR_CORRECTIONS = {
    r'\b1os\b': 'los',
    r'\b1a\b': 'la',
    r'\b1as\b': 'las',
    r'\be1\b': 'el',
    r'\bde1\b': 'del',
    r'\bs1\b': 'si',
    r'\bH1l\b': 'Mil',
    r'\bne8=\b': 'mes',
    r'\b0\b(?=\s+sea[ns]?\b)': 'o',  # "0 sean" -> "o sean"
}

# Document type classification based on keywords
DOCUMENT_TYPES = {
    "Last Will and Testament": ["last will", "testament", "testamento", "última voluntad", "herederos"],
    "Purchase & Sale Agreement": ["purchase", "sale", "compraventa", "compra", "venta", "buy", "sell"],
    "Property Deed": ["deed", "escritura", "escritura pública", "título"],
    "Power of Attorney": ["power of attorney", "poder", "atty", "attorney", "apoderado"],
    "Property Partition": ["partition", "partición", "partioning", "división", "segregación"],
    "Mortgage Document": ["mortgage", "hipoteca", "cancelacion de hipoteca", "gravamen"],
    "Property Declaration": ["declaration", "declaración", "declaratoria"],
    "Land Boundary Survey": ["boundaries", "boundary", "boundries", "linderos", "deslinde", "demarcación"],
    "Property Transfer": ["transfer", "transferencia", "cesión", "traspaso"],
    "Property Holdings Record": ["holdings", "bienes", "propiedades", "inmuebles"],
    "Heir Documentation": ["heir", "heredero", "herencia", "sucesión", "co-heir"],
    "Tax/Assessment Document": ["tax", "impuesto", "millage", "contribución", "avalúo"],
    "Land Registry Document": ["registry", "registro", "inscripción", "anotación"],
    "Administrative Form": ["planilla", "formulario", "solicitud", "check off"],
}

# Known family names for party extraction
KNOWN_FAMILIES = [
    "Rodriguez", "Rodriquez", "Queral", "Ceresa", "Cartaya", "Garcia", "Elizalde",
    "Tellez", "Giron", "Sanz", "Berga", "Peña", "Cruz", "Diaz", "Meneses"
]

# Known property names
KNOWN_PROPERTIES = [
    "Villa Aurelia", "Villa Aurielia", "Hacienda Aguaras", "Finca Aguaras",
    "San Augustin", "San Agustin", "Parada", "El Mamey", "La Cuaba",
    "La Caridad", "Villa Amelia", "Soledad de Parada", "Eloisa"
]

def get_base_document_name(filename):
    """
    Extract base document name from multi-page scan filenames.
    E.g., "Doc.2pdf.pdf" -> "Doc", "Doc.pdf" -> "Doc"
    """
    # Remove .pdf extension
    name = filename
    if name.lower().endswith('.pdf'):
        name = name[:-4]
    # Remove .Npdf suffix (e.g., .2pdf, .10pdf)
    name = re.sub(r'\.\d+pdf$', '', name, flags=re.IGNORECASE)
    # Also handle cases like "1pdf" at the end without dot
    name = re.sub(r'\d+pdf$', '', name, flags=re.IGNORECASE)
    return name.strip()

def find_and_group_files(directory):
    """
    Find all PDF files matching keywords and group multi-page scans together.
    Returns a dict: {base_name: [list of file paths in page order]}
    """
    all_pdfs = []
    print(f"Scanning {directory} for prioritized files...")

    for root, dirs, files in os.walk(directory):
        # Skip the output directory and .venv
        if 'critical_files' in root or '.venv' in root:
            continue
        for file in files:
            if file.lower().endswith(".pdf"):
                if any(k.lower() in file.lower() for k in KEYWORDS):
                    full_path = os.path.join(root, file)
                    all_pdfs.append(full_path)

    # Group by base document name and directory
    grouped = defaultdict(list)
    for path in all_pdfs:
        directory = os.path.dirname(path)
        filename = os.path.basename(path)
        base_name = get_base_document_name(filename)
        # Use directory + base_name as key to keep documents from same folder together
        key = os.path.join(directory, base_name)
        grouped[key].append(path)

    # Sort pages within each group
    def sort_key(path):
        filename = os.path.basename(path)
        # Extract page number if present
        match = re.search(r'\.?(\d+)pdf\.pdf$', filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        # Base file (no number) comes first
        if not re.search(r'\d+pdf\.pdf$', filename, re.IGNORECASE):
            return 0
        return 999

    for key in grouped:
        grouped[key].sort(key=sort_key)

    return grouped

def clean_ocr_text(text, use_spellcheck=True):
    """
    Apply common OCR error corrections to improve translation quality.
    Includes Spanish spell-checking to fix OCR artifacts.
    """
    cleaned = text

    # Step 1: Apply known OCR substitution patterns
    for pattern, replacement in OCR_CORRECTIONS.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

    # Fix common "V" used as "y" (and) - but only between lowercase words
    cleaned = re.sub(r'(\s)V(\s)', r'\1y\2', cleaned)

    # Step 2: Spanish spell-check correction
    if use_spellcheck:
        cleaned = spanish_spell_correct(cleaned)

    return cleaned


def spanish_spell_correct(text):
    """
    Apply targeted OCR error corrections for Spanish text.
    Uses conservative corrections based on known OCR error patterns,
    NOT general spell-checking which can introduce wrong words.
    """
    # Extended OCR-specific corrections (patterns that are clearly OCR errors)
    # These are based on common typewriter/scan artifacts
    ocr_fixes = [
        # Number-letter confusions
        (r'\b1a\b', 'la'),
        (r'\b1as\b', 'las'),
        (r'\b1os\b', 'los'),
        (r'\be1\b', 'el'),
        (r'\bde1\b', 'del'),
        (r'\bs1\b', 'si'),
        (r'\bn1\b', 'ni'),
        (r'\ba1\b', 'al'),
        (r'\b0\b(?=\s+[aeiou])', 'o'),  # standalone 0 before vowel → o

        # Common OCR letter confusions
        (r'\bquc\b', 'que'),
        (r'\bdcl\b', 'del'),
        (r'\bdcb\b', 'deb'),
        (r'\bccn\b', 'con'),
        (r'\bncn\b', 'non'),
        (r'\blcs\b', 'los'),
        (r'\bcsf\b', 'est'),
        (r'\bostá\b', 'está'),
        (r'\bestá\b', 'está'),  # ensure accent
        (r'\bscr\b', 'ser'),
        (r'\bpcr\b', 'por'),
        (r'\bpGr\b', 'por'),

        # Double letter fixes
        (r'\bcc\b', 'cc'),  # leave as is - could be valid
        (r'\bll\b', 'll'),  # leave as is - valid in Spanish

        # Common word fixes (very conservative - only clear errors)
        (r'\bdebc\b', 'debe'),
        (r'\bdcbe\b', 'debe'),
        (r'\bhacc\b', 'hace'),
        (r'\bmisno\b', 'mismo'),
        (r'\bnisma\b', 'misma'),
        (r'\bnismo\b', 'mismo'),
        (r'\bsobrc\b', 'sobre'),
        (r'\bentrc\b', 'entre'),
        (r'\bticne\b', 'tiene'),
        (r'\bpucde\b', 'puede'),
        (r'\bdonde\b', 'donde'),
        (r'\bcuando\b', 'cuando'),
        (r'\btodos\b', 'todos'),
        (r'\bcada\b', 'cada'),
        (r'\besta\b', 'esta'),
        (r'\beste\b', 'este'),
        (r'\besos\b', 'esos'),
        (r'\besas\b', 'esas'),

        # Legal document specific
        (r'\bnotario\b', 'notario'),
        (r'\bnotaric\b', 'notario'),
        (r'\bescritura\b', 'escritura'),
        (r'\bfinca\b', 'finca'),
        (r'\bpropiedad\b', 'propiedad'),
        (r'\bpropictario\b', 'propietario'),
        (r'\binmucble\b', 'inmueble'),
        (r'\binmucbles\b', 'inmuebles'),
        (r'\bhipotcca\b', 'hipoteca'),
        (r'\btcstamento\b', 'testamento'),
        (r'\bheredcro\b', 'heredero'),
        (r'\bplanilla\b', 'planilla'),
        (r'\bplanille\b', 'planilla'),

        # Month names (common in dates)
        (r'\bEncro\b', 'Enero'),
        (r'\bFcbrero\b', 'Febrero'),
        (r'\bMarzo\b', 'Marzo'),
        (r'\bAbril\b', 'Abril'),
        (r'\bMayo\b', 'Mayo'),
        (r'\bJunio\b', 'Junio'),
        (r'\bJulio\b', 'Julio'),
        (r'\bAgosto\b', 'Agosto'),
        (r'\bScptiembre\b', 'Septiembre'),
        (r'\bOctubre\b', 'Octubre'),
        (r'\bNovicmbre\b', 'Noviembre'),
        (r'\bDicicmbre\b', 'Diciembre'),
    ]

    corrected = text
    for pattern, replacement in ocr_fixes:
        corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

    return corrected


def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def clean_translation(text):
    """
    Remove repetitive artifacts from machine translation output.
    MarianMT sometimes gets stuck in loops producing repeated phrases.
    """
    from collections import Counter

    words = text.split()
    if len(words) < 20:
        return text

    # Method 1: Detect exact repeated phrases
    for phrase_len in [6, 5, 4, 3]:
        phrases = []
        for i in range(len(words) - phrase_len + 1):
            phrase = " ".join(words[i:i + phrase_len])
            phrases.append((phrase, i))

        phrase_counts = Counter([p[0] for p in phrases])

        for phrase, count in phrase_counts.items():
            if count >= 4:  # Phrase appears 4+ times
                # Find first occurrence position
                first_pos = None
                for p, pos in phrases:
                    if p == phrase:
                        first_pos = pos
                        break

                if first_pos and first_pos > len(words) // 4:
                    # Truncate before the repetition starts
                    truncated_words = words[:first_pos + phrase_len]
                    result = " ".join(truncated_words)
                    # Find last good sentence break
                    last_period = result.rfind('. ')
                    if last_period > len(result) // 2:
                        return result[:last_period + 1] + "\n\n*[Translation truncated - repetitive output detected]*"
                    return result + "\n\n*[Translation truncated - repetitive output detected]*"

    # Method 2: Detect "of the X of the X" patterns specifically
    pattern_text = " ".join(words)
    repetitive_patterns = [
        r'(of the \w+ of the \w+\s*){3,}',
        r'(the totals? of\s*){3,}',
        r'(\w+ of the \w+\s*){5,}',
    ]

    import re
    for pattern in repetitive_patterns:
        match = re.search(pattern, pattern_text)
        if match:
            start_pos = match.start()
            if start_pos > len(pattern_text) // 4:
                # Find sentence break before repetition
                truncated = pattern_text[:start_pos]
                last_period = truncated.rfind('. ')
                if last_period > len(truncated) // 2:
                    return truncated[:last_period + 1] + "\n\n*[Translation truncated - repetitive output detected]*"
                return truncated.rstrip() + "\n\n*[Translation truncated - repetitive output detected]*"

    return text


def classify_document(filename, text):
    """
    Classify document type based on filename and content.
    Returns (document_type, confidence)
    """
    combined = (filename + " " + text).lower()

    scores = {}
    for doc_type, keywords in DOCUMENT_TYPES.items():
        score = sum(1 for kw in keywords if kw.lower() in combined)
        if score > 0:
            scores[doc_type] = score

    if not scores:
        return ("Unclassified Legal Document", "Low")

    best_type = max(scores, key=scores.get)
    confidence = "High" if scores[best_type] >= 3 else "Medium" if scores[best_type] >= 2 else "Low"

    return (best_type, confidence)


def extract_parties(text):
    """
    Extract names of parties/persons mentioned in the document.
    """
    parties = set()

    # Look for known family names with context
    for family in KNOWN_FAMILIES:
        # Pattern: First name + Family name or Family name + First name
        patterns = [
            rf'\b([A-Z][a-záéíóúñ]+)\s+{family}\b',
            rf'\b{family}\s+([A-Z][a-záéíóúñ]+)\b',
            rf'\b([A-Z][a-záéíóúñ]+)\s+[A-Z][a-záéíóúñ]+\s+{family}\b',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                if len(match) > 2:  # Skip very short matches
                    # Reconstruct full name from context
                    full_pattern = rf'\b[A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)*\s*{family}(?:\s+[A-Z][a-záéíóúñ]+)*\b'
                    full_matches = re.findall(full_pattern, text, re.IGNORECASE)
                    for fm in full_matches:
                        if len(fm) > 5:
                            parties.add(fm.strip())

    # Also look for common title patterns
    title_patterns = [
        r'\b(?:Don|Doña|Sr\.|Sra\.|Señor|Señora)\s+([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,3})',
        r'\b(?:el señor|la señora)\s+([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){1,3})',
    ]
    for pattern in title_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match) > 5:
                parties.add(match.strip())

    return list(parties)[:10]  # Limit to top 10


def extract_properties(text, filename):
    """
    Extract property/land references from document.
    """
    properties = set()

    # Check for known property names
    combined = filename + " " + text
    for prop in KNOWN_PROPERTIES:
        if prop.lower() in combined.lower():
            properties.add(prop)

    # Look for "Finca X" patterns
    finca_pattern = r'\b[Ff]inca\s+(?:denominada\s+)?["\']?([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)?)["\']?'
    matches = re.findall(finca_pattern, text)
    for match in matches:
        if len(match) > 2:
            properties.add(f"Finca {match}")

    # Look for "Hacienda X" patterns
    hacienda_pattern = r'\b[Hh]acienda\s+(?:denominada\s+)?["\']?([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+)?)["\']?'
    matches = re.findall(hacienda_pattern, text)
    for match in matches:
        if len(match) > 2:
            properties.add(f"Hacienda {match}")

    return list(properties)


def extract_legal_references(text):
    """
    Extract legal references like notary info, registry numbers, protocol numbers.
    """
    references = {}

    # Notary pattern
    notary_pattern = r'[Nn]otario[:\s]+(?:público\s+)?(?:de\s+)?([A-Z][a-záéíóúñ]+(?:\s+[A-Z][a-záéíóúñ]+){0,3})'
    match = re.search(notary_pattern, text)
    if match:
        references["Notary"] = match.group(1).strip()

    # Protocol number
    protocol_pattern = r'(?:número|no\.?|#)\s*(\d+)\s*(?:del\s+)?protocolo'
    match = re.search(protocol_pattern, text, re.IGNORECASE)
    if match:
        references["Protocol Number"] = match.group(1)

    # Registry reference
    registry_pattern = r'[Rr]egistro\s+de\s+(?:la\s+)?[Pp]ropiedad\s+(?:de\s+)?([A-Za-záéíóúñ\s]+?)(?:,|\.|al)'
    match = re.search(registry_pattern, text)
    if match:
        references["Property Registry"] = match.group(1).strip()

    return references


def assess_legal_relevance(doc_type, parties, properties, text):
    """
    Assess the legal relevance of document for land claim purposes.
    Returns (relevance_level, reasoning)
    """
    score = 0
    reasons = []

    # High-value document types
    high_value_types = ["Last Will and Testament", "Purchase & Sale Agreement",
                        "Property Deed", "Property Partition", "Heir Documentation"]
    medium_value_types = ["Power of Attorney", "Land Boundary Survey",
                          "Property Transfer", "Mortgage Document"]

    if doc_type in high_value_types:
        score += 3
        reasons.append(f"Document type '{doc_type}' is directly relevant to ownership claims")
    elif doc_type in medium_value_types:
        score += 2
        reasons.append(f"Document type '{doc_type}' may support ownership chain")
    else:
        score += 1
        reasons.append(f"Document type '{doc_type}' provides contextual information")

    # Check for key family names
    key_families = ["Ceresa", "Rodriguez", "Queral"]
    found_families = [p for p in parties if any(f.lower() in p.lower() for f in key_families)]
    if found_families:
        score += 2
        reasons.append(f"References key family members: {', '.join(found_families[:3])}")

    # Check for key properties
    key_props = ["Villa Aurelia", "Aguaras"]
    found_props = [p for p in properties if any(k.lower() in p.lower() for k in key_props)]
    if found_props:
        score += 2
        reasons.append(f"References target properties: {', '.join(found_props)}")

    # Determine relevance level
    if score >= 5:
        level = "CRITICAL"
    elif score >= 3:
        level = "HIGH"
    elif score >= 2:
        level = "MEDIUM"
    else:
        level = "LOW"

    return (level, reasons)

def extract_metadata(text):
    """
    Extracts Date and Location using Regex patterns common in Spanish legal docs.
    More robust patterns that handle OCR artifacts and case variations.
    """
    metadata = {
        "Date": "Unknown Date",
        "Location": "Unknown Location"
    }

    # Multiple date patterns to catch various formats
    date_patterns = [
        # "veinticinco dias del mes de Febrero de Mil Novecientos Sesenta"
        r"(?:a los )?(.{1,30}(?:dias?|día) del mes de .{1,20} de .{1,35})",
        # "25 de Febrero de 1960"
        r"(\d{1,2}\s+de\s+\w+\s+de\s+\d{4})",
        # "Febrero 25, 1960" or "Febrero 25 de 1960"
        r"(\w+\s+\d{1,2},?\s+(?:de\s+)?\d{4})",
        # Just year mentions like "mil novecientos treinta y tres"
        r"(mil novecientos .{1,30})",
    ]

    for pattern in date_patterns:
        date_match = re.search(pattern, text, re.IGNORECASE)
        if date_match:
            metadata["Date"] = date_match.group(1).strip()
            break

    # Multiple location patterns (case-insensitive)
    loc_patterns = [
        # "En la ciudad de la Habana"
        r"[Ee]n la ciudad de (?:la )?(\w+(?:\s+\w+)?)",
        # "En Puerto Padre"
        r"[Ee]n (\w+(?:\s+\w+)?),?\s*(?:provincia|partido|oriente|occidente)",
        # "Puerto Padre, Oriente"
        r"(\w+(?:\s+\w+)?),\s*(?:Oriente|Occidente|Habana|Camaguey|Las Villas)",
        # Notary location "Notario de X"
        r"[Nn]otario (?:que fue )?de (?:este )?(?:Distrito de )?(\w+(?:\s+\w+)?)",
    ]

    for pattern in loc_patterns:
        loc_match = re.search(pattern, text)
        if loc_match:
            location = loc_match.group(1).strip()
            # Skip generic words
            if location.lower() not in ['este', 'esta', 'el', 'la', 'los', 'las']:
                metadata["Location"] = location
                break

    return metadata

def translate_in_chunks(translator, text, chunk_size=1200, overlap=100):
    """
    Translate long text in overlapping chunks to preserve context.
    MarianMT works best with ~512 tokens, roughly 1200-1500 chars for Spanish.
    """
    if len(text) <= chunk_size:
        result = translator(text, max_length=512)
        return result[0]['translation_text']

    translations = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence end within last 200 chars of chunk
            last_period = text.rfind('.', end - 200, end)
            if last_period > start:
                end = last_period + 1

        chunk = text[start:end]

        try:
            result = translator(chunk, max_length=512)
            translations.append(result[0]['translation_text'])
        except Exception as e:
            translations.append(f"[Translation error: {str(e)[:50]}]")

        # Move start, accounting for overlap on subsequent chunks
        if start == 0:
            start = end
        else:
            start = end - overlap if end < len(text) else end

    return " ".join(translations)

def process_document_group(doc_name, file_paths, reader, translator):
    """
    Process a group of files representing a single multi-page document.
    Returns the generated report content or None if processing fails.
    """
    all_ocr_text = []
    all_pages_info = []

    print(f"  Processing {len(file_paths)} page(s)...")

    for page_num, file_path in enumerate(file_paths, 1):
        try:
            # Convert PDF page to image
            images = convert_from_path(file_path, first_page=1, last_page=1)
            if not images:
                print(f"    Page {page_num}: No image extracted")
                continue

            # OCR the page
            image_np = np.array(images[0])
            ocr_result = reader.readtext(image_np, detail=0, paragraph=True)
            page_text = "\n".join(ocr_result)

            if len(page_text) > 20:
                all_ocr_text.append(page_text)
                all_pages_info.append(f"[Page {page_num}]")
                print(f"    Page {page_num}: {len(page_text)} chars extracted")
            else:
                print(f"    Page {page_num}: Insufficient text")

        except Exception as e:
            print(f"    Page {page_num}: Error - {str(e)[:50]}")

    if not all_ocr_text:
        return None

    # Combine all pages
    combined_spanish = "\n\n".join(all_ocr_text)

    # Apply OCR corrections
    cleaned_spanish = clean_ocr_text(combined_spanish)

    # Extract metadata from cleaned text
    print(f"  Extracting metadata and legal entities...")
    meta = extract_metadata(cleaned_spanish)

    # Classify document type
    doc_type, type_confidence = classify_document(doc_name, cleaned_spanish)

    # Extract parties
    parties = extract_parties(cleaned_spanish)

    # Extract properties
    properties = extract_properties(cleaned_spanish, doc_name)

    # Extract legal references
    legal_refs = extract_legal_references(cleaned_spanish)

    # Translate (using chunked translation for longer documents)
    print(f"  Translating {len(cleaned_spanish)} chars...")
    english_text = translate_in_chunks(translator, cleaned_spanish)

    # Clean translation of repetition artifacts
    english_text = clean_translation(english_text)

    # Assess legal relevance first (needed for summary)
    relevance_level, relevance_reasons = assess_legal_relevance(doc_type, parties, properties, cleaned_spanish)

    # Generate structured executive summary (template-based, not AI-generated)
    print(f"  Generating executive summary...")
    summary_parts = []

    # Document type description
    summary_parts.append(f"This is a **{doc_type}**")

    # Date/Location if known
    date_loc_parts = []
    if meta['Date'] != "Unknown Date":
        date_loc_parts.append(f"dated **{meta['Date']}**")
    if meta['Location'] != "Unknown Location":
        date_loc_parts.append(f"from **{meta['Location']}**")
    if date_loc_parts:
        summary_parts.append(", ".join(date_loc_parts))

    # Parties
    if parties:
        if len(parties) == 1:
            summary_parts.append(f"involving **{parties[0]}**")
        elif len(parties) == 2:
            summary_parts.append(f"involving **{parties[0]}** and **{parties[1]}**")
        else:
            summary_parts.append(f"involving **{parties[0]}**, **{parties[1]}**, and {len(parties)-2} other party/parties")

    # Properties
    if properties:
        props_str = ", ".join([f"**{p}**" for p in properties[:3]])
        summary_parts.append(f"concerning {props_str}")

    # Build the summary sentence
    executive_summary = " ".join(summary_parts) + "."

    # Add relevance statement
    relevance_descriptions = {
        "CRITICAL": "\n\nThis document appears to be **directly relevant** to establishing property ownership or inheritance rights and should be prioritized for legal review.",
        "HIGH": "\n\nThis document contains information that **may support** property claims or help establish the chain of ownership.",
        "MEDIUM": "\n\nThis document provides **supporting context** but may not directly establish ownership rights.",
        "LOW": "\n\nThis document has **limited direct relevance** to property claims but may provide historical or procedural context."
    }
    executive_summary += relevance_descriptions.get(relevance_level, "")

    # Add note about OCR quality if text is very garbled
    ocr_error_indicators = ['ccn', 'dcb', 'ncn', 'lcs', 'quc', 'cst', 'ostá', 'ccnt']
    error_count = sum(1 for indicator in ocr_error_indicators if indicator in cleaned_spanish.lower())
    if error_count >= 3:
        executive_summary += "\n\n⚠️ *Note: Source document shows significant OCR degradation. Manual review of original recommended.*"

    # Generate comprehensive legal report
    print(f"  Generating legal analysis report...")

    # Format file list for multi-page documents
    if len(file_paths) > 1:
        files_list = "\n".join([f"  - `{os.path.basename(f)}`" for f in file_paths])
        file_section = f"**Source Files** ({len(file_paths)} pages):\n{files_list}"
    else:
        file_section = f"**Source File**: `{file_paths[0]}`"

    # Format parties list
    if parties:
        parties_section = "\n".join([f"- {p}" for p in parties])
    else:
        parties_section = "- No parties identified (manual review recommended)"

    # Format properties list
    if properties:
        properties_section = "\n".join([f"- {p}" for p in properties])
    else:
        properties_section = "- No specific properties identified"

    # Format legal references
    if legal_refs:
        refs_section = "\n".join([f"- **{k}**: {v}" for k, v in legal_refs.items()])
    else:
        refs_section = "- No formal legal references extracted"

    # Format relevance reasoning
    relevance_section = "\n".join([f"- {r}" for r in relevance_reasons])

    # OCR display (expanded for legal documents)
    ocr_display_limit = min(3000, len(cleaned_spanish))
    ocr_truncated = cleaned_spanish[:ocr_display_limit]
    if len(cleaned_spanish) > ocr_display_limit:
        ocr_truncated += f"\n\n... [truncated, {len(cleaned_spanish) - ocr_display_limit} more chars]"

    # Build comprehensive report
    report_content = f"""# LEGAL DOCUMENT ANALYSIS REPORT

---

## Document Identification

| Field | Value |
|-------|-------|
| **Document Name** | {doc_name} |
| **Document Type** | {doc_type} |
| **Classification Confidence** | {type_confidence} |
| **Legal Relevance** | **{relevance_level}** |
| **Date** | {meta['Date']} |
| **Location** | {meta['Location']} |
| **Pages** | {len(file_paths)} |

{file_section}

---

## Executive Summary

{executive_summary}

---

## Legal Analysis

### Relevance Assessment: {relevance_level}

{relevance_section}

### Parties Identified

{parties_section}

### Properties Referenced

{properties_section}

### Legal References

{refs_section}

---

## Document Content

### Original Text (Spanish - OCR Extracted)

```
{ocr_truncated}
```

### English Translation

{english_text}


"""
    return report_content


def main():
    # 1. Setup Directories
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 2. Initialize Models
    print("=" * 60)
    print("CUBAN LAND DOCUMENT ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[1/3] Initializing EasyOCR (Spanish)...")
    reader = easyocr.Reader(['es'], gpu=False)

    print("[2/3] Initializing Translation Model (es->en)...")
    translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

    # 3. Find and Group Files
    print("\n[3/3] Scanning for documents...")
    grouped_docs = find_and_group_files(TARGET_DIR)

    total_docs = len(grouped_docs)
    total_files = sum(len(files) for files in grouped_docs.values())
    print(f"\nFound {total_docs} unique documents ({total_files} total files)")
    print("=" * 60)

    # 4. Process each document group
    success_count = 0
    fail_count = 0

    for i, (doc_key, file_paths) in enumerate(grouped_docs.items(), 1):
        doc_name = os.path.basename(doc_key)
        print(f"\n[{i}/{total_docs}] {doc_name}")

        try:
            report = process_document_group(doc_name, file_paths, reader, translator)

            if report:
                # Save report
                output_filename = f"{doc_name}.md"
                output_path = os.path.join(OUTPUT_DIR, output_filename)

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report)

                print(f"  -> SUCCESS: {output_filename}")
                success_count += 1
            else:
                print(f"  -> SKIPPED: No text extracted")
                fail_count += 1

        except Exception as e:
            print(f"  -> FAILED: {str(e)}")
            fail_count += 1

    # 5. Summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {success_count}")
    print(f"Documents failed/skipped: {fail_count}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
