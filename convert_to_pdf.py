#!/usr/bin/env python3
"""Convert markdown files to PDF for distribution."""

import os
import markdown
from weasyprint import HTML, CSS
from pathlib import Path

# Directories
SOURCE_DIR = "/Users/juanceresa/Desktop/cs/translations/critical_files_claude"
DIST_DIR = "/Users/juanceresa/Desktop/cs/translations/distribution"

# Folders to process
FOLDERS = ["Critical Importance", "High Importance", "Medium Importance", "Low Importance"]

# CSS for nice PDF styling
CSS_STYLE = """
@page {
    margin: 1in;
    size: letter;
}
body {
    font-family: Georgia, serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #333;
}
h1 {
    font-size: 18pt;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
h2 {
    font-size: 14pt;
    color: #2a2a2a;
    margin-top: 25px;
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
}
h3 {
    font-size: 12pt;
    color: #3a3a3a;
    margin-top: 20px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}
th {
    background-color: #f5f5f5;
    font-weight: bold;
}
blockquote {
    border-left: 4px solid #ccc;
    margin: 15px 0;
    padding-left: 15px;
    color: #555;
    font-style: italic;
}
code {
    background-color: #f4f4f4;
    padding: 2px 5px;
    font-family: monospace;
    font-size: 10pt;
}
pre {
    background-color: #f4f4f4;
    padding: 10px;
    overflow-x: auto;
    font-size: 9pt;
}
hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 20px 0;
}
"""

def convert_md_to_pdf(md_path, pdf_path):
    """Convert a markdown file to PDF."""
    try:
        # Read markdown content
        with open(md_path, 'r', encoding='utf-8') as f:
            md_content = f.read()

        # Convert to HTML
        html_content = markdown.markdown(
            md_content,
            extensions=['tables', 'fenced_code']
        )

        # Wrap in full HTML document
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Create PDF
        HTML(string=full_html).write_pdf(
            pdf_path,
            stylesheets=[CSS(string=CSS_STYLE)]
        )
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    total = 0
    success = 0

    for folder in FOLDERS:
        source_folder = os.path.join(SOURCE_DIR, folder)
        dist_folder = os.path.join(DIST_DIR, folder)

        # Create distribution folder if it doesn't exist
        os.makedirs(dist_folder, exist_ok=True)

        if not os.path.exists(source_folder):
            print(f"Skipping {folder} - source folder not found")
            continue

        # Find all .md files
        md_files = [f for f in os.listdir(source_folder) if f.endswith('.md')]
        print(f"\n{folder}: {len(md_files)} files")

        for md_file in sorted(md_files):
            total += 1
            md_path = os.path.join(source_folder, md_file)
            pdf_file = md_file.replace('.md', '.pdf')
            pdf_path = os.path.join(dist_folder, pdf_file)

            print(f"  Converting: {md_file}", end=" ... ", flush=True)

            if convert_md_to_pdf(md_path, pdf_path):
                print("OK")
                success += 1
            else:
                print("FAILED")

    print(f"\n{'='*50}")
    print(f"Conversion complete: {success}/{total} files converted")
    print(f"PDFs saved to: {DIST_DIR}")

if __name__ == "__main__":
    main()
