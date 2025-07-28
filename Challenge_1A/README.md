# PDF Heading Extraction - Challenge 1A

This project extracts a structured outline (title, H1-H4 headings with page numbers) from PDF documents using a combination of PDF bookmarks, font-size/style heuristics, and regex-based pattern matching.

## Folder Structure

```
Challenge_1a/
├── app/
│   ├── input/           # Place your input PDF files here (for Docker)
│   └── output/          # Output JSON files will be written here (for Docker)
├── input/               # Place your input PDF files here (for local run)
├── output/              # Output JSON files will be written here (for local run)
├── process_pdfs.py      # Main script for processing PDFs
├── Dockerfile           # For containerized execution
├── requirements.txt     # Python dependencies
├── README.md            # This file
```

## How It Works
- **PDF Bookmarks/Outline:** If the PDF has a valid outline (bookmarks), these are used for headings.
- **Font-Size & Style Heuristics:** If not, the script clusters font sizes and uses the largest as H1, next as H2, etc. Bold, italic, all-caps, centered, or much larger text is favored as headings.
- **Regex Patterns:** Numbered headings (e.g., "1. Introduction"), ALL-CAPS, and lines ending with a colon are also considered.
- **Filtering:** Fragments, body text, and lines with too many symbols or numbers are filtered out. Only H1-H4 levels are assigned.
- **Multilingual:** The code is language-agnostic and works for most scripts (including CJK, Devanagari, etc.) as it does not rely on case-based heuristics.

## Usage

### Local Run
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Place your PDFs:**
   - Put all input PDF files in the `input/` directory (create it if it doesn't exist).
3. **Run the script:**
   ```bash
   python process_pdfs.py
   ```
   - Output JSON files will be written to the `output/` directory.

### Docker Run
1. **Build the Docker image:**
   ```bash
   docker build --platform linux/amd64 -t pdf-heading-extractor .
   ```
2. **Run the Docker container:**
   ```bash
   docker run --rm -v ${PWD}/app/input:/app/input:ro -v ${PWD}/app/output:/app/output --network none pdf-heading-extractor
   ```
   - On Windows PowerShell, use `${PWD}` instead of `$(pwd)`.
   - Make sure your PDFs are in `app/input/` and the output will be in `app/output/`.

#### **Troubleshooting Docker**
- If you see no output, check that your input PDFs are in the correct folder and that you are mounting the folders properly.
- If you get permission errors, ensure the output folder exists and is writable.
- Always run the Docker command from the `Challenge_1a` directory.

## Output Format
Each output JSON will look like:
```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Section Title", "page": 1 },
    { "level": "H2", "text": "Subsection Title", "page": 2 },
    ...
  ]
}
```

## Approach
- **PDF Outline/Bookmarks:** Used if available and valid.
- **Font-Size Clustering:** Largest font sizes mapped to H1-H4.
- **Style Heuristics:** Bold, italic, all-caps, centered, or much larger text is favored.
- **Regex Patterns:** Numbered headings, ALL-CAPS, and lines ending with a colon are detected.
- **Filtering:** Removes fragments, body text, and non-heading lines.
- **Multilingual:** No reliance on case-based heuristics, so works for most languages/scripts.

## Dockerfile
A Dockerfile is provided for containerized execution. See above for build/run instructions.

## requirements.txt
```
PyMuPDF
```

## Notes
- No hardcoded heading text or keywords are used; all logic is based on document structure and style.
- The script is robust for a wide variety of PDFs, but results may vary for highly unstructured or scanned documents.

---
For any issues or improvements, please open an issue or pull request. 