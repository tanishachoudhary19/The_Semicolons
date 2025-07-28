import os
import json
import fitz  # PyMuPDF

# By default, use Docker paths. For local runs, set INPUT_DIR and OUTPUT_DIR environment variables or change the defaults below.
INPUT_DIR = os.environ.get("INPUT_DIR", "/app/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/app/output")

def extract_title(doc):
    # Try PDF metadata first
    title = doc.metadata.get("title")
    if title and title.strip():
        return title.strip()
    # Fallback: largest text in the whole document
    max_size = 0
    title_text = ""
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if span["size"] > max_size and len(span["text"].strip()) > 5:
                        max_size = span["size"]
                        title_text = span["text"].strip()
    return title_text if title_text else "Untitled Document"

def extract_headings(doc):
    # 1. Try PDF outline/bookmarks first
    toc = doc.get_toc()
    if toc:
        headings = []
        for item in toc:
            level, title, page = item
            if 1 <= level <= 4:
                headings.append({
                    "level": f"H{level}",
                    "text": title.strip(),
                    "page": page
                })
        if len(headings) > 2:
            return headings

    # 2. Visual heuristics (no hardcoded headlines)
    headings = []
    font_stats = {}
    all_font_sizes = []
    lines_by_page = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        page_lines = []
        for block in blocks:
            for line in block.get("lines", []):
                line_text = ""
                max_size = 0
                font_name = ""
                is_bold = False
                y0 = None
                x0 = None
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text:
                        continue
                    line_text += text + " "
                    if span["size"] > max_size:
                        max_size = span["size"]
                        font_name = span.get("font", "")
                        is_bold = "Bold" in font_name or "bold" in font_name
                        y0 = span["bbox"][1]
                        x0 = span["bbox"][0]
                line_text = line_text.strip()
                if line_text:
                    page_lines.append((line_text, max_size, font_name, is_bold, y0, x0))
                    font_stats[max_size] = font_stats.get(max_size, 0) + 1
                    all_font_sizes.append(max_size)
        lines_by_page.append(page_lines)

    # Compute median font size (body text)
    if all_font_sizes:
        median_size = sorted(all_font_sizes)[len(all_font_sizes)//2]
    else:
        median_size = 0

    # Assign heading levels to top 4 font sizes
    sorted_sizes = sorted(font_stats.keys(), reverse=True)
    max_levels = min(4, len(sorted_sizes))
    size_to_level = {}
    for i, size in enumerate(sorted_sizes[:max_levels]):
        size_to_level[size] = f"H{i+1}"

    # Merge lines that are very close vertically and have the same font size and style
    merged_lines_by_page = []
    for page_lines in lines_by_page:
        merged = []
        prev = None
        for line in page_lines:
            line_text, size, font, bold, y0, x0 = line
            if (
                prev is not None
                and size == prev[1]
                and font == prev[2]
                and abs(y0 - prev[4]) < 8
                and abs(x0 - prev[5]) < 20
                and len(prev[0]) < 60
            ):
                prev = (prev[0] + " " + line_text, size, font, bold, y0, x0)
            else:
                if prev is not None:
                    merged.append(prev)
                prev = (line_text, size, font, bold, y0, x0)
        if prev is not None:
            merged.append(prev)
        merged_lines_by_page.append(merged)

    seen = set()
    for page_num, page_lines in enumerate(merged_lines_by_page):
        for line_text, size, font_name, is_bold, y0, x0 in page_lines:
            if len(line_text) < 5 or len(line_text.split()) < 2:
                continue
            if line_text.lower() in seen:
                continue
            level = size_to_level.get(size)
            is_centered = x0 is not None and 100 < x0 < 300  # adjust as needed
            is_all_caps = line_text.isupper()
            is_much_larger = size > median_size * 1.15  # 15% larger than median
            # Accept if bold, all-caps, centered, or much larger than body text
            if level and (is_bold or is_all_caps or is_centered or is_much_larger):
                if not any(line_text in other and line_text != other for other in seen):
                    headings.append({
                        "level": level,
                        "text": line_text,
                        "page": page_num + 1
                    })
                    seen.add(line_text.lower())
    return headings

def process_pdf(pdf_path, output_path):
    doc = fitz.open(pdf_path)
    title = extract_title(doc)
    outline = extract_headings(doc)
    result = {
        "title": title,
        "outline": outline
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"ERROR: Input directory '{INPUT_DIR}' does not exist.")
        print("Please create it and add PDF files, or set the INPUT_DIR environment variable.")
        return
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)
            print(f"Processing {filename}...")
            process_pdf(pdf_path, output_path)
    print("Done.")

if __name__ == "__main__":
    main()