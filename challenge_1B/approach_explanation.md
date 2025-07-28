# 📝 Approach Explanation: Persona-Driven Document Intelligence

---

## 🚀 Overview

This system is designed as a **universal document analyst**, capable of extracting and prioritizing the most relevant sections from any collection of PDFs, tailored to a specific persona and their job-to-be-done. The approach is:
- **Fully data-driven**
- **Robust and generalizable** (no domain-specific hardcoding)
- **Adaptable to any document type, persona, or task**

---

## 1️⃣ PDF Parsing & Section Extraction

- **PyMuPDF** is used to parse each PDF, extracting text and structure page by page.
- **Heading detection** is based on logical rules:
  - Title-case
  - 2–4 words
  - Under 40 characters
  - Not a bullet, measurement, or colon
- **Section content** is all lines after a heading until the next heading or blank line—even across multiple pages.
- **Result:** Only meaningful section titles (e.g., chapter names, report sections, topic headings) are considered, and multi-page content blocks are never missed.

---

## 2️⃣ Persona & Task-Aware Filtering

- **Sections are filtered for persona/job relevance.**
  - The system can be configured for any persona (e.g., researcher, student, analyst, contractor) and any job-to-be-done (e.g., literature review, financial analysis, study guide creation, event planning).
  - Filtering logic is easily extensible: keyword lists and scoring rules can be adapted for any domain or requirement.

---

## 3️⃣ Snippet Extraction & Formatting

- For each relevant section, the most informative snippet is extracted for the `refined_text` field:
  - Prefer paragraphs or content blocks that best summarize or answer the persona's needs.
  - Fallback: first two non-empty paragraphs or first 500 characters of content.
- **Snippets are cleaned for readability:**
  - Remove extraneous formatting or repeated labels
  - Join lines into a coherent summary
  - Trim to a reasonable length, ending at a logical break
- **Result:** Every output field is filled with a meaningful, well-formatted summary or content snippet, regardless of document type.

---

## 4️⃣ Ranking & Output Structure

- **One-section-per-PDF cap** and **top-N unique sections** are enforced for diversity and relevance.
- **Output JSON** strictly matches the required structure:
  - All fields filled, no empty values
  - `extracted_sections` and `subsection_analysis` are aligned by index
  - Metadata block includes all required fields

---

## 5️⃣ Dockerization & Reproducibility

- **Containerized with Docker** for reproducibility and easy deployment.
- All dependencies are installed in the Docker image.
- Input/output directories are mounted for seamless data exchange.
- **Runs on CPU only**, with model size and runtime well within challenge constraints.

---

## 🌟 Conclusion

This approach is:
- **Robust and domain-agnostic**
- **Generalizes to any document collection, persona, or job-to-be-done**
- **Produces high-quality, relevant outputs for any use case**
- **Ready for production or hackathon deployment**
- **Easily extensible for new personas, domains, or extraction tasks** 