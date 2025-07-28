# Persona-Driven Document Intelligence

## 📝 Overview
This project is an intelligent document analysis system that extracts and prioritizes the most relevant sections from a collection of PDFs, tailored to a specific persona and their job-to-be-done. It is designed to be domain-agnostic, robust, and compliant with strict resource and runtime constraints.

## 🚀 Features
- Accepts 3–10 PDFs and a persona/job definition
- Extracts, ranks, and summarizes the most relevant sections
- Outputs a structured JSON file with metadata, extracted sections, and refined snippets
- Generalizes to any document domain or persona
- Fully containerized for reproducibility (Docker)
- CPU-only, model size < 1GB, no internet required at runtime

## 📦 Requirements
- Docker Desktop (Windows, Mac, or Linux)
- At least 4 CPUs and 8GB RAM recommended for best performance

## 🛠️ Build Instructions
1. Clone this repository and open a terminal in the project root.
2. Build the Docker image:
   ```sh
   docker build -t persona-doc-intel .
   ```

## ▶️ Run Instructions
1. Place your input PDFs and config file in the `input/` folder.
2. Run the container using the following command, replacing `<ABSOLUTE_PATH_TO_PROJECT>` with the full path to your project directory:
   
   **Windows (PowerShell):**
   ```sh
   docker run --rm -v <ABSOLUTE_PATH_TO_PROJECT>/input:/app/input -v <ABSOLUTE_PATH_TO_PROJECT>/output:/app/output persona-doc-intel
   ```
   
   **Mac/Linux (Terminal):**
   ```sh
   docker run --rm -v /absolute/path/to/project/input:/app/input -v /absolute/path/to/project/output:/app/output persona-doc-intel
   ```
   
   - Use forward slashes (`/`) in all paths.
   - Make sure to use the absolute path, not a relative one.
   - The output JSON will appear in the `output/` folder.

## 📂 Folder Structure
```
perp/
  input/    # Place your PDFs and config here
  output/   # Output JSON will be written here
  main.py
  section_extractor.py
  relevance_scorer.py
  ...
```

## ⏱️ Runtime & Constraints
- Typical runtime: **~59 seconds** for 3–5 PDFs (measured in Docker)
- Model size: < 1GB
- No internet required at runtime
- Output strictly matches the required JSON schema

## 📄 Deliverables
- `approach_explanation.md` — Methodology (300–500 words)
- `Dockerfile` — Container build instructions
- `requirements.txt` — Python dependencies
- `main.py`, `section_extractor.py`, `relevance_scorer.py`, etc.
- `README.md` — This file

## 👤 Contact / Credits
- Developed by: [Your Name or Team]
- For questions, contact: [your.email@example.com]

---
**Good luck and happy hacking!** 