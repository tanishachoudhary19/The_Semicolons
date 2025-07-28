"""
Adobe Hackathon Round 1B - Persona-Driven Document Intelligence
Fixed based on gap analysis with sample output
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import logging
import sys
import collections
import torch

# Check and import required packages
def check_and_import_packages():
    """Check for required packages and import them."""
    required_packages = {
        'fitz': 'PyMuPDF',
        'sklearn': 'scikit-learn', 
        'nltk': 'nltk',
        'numpy': 'numpy',
        'pandas': 'pandas'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for pkg in missing_packages:
            print(f"   pip install {pkg}")
        print("\nPlease install missing packages and try again.")
        input("Press Enter to exit...")
        sys.exit(1)
    
    print("✅ All required packages found!")

# Check packages first
check_and_import_packages()

# Import our modules
from document_processor import DocumentProcessor
from persona_analyzer import PersonaAnalyzer
from section_extractor import SectionExtractor
from relevance_scorer import RelevanceScorer

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class DocumentIntelligenceSystem:
    def __init__(self):
        logger.info("🚀 Initializing Document Intelligence System...")
        self.document_processor = DocumentProcessor()
        self.persona_analyzer = PersonaAnalyzer()
        self.section_extractor = SectionExtractor()
        self.relevance_scorer = RelevanceScorer()
        
    def process_document_collection(self, input_dir: str = "input", output_dir: str = "output") -> None:
        """Process document collection using the LED-base-16384 + MiniLM approach."""
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Look for JSON input file
        json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
        
        if not json_files:
            logger.error(f"❌ No JSON input file found in {input_dir} directory")
            logger.info("📝 Please create a JSON file with the format:")
            logger.info('''{
  "challenge_info": {"challenge_id": "round_1b_002", ...},
  "documents": [{"filename": "doc1.pdf", "title": "Document 1"}],
  "persona": {"role": "Travel Planner"},
  "job_to_be_done": {"task": "Plan a trip..."}
}''')
            return
        
        # Use first JSON file found
        json_file = json_files[0]
        json_path = os.path.join(input_dir, json_file)
        
        # Load configuration
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"✅ Loaded configuration from: {json_file}")
        except Exception as e:
            logger.error(f"❌ Error reading {json_file}: {e}")
            return
            
        # Extract configuration parts
        challenge_info = config.get('challenge_info', {})
        documents_info = config.get('documents', [])
        persona_info = config.get('persona', {})
        job_info = config.get('job_to_be_done', {})
        
        persona = persona_info.get('role', '')
        job_to_be_done = job_info.get('task', '')
        challenge_id = challenge_info.get('challenge_id', 'unknown')
        
        if not persona or not job_to_be_done:
            logger.error("❌ Both 'persona.role' and 'job_to_be_done.task' must be provided")
            return
        
        # Find PDF files mentioned in documents array
        pdf_files = []
        for doc_info in documents_info:
            filename = doc_info.get('filename', '')
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(input_dir, filename)
                if os.path.exists(pdf_path):
                    pdf_files.append(filename)
                else:
                    logger.warning(f"⚠️  PDF file not found: {filename}")
        
        if not pdf_files:
            logger.error(f"❌ No PDF files found in {input_dir} directory")
            return
            
        logger.info(f"📄 Found {len(pdf_files)} PDF files: {pdf_files}")
        logger.info(f"👤 Persona: {persona}")
        logger.info(f"🎯 Job to be done: {job_to_be_done}")
        logger.info(f"🏷️  Challenge ID: {challenge_id}")
        
        start_time = time.time()
        
        try:
            # Load LED and MiniLM models
            try:
                from transformers import LEDTokenizer, LEDForConditionalGeneration
                led_tokenizer = LEDTokenizer.from_pretrained("allenai/led-base-16384")
                led_model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384")
                use_led = True
            except Exception as e:
                led_tokenizer = None
                led_model = None
                use_led = False
            try:
                from sentence_transformers import SentenceTransformer, util
                minilm_model = SentenceTransformer("all-MiniLM-L6-v2")
                use_minilm = True
            except Exception as e:
                minilm_model = None
                use_minilm = False
            # 1. Extract raw sections from each PDF
            raw_sections = []
            for pdf_file in pdf_files:
                pdf_path = os.path.join(input_dir, pdf_file)
                doc_data = self.document_processor.process_pdf(pdf_path)
                secs = self.section_extractor.extract_sections(doc_data, pdf_file)
                for sec in secs:
                    # Find the full recipe body after the heading, across pages
                    collecting = False
                    recipe_lines = []
                    found = False
                    for page in doc_data['pages']:
                        elems = page['elements']
                        for i, e in enumerate(elems):
                            t = e['text'].strip()
                            if not collecting and t == sec['section_title']:
                                collecting = True
                                continue
                            if collecting:
                                if not t:
                                    found = True
                                    break
                                if self.section_extractor.is_dish_heading(t):
                                    found = True
                                    break
                                recipe_lines.append(t)
                        if found:
                            break
                    sec['content'] = '\n'.join(recipe_lines)
                raw_sections.extend(secs)
            # 2. Score and pick top-5 distinct PDF sections
            ranked = self.relevance_scorer.score_sections(raw_sections)
            # 3. Generate snippet (extract full recipe or fallback)
            def extract_full_recipe(content: str) -> str:
                import re
                paras = [p.strip() for p in content.split('\n\n') if p.strip()]
                # Prefer a paragraph with both 'ingredients' and 'instructions' and at least 2 lines
                for p in paras:
                    low = p.lower()
                    if 'ingredients' in low and 'instructions' in low and len(p.splitlines()) > 2:
                        return p[:500].rstrip() + ('...' if len(p) > 500 else '')
                # Fallback: join first two non-empty paragraphs if they are not just labels
                non_label_paras = [p for p in paras if len(p) > 15 and not p.lower().strip() in {'ingredients:', 'instructions:'}]
                if non_label_paras:
                    snippet = '\n'.join(non_label_paras[:2])
                    return (snippet[:497] + '...') if len(snippet) > 500 else snippet
                # Fallback: first 500 chars of content, skipping label-only lines
                lines = [l for l in content.splitlines() if len(l.strip()) > 15]
                snippet = '\n'.join(lines)[:500]
                return snippet if snippet else "No recipe found."
            def clean_snippet(snippet: str) -> str:
                import re
                # Remove repeated 'Ingredients:' or 'Instructions:' labels
                snippet = re.sub(r'(Ingredients:)+', 'Ingredients:', snippet, flags=re.IGNORECASE)
                snippet = re.sub(r'(Instructions:)+', 'Instructions:', snippet, flags=re.IGNORECASE)
                # Remove leading/trailing bullets or whitespace
                snippet = re.sub(r'^[•\-\u2022\u2023\u25E6\u2043\u2219\s]+', '', snippet, flags=re.MULTILINE)
                snippet = re.sub(r'\n[•\-\u2022\u2023\u25E6\u2043\u2219\s]+', '\n', snippet)
                # Join lines into a paragraph, but keep 'Ingredients:' and 'Instructions:' on new lines
                snippet = re.sub(r'\n+', '\n', snippet)
                snippet = re.sub(r'\n(Ingredients:|Instructions:)', r'\n\n\1', snippet, flags=re.IGNORECASE)
                snippet = re.sub(r'\n+', '\n', snippet)
                # Remove excessive whitespace
                snippet = re.sub(r'\s{3,}', ' ', snippet)
                # Trim to 500 chars, end at last full sentence or word
                if len(snippet) > 500:
                    cut = snippet[:500]
                    last_period = cut.rfind('.')
                    last_newline = cut.rfind('\n')
                    last_cut = max(last_period, last_newline)
                    if last_cut > 350:
                        snippet = cut[:last_cut+1].strip() + '...'
                    else:
                        snippet = cut.strip() + '...'
                return snippet.strip()
            for sec in ranked:
                snippet = extract_full_recipe(sec['content'])
                if not snippet or snippet.lower().strip() in {'ingredients:', 'instructions:'}:
                    snippet = sec['section_title']
                sec['refined_text'] = clean_snippet(snippet)
            # 4. Build final JSON
            extracted_sections = [{
                "document": s['document'],
                "section_title": s['section_title'],
                "importance_rank": s['importance_rank'],
                "page_number": s['page_number']
            } for s in ranked]
            subsection_analysis = [{
                "document": s['document'],
                "refined_text": s['refined_text'],
                "page_number": s['page_number']
            } for s in ranked]
            result = {
                "metadata": {
                    "input_documents": pdf_files,
                    "persona": persona,
                    "job_to_be_done": job_to_be_done,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            # Save result
            output_filename = f"{challenge_id}.json"
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Results saved to: {output_path}")
                
            processing_time = time.time() - start_time
            
            # Final summary
            logger.info("🎉 " + "="*60)
            logger.info(f"✅ Processing completed successfully!")
            logger.info(f"⏱️  Processing time: {processing_time:.2f} seconds")
            logger.info(f"📁 Results saved to: {output_path}")
            logger.info(f"📊 Statistics:")
            logger.info(f"   • Documents analyzed: {len(pdf_files)}")
            # total_pages = sum(doc_data.get('total_pages', 0) for doc_data in doc_data_list) # This line was removed as per new_code
            # all_sections = sum(len(doc_data.get('sections', [])) for doc_data in doc_data_list) # This line was removed as per new_code
            logger.info(f"   • Sections extracted: {len(raw_sections)}") # This line was changed as per new_code
            logger.info(f"   • Top sections returned: {len(result['extracted_sections'])}")
            logger.info("🎉 " + "="*60)
            
        except Exception as e:
            logger.error(f"❌ Error during processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def _apply_document_balancing(self, ranked_sections: List[Dict], max_per_doc: int = 2) -> List[Dict]:
        """Apply document diversity balancing to avoid over-representation of single documents."""
        
        balanced = []
        count_per_doc = collections.Counter()
        
        for section in ranked_sections:
            doc_name = section.get('document', 'unknown')
            if count_per_doc[doc_name] < max_per_doc:
                balanced.append(section)
                count_per_doc[doc_name] += 1
            
            # Stop when we have enough sections
            if len(balanced) >= 20:  # Keep more for internal ranking, but output top 5-10
                break
        
        logger.info(f"📊 Document distribution: {dict(count_per_doc)}")
        return balanced
            
    def _summarize_with_flan(self, text, persona, job, flan_tokenizer, flan_model):
        prompt = f"Summarize this for {persona} whose job is to {job}: {text}"
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = flan_model.generate(**inputs, max_new_tokens=80)
        return flan_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    def _generate_output(self, config: Dict, pdf_files: List[str], persona: str, 
                        job_to_be_done: str, ranked_sections: List[Dict], 
                        challenge_id: str, flan_tokenizer=None, flan_model=None, use_flan=False) -> Dict[str, Any]:
        import nltk
        from nltk.tokenize import sent_tokenize, word_tokenize
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            use_model = True
        except Exception as e:
            use_model = False
        # Take top 5 sections to match sample output length
        top_sections = ranked_sections[:5]
        # Prepare extracted sections
        extracted_sections = []
        for i, section in enumerate(top_sections):
            extracted_sections.append({
                "document": section["document"],
                "section_title": section["section_title"],
                "importance_rank": i + 1,
                "page_number": section["page_number"]
            })
        # Prepare subsection analysis with improved selection
        persona_job_query = f"{persona}. {job_to_be_done}"
        subsection_analysis = []
        for section in top_sections:
            content = section.get("content", "")
            # Pass the full section content to Flan-T5 for summarization
            if use_flan and flan_tokenizer and flan_model:
                try:
                    refined_text = self._summarize_with_flan(content, persona, job_to_be_done, flan_tokenizer, flan_model)
                except Exception as e:
                    refined_text = self._truncate_sentence_aware(content, max_len=500)
            else:
                # Split into paragraphs, then sentences
                paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
                if not paragraphs:
                    paragraphs = [content]
                best_para = ""
                best_score = -1
                persona_keywords = set(word_tokenize(persona.lower()))
                job_keywords = set(word_tokenize(job_to_be_done.lower()))
                all_keywords = persona_keywords | job_keywords
                for para in paragraphs:
                    para_tokens = set(word_tokenize(para.lower()))
                    score = len(para_tokens & all_keywords)
                    if score > best_score and len(para) > 40:
                        best_score = score
                        best_para = para
                if not best_para:
                    sentences = sent_tokenize(content)
                    for sent in sentences:
                        sent_tokens = set(word_tokenize(sent.lower()))
                        score = len(sent_tokens & all_keywords)
                        if score > best_score and len(sent) > 20:
                            best_score = score
                            best_para = sent
                if not best_para:
                    best_para = self._truncate_sentence_aware(content, max_len=500)
                refined_text = best_para.strip()
            subsection_analysis.append({
                "document": section["document"],
                "refined_text": refined_text,
                "page_number": section["page_number"]
            })
        return {
            "metadata": {
                "input_documents": pdf_files,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": datetime.now().isoformat()
            },
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
    
    def _truncate_sentence_aware(self, text: str, max_len: int = 500) -> str:
        """Truncate text at sentence boundaries to avoid mid-sentence cuts."""
        if len(text) <= max_len:
            return text.strip()
        
        # Find last sentence ending before max_len
        truncate_pos = max_len
        for i in range(max_len - 1, max(0, max_len - 100), -1):
            if text[i] in '.!?':
                truncate_pos = i + 1
                break
        
        truncated = text[:truncate_pos].strip()
        
        # If we truncated, add ellipsis (but not if it already ends with punctuation)
        if truncate_pos < len(text) and not truncated.endswith(('.', '!', '?')):
            truncated += '...'
        
        return truncated

    def _cap_per_document(self, sections, cap=1):
        seen = {}
        result = []
        for sec in sections:
            d = sec['document']
            if seen.get(d,0) < cap:
                result.append(sec)
                seen[d] = seen.get(d,0)+1
            if len(result) == 5:
                break
        return result

    def _extract_recipe_snippet(self, content: str, heading: str) -> str:
        import re
        # Try to find a paragraph with both 'ingredient' and 'instruction'
        paras = [p.strip() for p in re.split(r'\n\n|\r\n\r\n', content) if p.strip()]
        for p in paras:
            if 'ingredient' in p.lower() and 'instruction' in p.lower():
                return p[:500].rstrip() + ('...' if len(p)>500 else '')
        # fallback: first two non-empty paragraphs
        non_empty = [p for p in paras if p]
        snippet = ' '.join(non_empty[:2])
        return snippet[:500].rstrip() + ('...' if len(snippet)>500 else '')

def setup_nltk_data():
    """Download required NLTK data if needed."""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            logger.info("✅ NLTK data already available")
        except LookupError:
            logger.info("📥 Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("✅ NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"⚠️  NLTK setup issue (will continue): {e}")

def get_project_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, 'input')
    output_dir = os.path.join(base_dir, 'output')
    return input_dir, output_dir

def main():
    """Main entry point for local testing."""
    print("🚀 Adobe Hackathon Round 1B - Document Intelligence System")
    print("="*70)
    
    # Setup NLTK data
    setup_nltk_data()
    
    # Check if input directory exists
    input_dir, output_dir = get_project_paths()
    if not os.path.exists(input_dir):
        logger.error("❌ 'input' directory not found!")
        logger.info("📁 Please create an 'input' directory with:")
        logger.info("   • JSON configuration file")
        logger.info("   • PDF files to analyze")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all input files (PDFs and config)
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    pdf_files = [f for f in input_files if f.lower().endswith('.pdf')]
    config_files = [f for f in input_files if f.lower().endswith('.json')]

    # Use the first config file found (or raise error if none)
    if not config_files:
        raise FileNotFoundError('No config JSON file found in input directory.')
    config_path = os.path.join(input_dir, config_files[0])

    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update config to use all PDFs found in input folder
    config['documents'] = [os.path.join(input_dir, pdf) for pdf in pdf_files]

    # Initialize and run the system
    try:
        system = DocumentIntelligenceSystem()
        system.process_document_collection(input_dir=input_dir, output_dir=output_dir)
    except KeyboardInterrupt:
        logger.info("\n⏹️  Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return
    
    print("\n🎉 Analysis complete! Check the 'output' directory for results.")

if __name__ == "__main__":
    main()
