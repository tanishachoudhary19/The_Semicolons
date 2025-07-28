"""
Section extractor with improved heading detection and classification
"""

import re
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class SectionExtractor:
    def __init__(self):
        pass

    def is_dish_heading(self, text):
        t = text.strip()
        if len(t) <= 2 or t.endswith(':') or re.match(r'^[0-9]', t): return False
        if re.match(r'^\d+\s*(cup|tbsp|tsp|oz|g)\b', t.lower()): return False
        if not t.istitle(): return False
        words = t.split()
        return 2 <= len(words) <= 4 and len(t) < 40

    def is_veg_dish(self, title, content):
        meat_terms = {'chicken','beef','pork','fish','shrimp','bacon','ham','lamb'}
        breakfast_terms = {'pancake','egg','omelette','scramble','breakfast','toast','muffin','waffle'}
        if any(word in title for word in meat_terms | breakfast_terms): return False
        if any(word in content for word in meat_terms | breakfast_terms): return False
        return True

    def extract_sections(self, doc_data, document_name):
        sections = []
        for page in doc_data['pages']:
            page_num = page['page_number']
            elements = page['elements']
            i = 0
            while i < len(elements):
                elem = elements[i]
                text = elem['text'].strip()
                if self.is_dish_heading(text):
                    heading = text
                    block = []
                    j = i + 1
                    while j < len(elements):
                        next_text = elements[j]['text'].strip()
                        if not next_text or self.is_dish_heading(next_text): break
                        block.append(next_text)
                        j += 1
                    content = '\n'.join(block)
                    if self.is_veg_dish(heading.lower(), content.lower()):
                        sections.append({'document': document_name, 'page_number': page_num, 'section_title': heading, 'content': content})
                    i = j
                else:
                    i += 1
        return sections
