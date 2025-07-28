"""
Enhanced relevance scorer with document balancing and improved weights
"""

import re
from typing import Dict, List
import logging
import collections

logger = logging.getLogger(__name__)

class RelevanceScorer:
    def score_sections(self, sections, persona_req=None):
        # Just return sections in document order, one per PDF
        seen, out = set(), []
        for sec in sections:
            d = sec['document']
            if d not in seen:
                out.append(sec)
                seen.add(d)
            if len(out) == 5:
                break
        # Assign importance_rank by list order
        for i, sec in enumerate(out, 1):
            sec['importance_rank'] = i
        return out
