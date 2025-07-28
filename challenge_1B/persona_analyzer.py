"""
Enhanced persona analyzer for travel planning and other domains
"""

import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class PersonaAnalyzer:
    def __init__(self):
        # Enhanced domain keywords including travel planning
        self.domain_keywords = {
            'travel': ['travel', 'trip', 'vacation', 'holiday', 'journey', 'destination', 
                      'itinerary', 'tour', 'visit', 'explore', 'adventure', 'tourism',
                      'accommodation', 'hotel', 'flight', 'transport', 'sightseeing'],
            'research': ['research', 'study', 'analysis', 'methodology', 'experiment', 'hypothesis', 
                        'literature', 'review', 'findings', 'results', 'conclusion', 'data'],
            'business': ['revenue', 'profit', 'market', 'strategy', 'investment', 'financial', 
                        'performance', 'growth', 'analysis', 'trends', 'competitive'],
            'education': ['learning', 'concepts', 'principles', 'fundamentals', 'examples', 
                         'exercises', 'practice', 'theory', 'application', 'understanding'],
            'technical': ['implementation', 'algorithm', 'system', 'architecture', 'design', 
                         'development', 'framework', 'technology', 'solution'],
        }
        
        # Travel-specific patterns
        self.travel_patterns = {
            'duration': r'(\d+)\s*(?:day|night|week|month)',
            'group_size': r'(?:group of|party of|\b)(\d+)\s*(?:people|person|friend|adult|traveler)',
            'budget': r'(?:budget|spend|cost).*?(\d+)',
            'activity_type': r'(adventure|cultural|relaxation|food|business|family)',
        }
        
    def analyze_persona(self, persona: str, job_to_be_done: str) -> Dict:
        """Enhanced analysis for travel planning and other domains."""
        
        # Extract role and domain
        role_info = self._extract_role_information(persona)
        
        # Extract job requirements with travel-specific parsing
        job_requirements = self._extract_job_requirements(job_to_be_done)
        
        # Enhanced travel planning analysis
        travel_specifics = self._analyze_travel_specifics(job_to_be_done)
        
        # Identify domain focus
        domain_focus = self._identify_domain_focus(persona + " " + job_to_be_done)
        
        # Extract key terms and concepts
        key_terms = self._extract_key_terms(persona + " " + job_to_be_done)
        
        return {
            'role': role_info,
            'job_requirements': job_requirements,
            'travel_specifics': travel_specifics,
            'domain_focus': domain_focus,
            'key_terms': key_terms,
            'expertise_level': self._determine_expertise_level(persona),
            'priority_areas': self._identify_priority_areas(job_to_be_done)
        }
    
    def _extract_role_information(self, persona: str) -> Dict:
        """Extract role-specific information from persona description."""
        
        role_patterns = {
            'travel_planner': r'\b(?:travel planner|travel agent|trip planner|tour guide)\b',
            'student': r'\b(?:student|undergraduate|graduate|phd|master)\b',
            'researcher': r'\b(?:researcher|scientist|investigator|phd|postdoc)\b',
            'analyst': r'\b(?:analyst|analysis|analytical)\b',
            'manager': r'\b(?:manager|director|executive|lead)\b',
            'engineer': r'\b(?:engineer|developer|architect)\b',
            'consultant': r'\b(?:consultant|advisor|specialist)\b'
        }
        
        detected_roles = []
        for role, pattern in role_patterns.items():
            if re.search(pattern, persona.lower()):
                detected_roles.append(role)
        
        domain_patterns = {
            'travel': r'\b(?:travel|tourism|hospitality|trip|vacation)\b',
            'technology': r'\b(?:technology|tech|software|computing|ai|ml)\b',
            'business': r'\b(?:business|management|strategy|marketing)\b'
        }
        
        detected_domains = []
        for domain, pattern in domain_patterns.items():
            if re.search(pattern, persona.lower()):
                detected_domains.append(domain)
        
        return {
            'roles': detected_roles,
            'domains': detected_domains,
            'raw_persona': persona
        }
    
    def _extract_job_requirements(self, job_to_be_done: str) -> Dict:
        """Extract specific requirements from job description."""
        
        task_patterns = {
            'planning': r'\b(?:plan|planning|organize|schedule|arrange)\b',
            'itinerary': r'\b(?:itinerary|schedule|agenda|timeline)\b',
            'research': r'\b(?:research|investigate|study|explore|find)\b',
            'recommendation': r'\b(?:recommend|suggest|advice|best|top)\b',
            'analysis': r'\b(?:analy[sz]e|examination|evaluation|assessment)\b',
        }
        
        task_types = []
        for task, pattern in task_patterns.items():
            if re.search(pattern, job_to_be_done.lower()):
                task_types.append(task)
        
        return {
            'task_types': task_types,
            'focus_areas': self._extract_key_terms(job_to_be_done)[:10],
            'urgency': self._determine_urgency(job_to_be_done),
            'scope': self._determine_scope(job_to_be_done)
        }
    
    def _analyze_travel_specifics(self, job_description: str) -> Dict:
        """Extract travel-specific information from job description."""
        
        travel_info = {}
        job_lower = job_description.lower()
        
        # Extract trip duration
        duration_match = re.search(self.travel_patterns['duration'], job_lower)
        if duration_match:
            travel_info['duration'] = int(duration_match.group(1))
        
        # Extract group size
        group_match = re.search(self.travel_patterns['group_size'], job_lower)
        if group_match:
            travel_info['group_size'] = int(group_match.group(1))
        
        # Determine travel type
        travel_types = []
        if 'friend' in job_lower or 'group' in job_lower:
            travel_types.append('group')
        if 'family' in job_lower:
            travel_types.append('family')
        if 'business' in job_lower:
            travel_types.append('business')
        else:
            travel_types.append('leisure')
        
        if travel_types:
            travel_info['travel_types'] = travel_types
        
        return travel_info
    
    def _identify_domain_focus(self, text: str) -> List[str]:
        """Identify which domain areas are most relevant."""
        text_lower = text.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                domain_scores[domain] = score
        
        return sorted(domain_scores.keys(), key=lambda x: domain_scores[x], reverse=True)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text."""
        # Simple tokenization
        tokens = re.findall(r'\b\w+\b', text.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        key_terms = [token for token in tokens if token not in stop_words and len(token) > 2]
        return list(dict.fromkeys(key_terms))  # Remove duplicates
    
    def _determine_expertise_level(self, persona: str) -> str:
        """Determine the expertise level from persona description."""
        persona_lower = persona.lower()
        
        if any(term in persona_lower for term in ['expert', 'senior', 'lead', 'professional']):
            return 'expert'
        elif any(term in persona_lower for term in ['planner', 'coordinator', 'analyst']):
            return 'intermediate'
        else:
            return 'intermediate'
    
    def _identify_priority_areas(self, job_description: str) -> List[str]:
        """Identify priority areas from job description."""
        
        priority_patterns = {
            'accommodation': r'\b(?:hotel|stay|accommodation|lodging)\b',
            'activities': r'\b(?:activities|things to do|attractions)\b',
            'dining': r'\b(?:restaurant|food|cuisine|dining)\b',
            'transportation': r'\b(?:transport|travel|flight)\b',
            'itinerary': r'\b(?:itinerary|schedule|plan)\b',
            'tips': r'\b(?:tips|advice|recommendation)\b',
        }
        
        priorities = []
        job_lower = job_description.lower()
        
        for area, pattern in priority_patterns.items():
            if re.search(pattern, job_lower):
                priorities.append(area)
        
        return priorities
    
    def _determine_urgency(self, job_description: str) -> str:
        """Determine urgency level."""
        job_lower = job_description.lower()
        if any(term in job_lower for term in ['urgent', 'quick']):
            return 'high'
        return 'medium'
    
    def _determine_scope(self, job_description: str) -> str:
        """Determine scope of work."""
        job_lower = job_description.lower()
        if any(term in job_lower for term in ['comprehensive', 'complete']):
            return 'broad'
        return 'medium'
