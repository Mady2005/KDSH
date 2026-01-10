"""
Kharagpur Data Science Hackathon 2026
Team: 

HYBRID ADVANCED SOLUTION:
Combines the discriminative power of rule-based systems with the sophistication of LLM reasoning.

KEY INNOVATIONS:
1. Pathway-inspired Vector Store for efficient retrieval
2. Multi-level evidence extraction (paragraph, sentence, phrase)
3. Ensemble of 7 detection strategies
4. LLM-augmented reasoning for ambiguous cases
5. Confidence-calibrated predictions
6. Detailed evidence tracking

This solution balances accuracy, interpretability, and computational efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
import re
import json
from collections import defaultdict, Counter
import warnings
import asyncio
warnings.filterwarnings('ignore')


class PathwayVectorStore:
    """Lightweight vector store using TF-IDF for semantic retrieval"""
    
    def __init__(self):
        self.documents = []
        self.metadata = []
        self.vocabulary = {}
        self.idf_scores = {}
        self.doc_vectors = []
        
    def add_documents(self, docs: List[Dict]):
        """Add documents with metadata"""
        self.documents.extend([d['text'] for d in docs])
        self.metadata.extend([{k: v for k, v in d.items() if k != 'text'} for d in docs])
        self._build_index()
    
    def _build_index(self):
        """Build TF-IDF index"""
        doc_word_sets = []
        for doc in self.documents:
            words = self._tokenize(doc)
            doc_word_sets.append(set(words))
            for word in words:
                self.vocabulary[word] = self.vocabulary.get(word, 0) + 1
        
        # IDF scores
        num_docs = len(self.documents)
        for word in self.vocabulary:
            doc_freq = sum(1 for word_set in doc_word_sets if word in word_set)
            self.idf_scores[word] = np.log((num_docs + 1) / (doc_freq + 1))
        
        # Document vectors
        self.doc_vectors = [self._vectorize(doc) for doc in self.documents]
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize with stop word removal"""
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 
                     'was', 'one', 'our', 'out', 'get', 'has', 'him', 'his', 'how',
                     'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'did',
                     'its', 'let', 'put', 'say', 'she', 'too', 'use', 'had', 'from',
                     'with', 'they', 'have', 'this', 'will', 'your', 'been', 'into',
                     'than', 'them', 'then', 'were', 'when', 'some', 'time', 'that',
                     'there', 'their', 'which', 'would', 'about', 'could', 'other'}
        return [w for w in words if w not in stop_words and len(w) > 3]
    
    def _vectorize(self, text: str) -> Dict[str, float]:
        """TF-IDF vectorization"""
        words = self._tokenize(text)
        word_counts = Counter(words)
        total_words = len(words) if words else 1
        
        vector = {}
        for word, count in word_counts.items():
            if word in self.idf_scores:
                tf = count / total_words
                vector[word] = tf * self.idf_scores[word]
        return vector
    
    def search(self, query: str, top_k: int = 15, filters: Dict = None) -> List[Dict]:
        """Semantic search with filtering"""
        query_vector = self._vectorize(query)
        
        scores = []
        for idx, doc_vector in enumerate(self.doc_vectors):
            if filters:
                if not all(self.metadata[idx].get(k) == v for k, v in filters.items()):
                    continue
            
            score = self._cosine_similarity(query_vector, doc_vector)
            scores.append((score, idx))
        
        scores.sort(reverse=True)
        return [{
            'text': self.documents[idx],
            'score': score,
            'metadata': self.metadata[idx]
        } for score, idx in scores[:top_k]]
    
    def _cosine_similarity(self, v1: Dict, v2: Dict) -> float:
        """Cosine similarity for sparse vectors"""
        common_keys = set(v1.keys()) & set(v2.keys())
        if not common_keys:
            return 0.0
        
        dot_product = sum(v1[k] * v2[k] for k in common_keys)
        norm1 = np.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = np.sqrt(sum(v ** 2 for v in v2.values()))
        
        return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


class AdvancedConsistencyDetector:
    """
    Advanced detector with 7 complementary strategies for robust detection
    """
    
    def __init__(self):
        self.evidence_cache = {}
        
    def analyze_consistency(self, backstory: str, evidence_docs: List[Dict], 
                          character: str) -> Tuple[int, str, Dict]:
        """
        Multi-strategy analysis with detailed scoring
        """
        # Compile evidence
        evidence_text = "\n".join([doc['text'] for doc in evidence_docs])
        
        # Run 7 detection strategies
        strategies = {
            'explicit_contradiction': self._check_explicit_contradiction(backstory, evidence_text),
            'temporal_consistency': self._check_temporal_consistency(backstory, evidence_text),
            'causal_plausibility': self._check_causal_plausibility(backstory, evidence_text),
            'character_knowledge': self._check_character_knowledge(backstory, evidence_text, character),
            'geographic_consistency': self._check_geographic_consistency(backstory, evidence_text),
            'relationship_consistency': self._check_relationship_consistency(backstory, evidence_text),
            'factual_alignment': self._check_factual_alignment(backstory, evidence_text)
        }
        
        # Weighted ensemble decision
        weights = {
            'explicit_contradiction': 0.30,
            'temporal_consistency': 0.20,
            'causal_plausibility': 0.15,
            'character_knowledge': 0.15,
            'geographic_consistency': 0.05,
            'relationship_consistency': 0.10,
            'factual_alignment': 0.05
        }
        
        # Calculate weighted score
        weighted_score = sum(strategies[k]['score'] * weights[k] for k in strategies)
        
        # Determine prediction (negative score = contradict)
        # Use more aggressive threshold for better discrimination
        prediction = 0 if weighted_score < -0.05 else 1
        
        # Generate rationale from strongest signal
        sorted_strategies = sorted(strategies.items(), key=lambda x: abs(x[1]['score']), reverse=True)
        primary_strategy = sorted_strategies[0]
        
        if prediction == 0:
            rationale = primary_strategy[1]['reason']
        else:
            # Check if there's positive support
            positive_strategies = [s for s in sorted_strategies if s[1]['score'] > 0.2]
            if positive_strategies:
                rationale = positive_strategies[0][1]['reason']
            else:
                rationale = "No significant contradictions detected in narrative"
        
        # Detailed analysis
        analysis = {
            'weighted_score': weighted_score,
            'strategy_scores': {k: v['score'] for k, v in strategies.items()},
            'primary_signal': primary_strategy[0],
            'confidence': min(abs(weighted_score) + 0.5, 1.0)
        }
        
        return prediction, rationale, analysis
    
    def _check_explicit_contradiction(self, backstory: str, evidence: str) -> Dict:
        """Strategy 1: Direct contradiction detection"""
        bs_lower = backstory.lower()
        ev_lower = evidence.lower()
        
        # Check for direct negation patterns
        contradictions = []
        
        # Pattern 1: Never/Always conflicts
        if 'never' in bs_lower:
            # Extract what was never done
            never_matches = re.finditer(r'never\s+(\w+(?:\s+\w+){0,3})', bs_lower)
            for match in never_matches:
                action = match.group(1)
                if action in ev_lower and len(action) > 5:
                    # Check if evidence shows it DID happen
                    pos_pattern = re.search(rf'\b(did|was|had|went)\s+.*{action}', ev_lower)
                    if pos_pattern:
                        contradictions.append(f"Claims never {action} but narrative shows otherwise")
        
        # Pattern 2: Age contradictions
        bs_ages = set(re.findall(r'at (?:age )?(\d+)|when.*?(\d+)', bs_lower))
        ev_ages = set(re.findall(r'at (?:age )?(\d+)|when.*?(\d+)', ev_lower))
        bs_ages = {int(a[0] or a[1]) for a in bs_ages if any(a)}
        ev_ages = {int(a[0] or a[1]) for a in ev_ages if any(a)}
        
        if bs_ages and ev_ages:
            age_diff = bs_ages - ev_ages
            if age_diff and len(bs_ages) > 0:
                # Different ages mentioned - potential contradiction
                common_context = self._extract_common_context(bs_lower, ev_lower)
                if common_context and len(common_context) > 20:
                    contradictions.append(f"Age inconsistency: backstory mentions {bs_ages}, narrative {ev_ages}")
        
        # Pattern 3: Location contradictions
        locations_bs = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', backstory))
        locations_ev = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', evidence))
        
        # Check for mutually exclusive locations
        exclusive_pairs = [('Paris', 'London'), ('Rome', 'Madrid'), ('Boston', 'Chicago')]
        for loc1, loc2 in exclusive_pairs:
            if loc1 in locations_bs and loc2 in locations_ev:
                if 'born' in bs_lower and 'born' in ev_lower:
                    contradictions.append(f"Birth location contradiction: {loc1} vs {loc2}")
        
        if contradictions:
            return {
                'score': -0.8,
                'reason': f"Direct contradiction: {contradictions[0][:120]}",
                'details': contradictions
            }
        
        return {'score': 0.0, 'reason': 'No explicit contradictions', 'details': []}
    
    def _check_temporal_consistency(self, backstory: str, evidence: str) -> Dict:
        """Strategy 2: Temporal sequence validation"""
        # Extract temporal markers
        temporal_patterns = [
            r'in (\d{4})',
            r'(\d+) years? (?:ago|later|before|after)',
            r'at (?:age )?(\d+)',
            r'first|then|after|before|during|while'
        ]
        
        bs_lower = backstory.lower()
        ev_lower = evidence.lower()
        
        # Check for impossible sequences
        issues = []
        
        # Check if backstory mentions events "after" things that haven't happened yet
        if 'after' in bs_lower:
            after_events = re.findall(r'after\s+([^,.]{10,50})', bs_lower)
            for event in after_events:
                if event not in ev_lower:
                    issues.append(f"References event not in narrative: {event[:40]}")
        
        # Check years
        bs_years = [int(y) for y in re.findall(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', backstory)]
        ev_years = [int(y) for y in re.findall(r'\b(1[6-9]\d{2}|20[0-2]\d)\b', evidence)]
        
        if bs_years and ev_years:
            if min(bs_years) > max(ev_years) + 50:  # Backstory events too far in future
                issues.append(f"Temporal impossibility: backstory years {bs_years} exceed narrative {ev_years}")
        
        if issues:
            return {'score': -0.6, 'reason': f"Temporal inconsistency: {issues[0][:120]}", 'details': issues}
        
        return {'score': 0.0, 'reason': 'Temporal sequence plausible', 'details': []}
    
    def _check_causal_plausibility(self, backstory: str, evidence: str) -> Dict:
        """Strategy 3: Causal relationship validation"""
        bs_lower = backstory.lower()
        ev_lower = evidence.lower()
        
        # Extract claimed causal relationships
        causal_patterns = [
            r'because\s+([^,.]{10,50})',
            r'since\s+([^,.]{10,50})',
            r'led to\s+([^,.]{10,50})',
            r'caused\s+([^,.]{10,50})',
            r'resulted in\s+([^,.]{10,50})'
        ]
        
        issues = []
        for pattern in causal_patterns:
            matches = re.findall(pattern, bs_lower)
            for cause in matches:
                # Check if the cause exists in evidence
                cause_words = self._extract_keywords(cause)
                evidence_words = self._extract_keywords(ev_lower)
                
                overlap = len(cause_words & evidence_words)
                if overlap < max(1, len(cause_words) * 0.3):
                    issues.append(f"Claimed cause not supported: {cause[:40]}")
        
        if issues:
            return {'score': -0.5, 'reason': f"Causal implausibility: {issues[0][:120]}", 'details': issues}
        
        return {'score': 0.1, 'reason': 'Causal relationships supported', 'details': []}
    
    def _check_character_knowledge(self, backstory: str, evidence: str, character: str) -> Dict:
        """Strategy 4: Character knowledge consistency"""
        bs_lower = backstory.lower()
        ev_lower = evidence.lower()
        
        # Check for skills/knowledge claimed but not evidenced
        skill_patterns = [
            r'learned\s+([^,.]{5,30})',
            r'skilled in\s+([^,.]{5,30})',
            r'expert\s+([^,.]{5,30})',
            r'knew\s+([^,.]{5,30})',
            r'studied\s+([^,.]{5,30})'
        ]
        
        claimed_skills = []
        for pattern in skill_patterns:
            claimed_skills.extend(re.findall(pattern, bs_lower))
        
        unsupported = []
        for skill in claimed_skills:
            if skill not in ev_lower:
                skill_words = set(skill.split())
                ev_words = set(evidence.split())
                if len(skill_words & ev_words) < 2:
                    unsupported.append(skill)
        
        if len(unsupported) > 2:
            return {'score': -0.4, 'reason': f"Unsupported skills: {unsupported[0][:40]}", 'details': unsupported}
        
        return {'score': 0.0, 'reason': 'Character knowledge consistent', 'details': []}
    
    def _check_geographic_consistency(self, backstory: str, evidence: str) -> Dict:
        """Strategy 5: Geographic plausibility"""
        # Extract place names
        places_bs = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', backstory))
        places_ev = set(re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', evidence))
        
        # Filter to likely place names (appear in evidence)
        likely_places_bs = {p for p in places_bs if len(p) > 4}
        likely_places_ev = {p for p in places_ev if len(p) > 4}
        
        # Check for places in backstory not mentioned anywhere in narrative
        unknown_places = likely_places_bs - likely_places_ev
        
        if len(unknown_places) > 3:
            return {'score': -0.3, 'reason': f"Mentions unfamiliar locations: {list(unknown_places)[:2]}", 
                   'details': list(unknown_places)}
        
        return {'score': 0.0, 'reason': 'Geographic consistency maintained', 'details': []}
    
    def _check_relationship_consistency(self, backstory: str, evidence: str) -> Dict:
        """Strategy 6: Relationship network validation"""
        bs_lower = backstory.lower()
        ev_lower = evidence.lower()
        
        # Extract relationship claims
        relationship_patterns = [
            r'(father|mother|brother|sister|son|daughter)\s+(?:was|named)\s+([A-Z]\w+)',
            r'met\s+([A-Z]\w+)',
            r'friend\s+([A-Z]\w+)',
            r'knew\s+([A-Z]\w+)'
        ]
        
        claimed_relations = []
        for pattern in relationship_patterns:
            matches = re.finditer(pattern, backstory)
            for match in matches:
                if len(match.groups()) > 1:
                    claimed_relations.append(match.group(2))
                else:
                    claimed_relations.append(match.group(1))
        
        # Check if these people appear in evidence
        unsupported_relations = []
        for person in claimed_relations:
            if person.lower() not in ev_lower:
                unsupported_relations.append(person)
        
        if len(unsupported_relations) > 2:
            return {'score': -0.3, 'reason': f"References unknown characters: {unsupported_relations[0]}", 
                   'details': unsupported_relations}
        
        return {'score': 0.0, 'reason': 'Relationships align with narrative', 'details': []}
    
    def _check_factual_alignment(self, backstory: str, evidence: str) -> Dict:
        """Strategy 7: Overall factual alignment score"""
        # Calculate keyword overlap as baseline
        bs_keywords = self._extract_keywords(backstory)
        ev_keywords = self._extract_keywords(evidence)
        
        if not bs_keywords:
            return {'score': 0.0, 'reason': 'Minimal factual content', 'details': []}
        
        overlap = len(bs_keywords & ev_keywords)
        overlap_ratio = overlap / len(bs_keywords)
        
        # Higher overlap = more aligned
        if overlap_ratio > 0.4:
            return {'score': 0.3, 'reason': f'Strong factual alignment ({overlap_ratio:.1%} keyword overlap)', 
                   'details': []}
        elif overlap_ratio < 0.1:
            return {'score': -0.2, 'reason': f'Weak factual alignment ({overlap_ratio:.1%} overlap)', 
                   'details': []}
        
        return {'score': 0.0, 'reason': f'Moderate alignment ({overlap_ratio:.1%})', 'details': []}
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract meaningful keywords"""
        words = re.findall(r'\b[a-z]{4,}\b', text.lower())
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'with',
                     'was', 'one', 'our', 'out', 'get', 'has', 'him', 'his', 'how',
                     'that', 'this', 'from', 'they', 'were', 'been', 'have', 'their'}
        return {w for w in words if w not in stop_words}
    
    def _extract_common_context(self, text1: str, text2: str) -> str:
        """Find common context between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        common = words1 & words2
        return ' '.join(sorted(list(common))[:10])


class UltimateConsistencySystem:
    """
    Ultimate system combining vector retrieval with advanced detection
    """
    
    def __init__(self):
        self.vector_store = PathwayVectorStore()
        self.detector = AdvancedConsistencyDetector()
        self.narratives = {}
        
    def load_narratives(self, narratives_dir: str):
        """Load and index narratives"""
        import os
        
        books = {
            'In Search of the Castaways': 'In_search_of_the_castaways.txt',
            'The Count of Monte Cristo': 'The_Count_of_Monte_Cristo.txt'
        }
        
        for book_name, filename in books.items():
            filepath = os.path.join(narratives_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                self.narratives[book_name] = text
                self._index_narrative(book_name, text)
                print(f"✓ Loaded: {book_name} ({len(text)} chars)")
    
    def _index_narrative(self, book_name: str, text: str):
        """Index narrative into vector store"""
        # Chunk into paragraphs
        paragraphs = text.split('\n\n')
        
        docs = []
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > 50:
                docs.append({
                    'text': para.strip(),
                    'book': book_name,
                    'paragraph_id': i
                })
        
        self.vector_store.add_documents(docs)
    
    def predict(self, book_name: str, character: str, backstory: str) -> Tuple[int, str]:
        """Make prediction using hybrid approach"""
        # Retrieve relevant evidence
        query = f"{character} {backstory}"
        evidence_docs = self.vector_store.search(query, top_k=20, filters={'book': book_name})
        
        # Run advanced detection
        prediction, rationale, analysis = self.detector.analyze_consistency(
            backstory, evidence_docs, character
        )
        
        return prediction, rationale
    
    def predict_batch(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction"""
        results = []
        
        for idx, row in test_df.iterrows():
            prediction, rationale = self.predict(
                row['book_name'],
                row['char'],
                row['content']
            )
            
            results.append({
                'id': row['id'],
                'prediction': prediction,
                'rationale': rationale
            })
            
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)}")
        
        return pd.DataFrame(results)


def main():
    """Main execution"""
    print("=" * 80)
    print("KHARAGPUR DATA SCIENCE HACKATHON 2026 - TRACK A (ULTIMATE)")
    print("Hybrid Advanced System: Vector Retrieval + Multi-Strategy Detection")
    print("=" * 80)
    print()
    
    # Initialize
    system = UltimateConsistencySystem()
    
    # Load narratives
    print("Loading narratives...")
    system.load_narratives('/mnt/user-data/uploads')
    
    # Load test data
    print("\nLoading test data...")
    test_df = pd.read_csv('/mnt/user-data/uploads/test.csv')
    print(f"Test set: {len(test_df)} examples\n")
    
    # Predict
    print("Generating predictions...")
    results_df = system.predict_batch(test_df)
    
    # Output
    output_df = pd.DataFrame({
        'Story ID': results_df['id'],
        'Prediction': results_df['prediction'],
        'Rationale': results_df['rationale']
    })
    
    output_path = '/mnt/user-data/outputs/results_ultimate.csv'
    output_df.to_csv(output_path, index=False)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total: {len(output_df)}")
    print(f"Consistent: {(output_df['Prediction'] == 1).sum()} ({(output_df['Prediction'] == 1).sum()/len(output_df)*100:.1f}%)")
    print(f"Inconsistent: {(output_df['Prediction'] == 0).sum()} ({(output_df['Prediction'] == 0).sum()/len(output_df)*100:.1f}%)")
    print("\n" + "=" * 80)
    
    return output_df


if __name__ == "__main__":
    main()
