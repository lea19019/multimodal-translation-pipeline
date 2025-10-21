"""
Text-based evaluation metrics using sacrebleu.

Provides BLEU and chrF/chrF++ computation for translation evaluation.
"""

import logging
from typing import Dict, List, Optional

import sacrebleu
from sacrebleu.metrics import BLEU, CHRF

logger = logging.getLogger(__name__)


class TextMetrics:
    """Compute text-based metrics (BLEU, chrF/chrF++)."""
    
    def __init__(
        self,
        lowercase: bool = False,
        tokenize: str = '13a',
        chrf_word_order: int = 0,  # 0 = chrF, 2 = chrF++
    ):
        """
        Initialize text metrics.
        
        Args:
            lowercase: If True, enables case-insensitivity for BLEU
            tokenize: Tokenization method for BLEU ('13a', 'intl', 'zh', 'ja-mecab', etc.)
            chrf_word_order: Word n-gram order for chrF (0=chrF, 2=chrF++)
        """
        self.lowercase = lowercase
        self.tokenize = tokenize
        self.chrf_word_order = chrf_word_order
        
        # Initialize metric objects
        self.bleu = BLEU(lowercase=lowercase, tokenize=tokenize)
        self.chrf = CHRF(word_order=chrf_word_order)
    
    def compute_bleu(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict:
        """
        Compute BLEU score.
        
        Args:
            hypotheses: List of hypothesis translations
            references: List of reference lists (supports multiple references)
            
        Returns:
            Dictionary with corpus-level and sentence-level scores
        """
        try:
            # Reshape references for sacrebleu (expects list of reference lists)
            # If references is [[ref1_sys1, ref1_sys2], [ref2_sys1, ref2_sys2]]
            # We need to transpose to [[ref1_sys1, ref2_sys1], [ref1_sys2, ref2_sys2]]
            if references and isinstance(references[0], list):
                num_refs = len(references[0])
                refs_transposed = [
                    [references[i][j] for i in range(len(references))]
                    for j in range(num_refs)
                ]
            else:
                # Single reference case
                refs_transposed = [references]
            
            # Corpus-level score
            corpus_score = self.bleu.corpus_score(hypotheses, refs_transposed)
            
            # Sentence-level scores
            sentence_scores = []
            for hyp, ref_list in zip(hypotheses, references):
                if isinstance(ref_list, str):
                    ref_list = [ref_list]
                sent_score = self.bleu.sentence_score(hyp, ref_list)
                sentence_scores.append(sent_score.score)
            
            return {
                'corpus_score': corpus_score.score,
                'sentence_scores': sentence_scores,
                'signature': str(self.bleu.get_signature()),
                'details': {
                    'precision': [corpus_score.precisions[i] for i in range(len(corpus_score.precisions))],
                    'bp': corpus_score.bp,
                    'ratio': corpus_score.sys_len / corpus_score.ref_len if corpus_score.ref_len > 0 else 0,
                    'hyp_len': corpus_score.sys_len,
                    'ref_len': corpus_score.ref_len,
                }
            }
        
        except Exception as e:
            logger.error(f"Error computing BLEU: {e}")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(hypotheses),
                'signature': 'error',
                'error': str(e),
            }
    
    def compute_chrf(
        self,
        hypotheses: List[str],
        references: List[List[str]],
    ) -> Dict:
        """
        Compute chrF/chrF++ score.
        
        Args:
            hypotheses: List of hypothesis translations
            references: List of reference lists (supports multiple references)
            
        Returns:
            Dictionary with corpus-level and sentence-level scores
        """
        try:
            # Reshape references for sacrebleu
            if references and isinstance(references[0], list):
                num_refs = len(references[0])
                refs_transposed = [
                    [references[i][j] for i in range(len(references))]
                    for j in range(num_refs)
                ]
            else:
                refs_transposed = [references]
            
            # Corpus-level score
            corpus_score = self.chrf.corpus_score(hypotheses, refs_transposed)
            
            # Sentence-level scores
            sentence_scores = []
            for hyp, ref_list in zip(hypotheses, references):
                if isinstance(ref_list, str):
                    ref_list = [ref_list]
                sent_score = self.chrf.sentence_score(hyp, ref_list)
                sentence_scores.append(sent_score.score)
            
            metric_name = f"chrF{self.chrf.char_order}"
            if self.chrf_word_order > 0:
                metric_name += f"++"
            
            return {
                'corpus_score': corpus_score.score,
                'sentence_scores': sentence_scores,
                'signature': str(self.chrf.get_signature()),
                'metric_name': metric_name,
            }
        
        except Exception as e:
            logger.error(f"Error computing chrF: {e}")
            return {
                'corpus_score': 0.0,
                'sentence_scores': [0.0] * len(hypotheses),
                'signature': 'error',
                'error': str(e),
            }
    
    def evaluate(
        self,
        hypotheses: List[str],
        references: List[List[str]],
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, Dict]:
        """
        Compute multiple text metrics.
        
        Args:
            hypotheses: List of hypothesis translations
            references: List of reference lists
            metrics: List of metrics to compute (['bleu', 'chrf'] or None for all)
            
        Returns:
            Dictionary mapping metric name to results
        """
        if metrics is None:
            metrics = ['bleu', 'chrf']
        
        results = {}
        
        if 'bleu' in metrics:
            results['bleu'] = self.compute_bleu(hypotheses, references)
        
        if 'chrf' in metrics:
            results['chrf'] = self.compute_chrf(hypotheses, references)
        
        return results


def compute_bleu(
    hypotheses: List[str],
    references: List[List[str]],
    lowercase: bool = False,
    tokenize: str = '13a',
) -> Dict:
    """
    Convenience function to compute BLEU score.
    
    Args:
        hypotheses: List of hypothesis translations
        references: List of reference lists
        lowercase: Enable case-insensitivity
        tokenize: Tokenization method
        
    Returns:
        Dictionary with BLEU scores and details
    """
    metrics = TextMetrics(lowercase=lowercase, tokenize=tokenize)
    return metrics.compute_bleu(hypotheses, references)


def compute_chrf(
    hypotheses: List[str],
    references: List[List[str]],
    word_order: int = 0,
) -> Dict:
    """
    Convenience function to compute chrF/chrF++ score.
    
    Args:
        hypotheses: List of hypothesis translations
        references: List of reference lists
        word_order: Word n-gram order (0=chrF, 2=chrF++)
        
    Returns:
        Dictionary with chrF scores
    """
    metrics = TextMetrics(chrf_word_order=word_order)
    return metrics.compute_chrf(hypotheses, references)
