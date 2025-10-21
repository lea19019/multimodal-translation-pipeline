"""
Multimodal Translation Evaluation System

A comprehensive evaluation framework for assessing machine translation quality
across different modalities: text-to-text, audio-to-text, text-to-audio, and
audio-to-audio.

Supported Metrics:
- BLEU (BiLingual Evaluation Understudy)
- chrF/chrF++ (Character n-gram F-score)
- COMET (Crosslingual Optimized Metric for Evaluation of Translation)
- MCD (Mel-Cepstral Distance)
- BLASER 2.0 (Speech-to-Speech Translation Evaluation)

Quick Start:
    >>> from data_loader import load_samples
    >>> from text_metrics import TextMetrics
    >>> 
    >>> samples, errors = load_samples("../data/app_evaluation/text_to_text")
    >>> metrics = TextMetrics()
    >>> results = metrics.evaluate(hypotheses, references)

For CLI usage, see README.md
"""

__version__ = "0.1.0"
__author__ = "Translation Evaluation Team"

# Expose main classes
from data_loader import load_samples, TranslationSample
from text_metrics import TextMetrics, compute_bleu, compute_chrf
from audio_metrics import AudioMetrics, compute_mcd

__all__ = [
    'load_samples',
    'TranslationSample',
    'TextMetrics',
    'AudioMetrics',
    'compute_bleu',
    'compute_chrf',
    'compute_mcd',
]
