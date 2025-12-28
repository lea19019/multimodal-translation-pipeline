#!/usr/bin/env python3
"""
Pipeline Configuration Module

Defines all 8 translation pipelines and provides mapping functions
to locate their synthesized data.
"""

from pathlib import Path
from typing import Dict, List, Optional


# All 8 pipeline definitions
PIPELINES = [
    {
        'id': 'pipeline_1',
        'name': 'NLLB → MULTILINGUAL_TRAINING',
        'short_name': 'nllb_multilingual',
        'descriptor': 'nllb_tgt',
        'checkpoint': 'MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632',
        'flow': 'src_txt → NLLB → pred_txt → MULTILINGUAL_TRAINING → pred_wav',
        'uses_nmt': True,
        'text_input': 'predicted',
        'text_format': 'plain',
        'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
    },
    {
        'id': 'pipeline_2',
        'name': 'NLLB → Src_Tgt',
        'short_name': 'nllb_src_tgt',
        'descriptor': 'nllb_tgt',
        'checkpoint': 'Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8',
        'flow': 'src_txt → NLLB → pred_txt → Src_Tgt → pred_wav',
        'uses_nmt': True,
        'text_input': 'predicted',
        'text_format': 'plain',
        'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
    },
    {
        'id': 'pipeline_3',
        'name': 'Source → Src_Tgt',
        'short_name': 'src_src_tgt',
        'descriptor': 'src',
        'checkpoint': 'Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8',
        'flow': 'src_txt → Src_Tgt → pred_wav',
        'uses_nmt': False,
        'text_input': 'source',
        'text_format': 'plain',
        'metrics': ['mcd', 'blaser']  # No text metrics (no translation)
    },
    {
        'id': 'pipeline_4',
        'name': 'Custom Lang → Src_Tgt',
        'short_name': 'custom_lang_src_tgt',
        'descriptor': 'custom_lang',
        'checkpoint': 'Src_Tgt_8_12-December-09-2025_10+15PM-1e192a8',
        'flow': '<eng>{src} <{lang}>{pred} → Src_Tgt → pred_wav',
        'uses_nmt': True,
        'text_input': 'both',
        'text_format': 'custom_lang',
        'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
    },
    {
        'id': 'pipeline_5',
        'name': 'NLLB → Translate_Src_Tgt',
        'short_name': 'nllb_translate_src_tgt',
        'descriptor': 'nllb_tgt',
        'checkpoint': 'Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254',
        'flow': 'src_txt → NLLB → pred_txt → Translate_Src_Tgt → pred_wav',
        'uses_nmt': True,
        'text_input': 'predicted',
        'text_format': 'plain',
        'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
    },
    {
        'id': 'pipeline_6',
        'name': 'Source → Translate_Src_Tgt',
        'short_name': 'src_translate_src_tgt',
        'descriptor': 'src',
        'checkpoint': 'Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254',
        'flow': 'src_txt → Translate_Src_Tgt → pred_wav',
        'uses_nmt': False,
        'text_input': 'source',
        'text_format': 'plain',
        'metrics': ['mcd', 'blaser']  # No text metrics
    },
    {
        'id': 'pipeline_7',
        'name': 'Custom Translate → Translate_Src_Tgt',
        'short_name': 'custom_translate_translate_src_tgt',
        'descriptor': 'custom_translate',
        'checkpoint': 'Translate_Src_Tgt_8_12-December-08-2025_03+54PM-b2b8254',
        'flow': '<translate> <eng>{src} <{lang}>{pred} → Translate_Src_Tgt → pred_wav',
        'uses_nmt': True,
        'text_input': 'both',
        'text_format': 'custom_translate',
        'metrics': ['bleu', 'chrf', 'comet', 'mcd', 'blaser']
    },
    {
        'id': 'pipeline_8',
        'name': 'Source → Multilingual_Src',
        'short_name': 'src_multilingual_src',
        'descriptor': 'src',
        'checkpoint': 'Multilingual_Src_6_12-December-06-2025_10+30PM-b2b8254',
        'flow': 'src_txt → Multilingual_Src → pred_wav',
        'uses_nmt': False,
        'text_input': 'source',
        'text_format': 'plain',
        'metrics': ['mcd', 'blaser']  # No text metrics
    },
    {
        'id': 'pipeline_9',
        'name': 'Source Audio → XTTS_Efik (Efik only)',
        'short_name': 'xtts_efik',
        'descriptor': 'src',
        'checkpoint': 'Src_Efik_12_12-December-13-2025_03+12PM-da9effb',
        'flow': 'src_wav → XTTS_Efik → pred_wav',
        'uses_nmt': False,
        'text_input': 'source',
        'text_format': 'plain',
        'metrics': ['mcd', 'blaser'],  # Audio-to-audio, no text metrics
        'languages': ['efik']  # Efik only
    },
        {
        'id': 'pipeline_10',
        'name': 'Source Audio → XTTS_Swahili (Swahili only)',
        'short_name': 'xtts_swahili',
        'descriptor': 'src',
        'checkpoint': 'Src_Swahili_14_12-December-14-2025_04+27PM-da9effb',
        'flow': 'src_wav → XTTS_Swahili → pred_wav',
        'uses_nmt': False,
        'text_input': 'source',
        'text_format': 'plain',
        'metrics': ['mcd', 'blaser'],  # Audio-to-audio, no text metrics
        'languages': ['swahili']  # Efik only
    },
]

# Language mappings
LANGUAGES = {
    'efik': 'efi',
    'igbo': 'ibo',
    'swahili': 'swa',
    'xhosa': 'xho'
}

LANGUAGE_FULL_NAMES = {
    'efi': 'efik',
    'ibo': 'igbo',
    'swa': 'swahili',
    'xho': 'xhosa'
}


def get_pipeline(pipeline_id: str) -> Optional[Dict]:
    """
    Get pipeline configuration by ID.

    Args:
        pipeline_id: Pipeline identifier (e.g., 'pipeline_1')

    Returns:
        Pipeline configuration dict or None if not found
    """
    for pipeline in PIPELINES:
        if pipeline['id'] == pipeline_id:
            return pipeline
    return None


def get_all_pipeline_ids() -> List[str]:
    """Get list of all pipeline IDs."""
    return [p['id'] for p in PIPELINES]


def get_all_pipelines() -> List[Dict]:
    """Get all pipeline configurations."""
    return PIPELINES.copy()


def get_synthesis_path(pipeline_id: str, language: str, base_dir: Path = None) -> Dict[str, Path]:
    """
    Map pipeline and language to synthesis data paths.

    Args:
        pipeline_id: Pipeline identifier (e.g., 'pipeline_1')
        language: Full language name (e.g., 'efik')
        base_dir: Base data directory (default: /home/vacl2/multimodal_translation/services/data/languages)

    Returns:
        Dict containing:
            - csv_path: Path to predicted CSV file
            - audio_dir: Path to predicted audio directory
            - nmt_csv_path: Path to NMT predictions CSV (ground truth)
            - ref_audio_dir: Path to reference audio directory
            - pipeline: Pipeline configuration dict

    Example:
        >>> paths = get_synthesis_path('pipeline_1', 'efik')
        >>> print(paths['csv_path'])
        /home/.../efik/predicted_nllb_tgt_MULTILINGUAL_TRAINING_11_5-November-05-2025_10+57AM-cc09632.csv
    """
    if base_dir is None:
        base_dir = Path('/home/vacl2/multimodal_translation/services/data/languages')

    pipeline = get_pipeline(pipeline_id)
    if pipeline is None:
        raise ValueError(f"Pipeline not found: {pipeline_id}")

    descriptor = pipeline['descriptor']
    checkpoint = pipeline['checkpoint']

    # Construct paths following the pattern: predicted_{descriptor}_{checkpoint}
    lang_dir = base_dir / language
    csv_filename = f"predicted_{descriptor}_{checkpoint}.csv"
    audio_dir_name = f"predicted_{descriptor}_{checkpoint}"

    return {
        'csv_path': lang_dir / csv_filename,
        'audio_dir': lang_dir / audio_dir_name,
        'nmt_csv_path': lang_dir / "nmt_predictions_multilang_finetuned_final.csv",
        'ref_audio_dir': lang_dir / "processed_audio_normalized",
        'pipeline': pipeline,
        'language': language,
        'iso_code': LANGUAGES[language]
    }


def validate_synthesis_paths(pipeline_id: str, language: str, base_dir: Path = None) -> bool:
    """
    Check if synthesis paths exist for a given pipeline and language.

    Args:
        pipeline_id: Pipeline identifier
        language: Full language name
        base_dir: Base data directory

    Returns:
        True if all required paths exist, False otherwise
    """
    paths = get_synthesis_path(pipeline_id, language, base_dir)

    required_paths = [
        paths['csv_path'],
        paths['audio_dir'],
        paths['nmt_csv_path'],
        paths['ref_audio_dir']
    ]

    missing = []
    for path in required_paths:
        if not path.exists():
            missing.append(str(path))

    if missing:
        return False
    return True


def get_available_syntheses(base_dir: Path = None) -> List[Dict]:
    """
    Discover all available synthesized data.

    Args:
        base_dir: Base data directory

    Returns:
        List of dicts with pipeline_id, language, and paths for available syntheses
    """
    if base_dir is None:
        base_dir = Path('/home/vacl2/multimodal_translation/services/data/languages')

    available = []

    for pipeline in PIPELINES:
        for language in LANGUAGES.keys():
            if validate_synthesis_paths(pipeline['id'], language, base_dir):
                paths = get_synthesis_path(pipeline['id'], language, base_dir)
                available.append({
                    'pipeline_id': pipeline['id'],
                    'pipeline_name': pipeline['name'],
                    'language': language,
                    'paths': paths
                })

    return available


if __name__ == '__main__':
    # Test the configuration
    print("=" * 70)
    print("PIPELINE CONFIGURATION TEST")
    print("=" * 70)

    print(f"\nTotal pipelines defined: {len(PIPELINES)}")
    print(f"Total languages: {len(LANGUAGES)}")
    print(f"Total possible combinations: {len(PIPELINES) * len(LANGUAGES)}")

    print("\n" + "=" * 70)
    print("PIPELINE DEFINITIONS")
    print("=" * 70)

    for i, pipeline in enumerate(PIPELINES, 1):
        print(f"\n{i}. {pipeline['name']}")
        print(f"   ID: {pipeline['id']}")
        print(f"   Flow: {pipeline['flow']}")
        print(f"   Uses NMT: {pipeline['uses_nmt']}")
        print(f"   Metrics: {', '.join(pipeline['metrics'])}")

    print("\n" + "=" * 70)
    print("PATH MAPPING TEST")
    print("=" * 70)

    # Test path mapping for first pipeline
    test_pipeline = 'pipeline_1'
    test_language = 'efik'

    print(f"\nTesting: {test_pipeline} × {test_language}")
    paths = get_synthesis_path(test_pipeline, test_language)

    print(f"\nGenerated paths:")
    print(f"  CSV: {paths['csv_path']}")
    print(f"  Audio dir: {paths['audio_dir']}")
    print(f"  NMT CSV: {paths['nmt_csv_path']}")
    print(f"  Ref audio: {paths['ref_audio_dir']}")

    print(f"\nPath validation:")
    is_valid = validate_synthesis_paths(test_pipeline, test_language)
    print(f"  All paths exist: {is_valid}")

    print("\n" + "=" * 70)
    print("AVAILABLE SYNTHESES")
    print("=" * 70)

    available = get_available_syntheses()
    print(f"\nFound {len(available)} available synthesis runs:")

    by_lang = {}
    for item in available:
        lang = item['language']
        if lang not in by_lang:
            by_lang[lang] = []
        by_lang[lang].append(item['pipeline_name'])

    for lang, pipelines in sorted(by_lang.items()):
        print(f"\n{lang.capitalize()}: {len(pipelines)} pipelines")
        for p_name in pipelines:
            print(f"  ✓ {p_name}")

    print("\n" + "=" * 70)
