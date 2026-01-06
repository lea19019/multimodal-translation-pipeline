# Multimodal Translation Pipelines for Low-Resource African Languages

## Project Summary
This repository documents a research and engineering effort focused on building, evaluating, and scaling speech-to-speech translation pipelines for four low-resource African languages: Efik, Igbo, Swahili, and Xhosa. The project blends hands-on deep learning, large-scale experimentation, and robust engineering:

- Fine-tuned multiple versions of Coqui XTTS for TTS on African languages
- Fine-tuned Meta NLLB-600M for neural machine translation (NMT)
- Developed and integrated custom BLASER 2.0 encoders for speech-to-speech evaluation
- Designed and benchmarked 10+ pipeline combinations (ASR, NMT, TTS)
- Built a modular microservice architecture for scalable, reproducible experiments
- Ran and debugged experiments on supercomputing clusters (SLURM, multi-GPU)
- Automated data processing, model training, and evaluation workflows
- Explored different data types, model architectures, and evaluation strategies

**Full technical report:** [project_report/main.tex](project_report/main.pdf)

## System Overview

```mermaid
graph TD;
    Client[Web Client / API User]
    Gateway[API Gateway]
    ASR[ASR Service<br>(Whisper)]
    NMT[NMT Service<br>(NLLB)]
    TTS[TTS Service<br>(XTTS)]
    Eval[Evaluation Service<br>(BLASER, COMET, etc.)]
    Client-->|REST/HTTP|Gateway
    Gateway-->|Audio/Text|ASR
    Gateway-->|Text|NMT
    Gateway-->|Text|TTS
    Gateway-->|Results|Eval
    ASR-->|Transcription|Gateway
    NMT-->|Translation|Gateway
    TTS-->|Speech|Gateway
    Eval-->|Metrics/Visuals|Gateway
```

- **API Gateway**: Orchestrates the full translation pipeline, exposing a unified API for all services.
- **ASR Service**: Speech-to-text using Whisper (OpenAI), with support for African accents.
- **NMT Service**: Text-to-text translation using fine-tuned NLLB-600M.
- **TTS Service**: Text-to-speech using multiple fine-tuned XTTS models, including language-specific and cross-lingual variants.
- **Evaluation Service**: Computes BLEU, chrF, COMET, MCD, and BLASER 2.0 metrics; supports custom BLASER encoders for African languages.

## Pipeline Variants & Model Details

| Model Name   | Input                | Output         |
|-------------|----------------------|----------------|
| Native      | African text         | African audio  |
| Eng2Multi   | English text         | African audio  |
| Eng2Efik    | English text         | Efik audio     |
| Eng2Swa     | English text         | Swahili audio  |
| BiTag       | <eng>{eng} <lang>{lang} | African audio  |
| TransTag    | <translate> <eng>{eng} <lang>{lang} | African audio  |

## Pipeline Combinations

| Pipeline   | Translation         | TTS Model      |
|------------|---------------------|----------------|
| Pipeline 1 | NLLB                | Native         |
| Pipeline 2 | NLLB                | BiTag          |
| Pipeline 3 | None                | BiTag          |
| Pipeline 4 | NLLB (custom format)| BiTag          |
| Pipeline 5 | NLLB                | TransTag       |
| Pipeline 6 | None                | TransTag       |
| Pipeline 7 | NLLB (custom format)| TransTag       |
| Pipeline 8 | None                | Eng2Multi      |
| Pipeline 9 | None                | Eng2Efik       |
| Pipeline 10| None                | Eng2Swa        |

## Evaluation Metrics
- **Text**: BLEU, chrF, COMET (McGill-NLP/ssa-comet-qe)
- **Audio**: MCD (Mel-Cepstral Distance), BLASER 2.0 (custom encoders for African languages)

## Results

### NLLB Translation Quality
| Language | BLEU | chrF | COMET |
|----------|---------------------|---------------------|-------------------|
| Efik     | 29.8 ± 18.6         | 59.0 ± 14.2         | 0.603 ± 0.068     |
| Igbo     | 33.0 ± 23.5         | 60.6 ± 17.8         | 0.641 ± 0.068     |
| Swahili  | 49.2 ± 18.5         | 76.0 ± 10.5         | 0.676 ± 0.051     |
| Xhosa    | 32.4 ± 20.7         | 71.1 ± 12.8         | 0.649 ± 0.048     |
| **Overall** | **36.1 ± 7.7**   | **66.7 ± 7.1**      | **0.642 ± 0.026** |

### Audio & Speech Quality (MCD, BLASER)
| Metric | Pipeline | Efik | Igbo | Swahili | Xhosa | Overall |
|--------|----------|------|------|---------|-------|---------|
| **MCD** | NLLB → Native | **13.12 ± 1.35** | **12.96 ± 1.21** | **13.40 ± 0.95** | **11.89 ± 1.06** | **12.84 ± 0.57** |
| **BLASER** | Source → Eng2Multi | 2.72 ± 0.21 | 3.07 ± 0.25 | **2.84 ± 0.24** | **2.85 ± 0.24** | **2.87 ± 0.13** |

## What You'll Find Here
- End-to-end code for training, evaluation, and deployment of translation pipelines
- Scripts for data preprocessing, model finetuning, and evaluation
- Modular microservices for each pipeline component (ASR, NMT, TTS, Evaluation)
- Reproducible experiments and results, with clear documentation

## Professional/Technical Highlights
- 140+ hours of hands-on engineering and research
- Experience with supercomputing (SLURM, multi-GPU jobs, debugging at scale)
- Deep learning: model finetuning, transfer learning, and custom evaluation
- Data engineering: cleaning, aligning, and managing multilingual/multimodal datasets
- Experimentation: rapid prototyping, ablation studies, and pipeline benchmarking
- Robust, production-style codebase with clear separation of concerns

