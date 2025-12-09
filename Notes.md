# Pipelines Tested

## First Approach
Source Text -> NLLB -> Predicted Text
Target Text -> XTTS -> Predicted Audio

## Flow of Evaluation
Predicted Text -> |      |
Target Text    -> | BLEU | -> Score

Predicted Text -> |      |
Target Text    -> | chrF | -> Score

Source Text    -> |       |
Predicted Text -> |       |
Target Text    -> | COMET | -> Score

Predicted Audio -> |      |
Target Audio    -> | MCD | -> Score

Source Audio    -> |        |
Predicted Aidio -> | BLASER | -> Score

## Pipelines to Test

### Using Predicted Text as Input
Source Text -> NLLB -> Predicted Text
Predicted Text -> XTTS -> Predicted Audio

### NLLB and XTTS Pipeline Training (w/target_text)
Source Text -> NLLB -> Predicted Text
Target Text -> XTTS -> Predicted Audio

### NLLB and XTTS Pipeline Training (w/predicted_text)
Source Text -> NLLB -> Predicted Text
Predicted Text -> XTTS -> Predicted Audio