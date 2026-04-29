# XNLI Tasks

This directory contains XNLI (Cross-lingual Natural Language Inference) task configurations for Chichewa evaluation.

## Files

- `_xnli.yaml` - Group configuration that combines all XNLI tasks
- `xnli_common_yaml` - Common template configuration shared by all XNLI language tasks
- `xnli_ny.yaml` - Chichewa (Nyanja) XNLI task configuration

## Task Structure

### XNLI Format
XNLI evaluates natural language inference across languages using premise-hypothesis pairs with three labels:
- **0**: Entailment (hypothesis follows from premise)
- **1**: Neutral (hypothesis neither follows nor contradicts premise)  
- **2**: Contradiction (hypothesis contradicts premise)

### Chichewa Prompts
The task uses natural Chichewa prompts:
- **Entailment**: `"choona? Inde, "` (right? Yes,)
- **Neutral**: `"choona? Choncho, "` (right? So,)
- **Contradiction**: `"choona? Ayi, "` (right? No,)

## Dataset Requirements

The task expects a dataset file at:
```
eval/data/nya/eval/xnli_*_english2chichewa_microsoft_translated.jsonl
```

With fields:
- `premise` - The premise sentence in Chichewa
- `hypothesis` - The hypothesis sentence in Chichewa
- `label` - The entailment label (0, 1, or 2)

## Usage

Add to your `lm_eval_config.yaml`:
```yaml
tasks:
  - xnli_ny
  # or use the group:
  - xnli
```

## Standards Compliance

This implementation follows lm-evaluation-harness XNLI standards:
- Uses standard `multiple_choice` output type
- Implements accuracy metric with mean aggregation
- Follows standard include/override pattern for common configuration
- Uses same splits configuration as standard XNLI tasks
