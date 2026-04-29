# RTTBench-Mono Dataset

This directory should contain the RTTBench-Mono.jsonl dataset file.

## Getting the data

Copy the dataset from the original location:

```bash
cp /path/to/original/RTTBench-Mono.jsonl ./RTTBench-Mono.jsonl
```

Or if you have the original `language_resource_assessment` codebase:

```bash
cp ../../../language_resource_assessment/data/rttbench_mono_dataset/RTTBench-Mono.jsonl .
```

## Dataset format

The dataset is in JSONL format with the following fields:

```json
{
  "id": 1,
  "domain": "science",
  "text": "The quick brown fox jumps over the lazy dog."
}
```
