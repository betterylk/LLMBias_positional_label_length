Python scripts to generate LLM outputs under positional, label, and length bias are provided.

Scripts for positional and label bias follow the format:

runCX_Y.py

where X is 1–6 representing the 6 categories, and Y can be PB (positional bias) or LaB (label bias).

Benchmarks categorised by input–output structure:

Category Input structure => Output type
C1 Question + Options => Single-answer
C2 Question + Options + Topic => Single-answer
C3 Question + Options + Passage => Single-answer
C4 Question + Options + Facts => Single-answer
C5 Question + Options => Multiple-answer
C6 Question + Options + Passage => Multiple-answer

Benchmarks used in this study:

C1 ai2_arc
https://huggingface.co/datasets/allenai/ai2_arc

C2 MMLU
https://huggingface.co/datasets/cais/mmlu

C3 cosmosqa
https://huggingface.co/datasets/allenai/cosmos_qa

C4 openbookqa
https://huggingface.co/datasets/allenai/openbookqa

C5 mmluMA
https://huggingface.co/datasets/Obsismc/mmlu-multi_answers

C6 sata-bench
https://huggingface.co/datasets/sata-bench/sata-bench

Script for length bias:

run_length_bias.py

The length bias script perturbs the relative lengths of answer options while keeping their semantic content unchanged, and evaluates whether model predictions correlate with option length. Length bias is benchmark-agnostic and can be applied across all applicable datasets.
