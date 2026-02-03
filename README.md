Python scripts to generate LLM outputs under the positional and label bias are provided in the format of runCX_Y.py, where X is 1-6 representing the 6 categories, and Y can be PB (positional) or LaB (label) bias.

Benchmarks categorised by input--output structure
Category  Input structure  \Rightarrow  Output type
C1  Question + Options  \Rightarrow  Single-answer 
C2  Question + Options + Topic  \Rightarrow  Single-answer 
C3  Question + Options + Passage  \Rightarrow  Single-answer 
C4  Question + Options + Facts  \Rightarrow  Single-answer
C5  Question + Options  \Rightarrow  Multiple-answer 
C6  Question + Options + Passage  \Rightarrow  Multiple-answer 


Benchmarks used in this study are from:
C1 ai2\_arc      https://huggingface.co/datasets/allenai/ai2_arc  
C2 MMLU        https://huggingface.co/datasets/cais/mmlu                                
C3 cosmosqa      https://huggingface.co/datasets/allenai/cosmos_qa                                
C4 openbookqa     https://huggingface.co/datasets/allenai/openbookqa
C5 mmluMA      https://huggingface.co/datasets/Obsismc/mmlu-multi_answers  
C6 sata-bench        https://huggingface.co/datasets/sata-bench/sata-bench      

Script for Length bias is 
