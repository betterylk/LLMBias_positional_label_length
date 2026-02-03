import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import re
import time
import random

# -------------------------
# Regex definitions for prediction
# -------------------------
# ✅ Explicit multi-answer regex: capture 1-7 letters A-G, possibly separated by commas, spaces, semicolons
EXPLICIT_MULTI_ANS_REGEX = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|output)\s*[:is]*\s*([A-G\s,;]{1,20})",
    re.IGNORECASE
)

OPTION_LINE_REGEX = re.compile(
    r"^\s*([A-G])[\.\:\)]", re.IGNORECASE
)

def extract_prediction(text):
    """
    Robust multi-answer extraction.
    Handles messy outputs like:
      '" ABC\nExample...' -> 'ABC'
      ' C, D' -> 'CD'
      'Answer: B, D' -> 'BD'
    Returns uppercase letters in order, no duplicates.
    """
    if not isinstance(text, str):
        return "N"

    # Clean obvious noise
    text_clean = text.upper().replace('"', '').replace("'", '').replace('\t', ' ')

    # 1️⃣ Explicit answer
    match = EXPLICIT_MULTI_ANS_REGEX.search(text_clean)
    if match:
        span = match.group(1)
        # Extract all letters A-G ignoring spaces, commas, semicolons
        letters = re.findall(r"[A-G]", span)
        if letters:
            return "".join(dict.fromkeys(letters))

    # 2️⃣ Option-line fallback
    preds = []
    for line in text_clean.splitlines():
        m = OPTION_LINE_REGEX.match(line.strip())
        if m and m.group(1) not in preds:
            preds.append(m.group(1))
    if preds:
        return "".join(preds)

    # 3️⃣ Anywhere in text fallback
    letters = re.findall(r"[A-G]", text_clean)
    if letters:
        return "".join(dict.fromkeys(letters))

    return "N"

# ---------------- CONFIG ----------------
LLMs = [
    # "Models/Llama-3.2-1B-Instruct",
    # "Models/Llama-3.2-3B-Instruct",
    # "Models/Llama-2-7b-chat-hf",
    # "Models/Llama-2-7b-hf",
    # "Models/Qwen2.5-0.5B-Instruct",
    # "Models/Qwen2.5-1.5B-Instruct",
    # "Models/Qwen2.5-3B-Instruct",
    # "Models/Qwen2.5-7B-Instruct",
    # "Models/Qwen3-0.6B",
    # "Models/Qwen3-4B-Instruct-2507",
    # "Models/Qwen3-4B-Thinking-2507",
    # "Models/Qwen3-8B",
    # "Models/phi-2",
    # "Models/SmolLM3-3B",
    # "Models/Ministral-3b-instruct",
    "Models/Llama-2-13b-chat-hf",
]
BENCHMARK_DIRS = [
    "./Benchmarks/mmluMA"
]
RESULTS_DIR = "./Results"

MC_SAMPLES = 2   # Monte Carlo permutations per correct option

MAX_TOKENS = 20
NPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
ALLOWED_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
ALLOWED_POS = [1, 2, 3, 4, 5, 6, 7]
RANDOM_SEED = 42

os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================
# Utilities
# =====================================================
# Input: "ACD" or "B"; Output: ["A","C","D"]
def parse_answer_set(ans):
    if isinstance(ans, str):
        return list(ans.strip().upper())
    return []

def get_available_letters(row):
    letters = []
    for l in list("ABCDEFG"):
        if f"option{l}" in row and pd.notna(row[f"option{l}"]):
            if str(row[f"option{l}"]).strip():
                letters.append(l)
    return letters

# ---------------- PROMPT ----------------
def build_prompt(question, options):
    text = ""
    for i in ALLOWED_POS:
        val = options.get(i, None)
        if val is None or pd.isna(val):
            continue
        text += f"{val}\n"

    return (
        "Instruction: Solve the multiple-answer multiple choice question based on the given options. "
        "ONLY write the answer in the format 'Answer: X', where X is the uppercase letters of the correct options,for example, if the correct options are A, B, and C, the answer should be 'ABC'. "
        "Do not add explanations. Use uppercase letters.\n\n"
        "Example 1:\n"
        "Question: What are some benefits of consuming nuts as part of a healthy diet?	\n"
        "A: They can help reduce cholesterol levels\nB: They provide essential vitamins and minerals\nC: They are a good source of protein\nD: They are high in carbohydrates\n"
        "Answer: ABC\n\n"
        "Example 2:\n"
        "Question: What are some benefits of using renewable energy sources?\n"
        "A: Reduced greenhouse gas emissions\nB: Lower energy bills for consumers\nC: Limited availability of energy resources\nD: Decreased dependency on fossil fuels\n"
        "Answer: ABD\n\n"
        "Task:\n"
        f"Question: {question}\n"
        f"{text}"
        "Answer: "
    )

# ---------------- LABEL-BIAS GENERATOR (multi-answer compatible) ----------------
def generate_cases(df, seed=42):
    random.seed(seed)
    labels = ALLOWED_LETTERS
    all_cases = []  # flat list, no label-bias needed

    for _, row in df.iterrows():
        # Parse multi-answer set, e.g., "ACD" -> ['A','C','D']
        correct_set = list(str(row["answer"]).strip().upper())

        # Original option texts
        option_texts = {l: row.get(f"option{l}", None) for l in labels}

        # 1️⃣ Loop over correct options
        for correct_label in correct_set:
            correct_text = option_texts[correct_label]

            # Get distractors (other options that exist)
            distractors = [option_texts[l] for l in labels if l != correct_label and pd.notna(option_texts[l])]

            # 2️⃣ Monte Carlo sampling
            for _ in range(MC_SAMPLES):
                # Shuffle distractors
                random.shuffle(distractors)

                # Combine correct + distractors
                options_pool = [correct_text] + distractors

                # Assign labels randomly to options
                assigned_labels = labels[:len(options_pool)]
                random.shuffle(assigned_labels)

                # Make sure correct option is somewhere in assigned labels
                # Find index of correct option
                correct_idx = options_pool.index(correct_text)
                # Swap assigned label for correct option to correct_idx
                assigned_labels[correct_idx] = assigned_labels[0]
                assigned_labels[0] = assigned_labels[correct_idx]

                # Build final options mapping
                final_options = {}
                for i, (opt_text, lbl) in enumerate(zip(options_pool, assigned_labels), start=1):
                    final_options[f"option{i}"] = f"{lbl}. {opt_text}"

                # 3️⃣ Build case
                final_case = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": "".join(correct_set),  # full multi-answer set
                    "target_answer": correct_label,  # for tracking which correct we focus on
                    "predicted": "",
                }
                final_case.update(final_options)

                all_cases.append(final_case)

    return all_cases

# ---------------- WORKER ----------------
def worker(npu_id, questions, progress_counter, output_path, model_path, max_tokens, debug=False):
    device = f"npu:{npu_id}"
    torch.npu.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()

    records = []
    for idx, q in enumerate(questions):
        # print(q)
        prompt = build_prompt(q["question"], {l: q.get(f"option{l}", None) for l in ALLOWED_POS})
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        input_len = len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True))
        generated_part = full_text[input_len:]

        # ----------------- NEW PREDICTION LOGIC -----------------
        pred = extract_prediction(generated_part)

        q["predicted"] = pred
        q["raw_output"] = generated_part
        records.append(q)

        # -------- DEBUG (ONLY ONCE) --------
        if debug and npu_id == 0 and idx == 0:
            preview = prompt if len(prompt) < 500 else "..." + prompt[-500:]
            print("\n================ DEBUG SAMPLE ================")
            print("MODEL:", model_path)
            print("DEVICE:", device)
            print("\n--- PROMPT ---\n", preview)
            print("\n--- FULL OUTPUT ---\n", full_text)
            print("\n--- GENERATED SUFFIX ---\n", generated_part)
            print("\nFINAL ANSWER:", pred)
            print("============================================\n")

        with progress_counter.get_lock():
            progress_counter.value += 1

    pd.DataFrame(records).to_csv(output_path, index=False)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["HCCL_VISIBLE_DEVICES"] = ",".join(map(str, NPU_IDS))

    for BENCHMARK in BENCHMARK_DIRS:
        for csv_file in os.listdir(BENCHMARK):
            if not csv_file.endswith(".csv"):
                continue

            file_path = os.path.join(BENCHMARK, csv_file)
            df = pd.read_csv(file_path)

            benchmark = os.path.basename(BENCHMARK)
            base = os.path.splitext(csv_file)[0]

            for model_path in LLMs:
                model_name = os.path.basename(model_path)

                all_questions = generate_cases(df)

                out_dir = os.path.join(RESULTS_DIR, benchmark, "LaB")
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, f"{base}-{model_name}.csv")

                # Split across NPUs
                batches = [all_questions[i::len(NPU_IDS)] for i in range(len(NPU_IDS))]
                counter = mp.Value("i", 0)
                procs = []

                for i, batch in enumerate(batches):
                    if not batch:
                        continue

                    tmp = f"{output_path}.tmp{i}"
                    debug_flag = (i == 0)   # debug once per model

                    p = mp.Process(
                        target=worker,
                        args=(NPU_IDS[i], batch, counter, tmp, model_path, MAX_TOKENS, debug_flag)
                    )
                    p.start()
                    procs.append(p)

                with tqdm(total=len(all_questions), desc=f"{model_name} | LaB-MC") as bar:
                    prev = 0
                    while any(p.is_alive() for p in procs):
                        cur = counter.value
                        if cur > prev:
                            bar.update(cur - prev)
                            prev = cur
                        time.sleep(0.3)
                    bar.update(counter.value - prev)

                for p in procs:
                    p.join()

                # Merge temp files
                dfs = []
                for i in range(len(NPU_IDS)):
                    tmp = f"{output_path}.tmp{i}"
                    if os.path.exists(tmp):
                        dfs.append(pd.read_csv(tmp))
                        os.remove(tmp)

                final_df = pd.concat(dfs, ignore_index=True)
                # Reorder columns: answer first, then options, then positional + predicted
                option_cols = [f"option{i}" for i in range(1, 8) if f"option{i}" in final_df.columns]
                ordered_cols = (
                    ["id", "question", "answer"] +  # always first
                    option_cols +                   # option1-option7 if they exist
                    ["target_answer", "predicted", "raw_output"])

                # Keep only existing columns to avoid KeyError
                ordered_cols = [c for c in ordered_cols if c in final_df.columns]

                # Save final CSV
                final_df = final_df[ordered_cols]
                final_df.to_csv(output_path, index=False)

                print(f"[OK] {output_path}")

                os.system("pkill -9 -f python runC5_LaB.py")
