import os
# Suppress Huawei CANN/ACL warnings completely

os.environ["ASCEND_GLOBAL_LOG_LEVEL"] = "3"   # only errors
os.environ["ACL_LOG_OUTPUT_TYPE"] = "none"    # hide ACL logs
os.environ["NPU_DISABLE_WARNINGS"] = "1"      # optional extra suppression

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import pandas as pd
import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
import logging
import re
import time
from pathlib import Path

# -------------------------
# Regex definitions for prediction
# -------------------------
EXPLICIT_ANS_REGEX = re.compile(
    r"(?:final\s+answer|correct\s+answer|answer|output)\s*[:is]*\s*[\(\[\"]*([A-E])\b",
    re.IGNORECASE
)

def extract_prediction(raw_output: str) -> str:
    """
    Extract prediction from raw_output.
    Priority:
    1) Explicit answer keywords (Answer:, Final answer, etc.) → first occurrence
    2) Lines starting with letter A-E, optional punctuation → first occurrence
    3) Return 'N' if nothing found
    """
    if not isinstance(raw_output, str):
        return "N"

    # Split output into lines
    lines = raw_output.splitlines()

    for line in lines:
        # Remove leading/trailing whitespace and quotes
        line_clean = line.strip().strip('"').strip("'")

        # 1️⃣ Explicit answer in the line
        explicit = EXPLICIT_ANS_REGEX.findall(line_clean)
        if explicit:
            return explicit[0].upper()

        # 2️⃣ Check if line starts with a letter A-E (optionally with punctuation)
        m = re.match(r"^\s*([A-Ea-e])[\.\;\:\)]?\s*(?:.*)?$", line_clean)
        if m:
            return m.group(1).upper()

    # 3️⃣ Nothing usable
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
    "Models/Qwen3-8B",
    "Models/phi-2",
    "Models/SmolLM3-3B",
    "Models/Ministral-3b-instruct",
    "Models/Llama-2-13b-chat-hf",
]
BENCHMARK_DIR = "./Benchmarks/AI2ARC"
RESULTS_DIR = "./Results"

MAX_TOKENS = 20
NPU_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
ALLOWED_LETTERS = ["A", "B", "C", "D", "E"]

os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------- PROMPT BUILDER ----------------
def build_prompt(question, options_dict):
    options_text = ""
    for letter in ALLOWED_LETTERS:
        val = options_dict.get(letter, "N/A")
        if pd.isna(val) or str(val).strip() == "":
            val = "N/A"
        options_text += f"{letter}: {val}\n"

    return (
        "Instruction: Solve the multiple choice question. "
        "ONLY write the answer in the format 'Answer: X'. "
        "Do not add explanations. Use uppercase letters A-E.\n\n"
        "Example 1:\n"
        "Question: What is the capital of France?\n"
        "A: London\nB: Paris\nC: Berlin\nD: Rome\nE: Madrid\n"
        "Answer: B\n\n"
        "Example 2:\n"
        "Question: Which planet is known as the Red Planet?\n"
        "A: Earth\nB: Venus\nC: Mars\nD: Jupiter\nE: Saturn\n"
        "Answer: C\n\n"
        "Task:\n"
        f"Question: {question}\n"
        f"{options_text}"
        "Answer: "
    )
# ---------------- POSITIONAL PERMUTATIONS ----------------
def generate_positional_permutations(df):
    perms = {pos: [] for pos in ALLOWED_LETTERS}

    for _, row in df.iterrows():
        correct_letter = row["answer"]
        correct_text = row[f"option{correct_letter}"]
        distractors = [row[f"option{l}"] for l in ALLOWED_LETTERS if l != correct_letter]

        for pos in ALLOWED_LETTERS:
            idx = ALLOWED_LETTERS.index(pos)
            options = distractors.copy()
            options.insert(idx, correct_text)

            perms[pos].append({
                "id": row["id"],
                "question": row["question"],
                "optionA": options[0],
                "optionB": options[1],
                "optionC": options[2],
                "optionD": options[3],
                "optionE": options[4],
                "answer": pos,
                "predicted": ""
            })

    return perms

# ---------------- WORKER ----------------
def worker(
    npu_id: int,
    questions: list,
    progress_counter,
    output_path: str,
    model_path: str,
    max_tokens: int,
    debug: bool = False,
):
    device = f"npu:{npu_id}"
    try:
        torch.npu.set_device(device)
    except Exception as e:
        print(f"[ERROR] Failed to set {device}: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(device).eval()

    records = []

    for idx, q in enumerate(questions):
        prompt = build_prompt(
            q["question"],
            {l: q[f"option{l}"] for l in ALLOWED_LETTERS}
        )

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

    for csv_file in os.listdir(BENCHMARK_DIR):
        if not csv_file.endswith(".csv"):
            continue

        file_path = os.path.join(BENCHMARK_DIR, csv_file)

        for model_path in LLMs:
            model_name = os.path.basename(os.path.normpath(model_path))

            df = pd.read_csv(file_path)
            perms = generate_positional_permutations(df)

            benchmark = os.path.basename(os.path.normpath(BENCHMARK_DIR))
            base = os.path.splitext(os.path.basename(file_path))[0]

            for pos, questions in perms.items():
                out_dir = os.path.join(RESULTS_DIR, benchmark, 'PB')
                os.makedirs(out_dir, exist_ok=True)

                output_path = os.path.join(
                    out_dir, f"{base}-{model_name}-PB-{pos}.csv"
                )

                batches = [questions[i::len(NPU_IDS)] for i in range(len(NPU_IDS))]
                counter = mp.Value("i", 0)
                procs = []

                for i, batch in enumerate(batches):
                    if not batch:
                        continue

                    tmp = f"{output_path}.tmp{i}"
                    debug_flag = (i == 0 and pos == "A")

                    p = mp.Process(
                        target=worker,
                        args=(NPU_IDS[i], batch, counter, tmp, model_path, MAX_TOKENS, debug_flag)
                    )
                    p.start()
                    procs.append(p)

                with tqdm(total=len(questions), desc=f"{model_name} | pos {pos}") as bar:
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

                dfs = []
                for i in range(len(NPU_IDS)):
                    tmp = f"{output_path}.tmp{i}"
                    if os.path.exists(tmp):
                        dfs.append(pd.read_csv(tmp))
                        os.remove(tmp)

                pd.concat(dfs, ignore_index=True).to_csv(output_path, index=False)
                print(f"[OK] {output_path}")

                os.system("pkill -9 -f python runC1_PB.py")