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

# =====================================================
# Answer extraction
# =====================================================
# ✅ Explicit multi-answer regex: capture 1-7 letters A-P, possibly separated by commas, spaces, semicolons
ALLOWED_LETTERS = list("ABCDEFGHIJKLMNOP")

EXPLICIT_MULTI_ANS_REGEX = re.compile(
    r"""
    (?:^|\n|\r)                       # start or new line
    \s*
    (?:FINAL\s+ANSWER
      |CORRECT\s+ANSWER
      |ANSWER
      |OUTPUT
    )
    \s*[:\-is]*\s*
    ["']?
    (?P<ans>
        (?:[A-P])                     # first option
        (?:\s*[,;/\s]\s*[A-P])*       # optional more
    )
    ["']?
    (?=\s*$|\s*\n)                    # stop at line end
    """,
    re.IGNORECASE | re.VERBOSE
)

# 2️⃣ Line-level option marker (strict)
OPTION_LINE_REGEX = re.compile(
    r'^\s*["\']?\s*([A-P])\s*["\']?\s*$',
    re.IGNORECASE
)

# 3️⃣ Comma-separated short line: "C, D, E"
SHORT_LIST_REGEX = re.compile(
    r'^\s*["\']?\s*([A-P](?:\s*,\s*[A-P]){1,6})\s*["\']?\s*$',
    re.IGNORECASE
)

# 4️⃣ Lines to ignore entirely
IGNORE_LINE_REGEX = re.compile(
    r'(TASK\s*:|PASSAGE\s*:|SENT\s*\d+|PARAGRAPH\s*:)',
    re.IGNORECASE
)


def extract_prediction(text):
    """
    Strict multi-answer extractor.
    Returns concatenated uppercase letters (e.g. 'ACD').
    Returns 'N' if nothing valid found.
    """

    if not isinstance(text, str):
        return "N"

    text = text.upper().replace('\t', ' ')

    # 1️⃣ Explicit answer (highest confidence)
    m = EXPLICIT_MULTI_ANS_REGEX.search(text)
    if m:
        letters = re.findall(r'[A-P]', m.group('ans'))
        if letters:
            return ''.join(dict.fromkeys(letters))

    # 2️⃣ Line-by-line analysis (prevents passage bleed)
    preds = []
    for line in text.splitlines():
        line = line.strip()

        if not line or IGNORE_LINE_REGEX.search(line):
            continue

        # "C, D, E, F"
        m = SHORT_LIST_REGEX.match(line)
        if m:
            letters = re.findall(r'[A-P]', m.group(1))
            return ''.join(dict.fromkeys(letters))

        # Single-letter line: "B"
        m = OPTION_LINE_REGEX.match(line)
        if m:
            preds.append(m.group(1))

    if preds:
        return ''.join(dict.fromkeys(preds))

    return "N"
    
# =====================================================
# CONFIG
# =====================================================
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
    "./Benchmarks/sata"
]
RESULTS_DIR = "./Results"

MAX_TOKENS = 20
NPU_IDS = [0,1,2,3,4,5,6,7]
MC_SAMPLES = 2   # Monte Carlo permutations per correct option

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
    for l in ALLOWED_LETTERS:
        if f"option{l}" in row and pd.notna(row[f"option{l}"]):
            if str(row[f"option{l}"]).strip():
                letters.append(l)
    return letters

# =====================================================
# Prompt
# =====================================================
def build_prompt(question, options, letters, passage):
    text = ""
    for l in letters:
        text += f"{l}: {options[l]}\n"

    passage_text = f"passage: {passage}\n\n" if passage else ""

    return (
        "Instruction: Solve the multiple-answer multiple choice question based on the given {passage}. "
        "ONLY write the answer in the format 'Answer: X', where X is the uppercase letters of the correct options,for example, if the correct options are A, B, and C, the answer should be 'ABC'. "
        "Do not add explanations. Use uppercase letters.\n\n"
        "Example:\n"
        "passage: <b>Sent 1: </b>Born in Moscow, Pushkin published his first poem at the age of fifteen.<br><b>Sent 2: </b>By the time he finished school as part of the first graduating class of the prestigious Imperial Lyceum in Tsarskoe Selo near Saint Petersburg, his talent was already widely recognized within the Russian literary scene.<br><b>Sent 3: </b>After school, Pushkin plunged into the vibrant and raucous intellectual youth culture of the capital, Saint Petersburg.<br><b>Sent 4: </b>In 1820 he published his first long poem, Ruslan and Lyudmila, amidst much controversy about its subject and style.<br><b>Sent 5: </b>Pushkin gradually became committed to social reform and emerged as a spokesman for literary radicals.<br><b>Sent 6: </b>This angered the government, and led to his transfer from the capital (1820).<br><b>Sent 7: </b>He went to the Caucasus and to the Crimea, then to Kamenka and Chisinau, where he became a Freemason\n"
        "Question: What was going on with Pushkin in 1820?	\n"
        "A: He had angered the government with his poem  Ruslan and Lyudmila and it led to his transfer from the capital\nB: He was being transferred from the capital\nC: He published his first long poem and was transferred from the capital after angering the government\nD: He became a freemason\nE: He was moving to the united states\n"
        "Answer: ABC\n\n"
        "Task:\n"
        f"{passage_text}"
        f"Question: {question}\n"
        f"{text}"
        "Answer: "
    )

# =====================================================
# Positional Bias Case Generator
# =====================================================
def generate_cases(df):
    cases = []

    for _, row in df.iterrows():
        letters = get_available_letters(row)
        options = {l: row[f"option{l}"] for l in letters}
        correct_set = parse_answer_set(row["answer"])
        correct_set = [a for a in correct_set if a in letters]
        if not correct_set:
            continue

        passage = row.get("passage", "")

        # ---------- SINGLE ANSWER ----------
        if len(correct_set) == 1:
            c = correct_set[0]
            for pos in letters:
                perm = [l for l in letters if l != c]
                perm.insert(letters.index(pos), c)

                rec = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer_set": "".join(correct_set),
                    "target_answer": c,
                    "target_position": pos,
                    "passage": passage
                }

                for i,l in enumerate(letters):
                    rec[f"option{l}"] = options[perm[i]]

                cases.append(rec)

        # ---------- MULTI ANSWER ----------
        else:
            for target in correct_set:
                for _ in range(MC_SAMPLES):
                    perm = letters.copy()
                    random.shuffle(perm)

                    rec = {
                        "id": row["id"],
                        "question": row["question"],
                        "answer_set": "".join(correct_set),
                        "target_answer": target,
                        "target_position": perm.index(target),
                        "passage": passage
                    }

                    for i,l in enumerate(letters):
                        rec[f"option{l}"] = options[perm[i]]

                    rec["target_position"] = letters[perm.index(target)]
                    cases.append(rec)

    return cases

# =====================================================
# Worker
# =====================================================
def worker(npu_id, batch, counter, out_path, model_path):
    device = f"npu:{npu_id}"
    torch.npu.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True
    ).to(device).eval()

    records = []

    for q in batch:
        letters = sorted([k[-1] for k in q if k.startswith("option")])
        prompt = build_prompt(
            q["question"],
            {l: q[f"option{l}"] for l in letters},
            letters,
            q.get("passage", "")
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)
        gen = text[len(prompt):]

        q["predicted"] = extract_prediction(gen)
        q["raw_output"] = gen

        records.append(q)

        with counter.get_lock():
            counter.value += 1

    pd.DataFrame(records).to_csv(out_path, index=False)

# =====================================================
# Main
# =====================================================
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    os.environ["HCCL_VISIBLE_DEVICES"] = ",".join(map(str, NPU_IDS))

    for BENCHMARK in BENCHMARK_DIRS:
        for csv_file in os.listdir(BENCHMARK):
            if not csv_file.endswith(".csv"):
                continue

            df = pd.read_csv(os.path.join(BENCHMARK, csv_file))
            cases = generate_cases(df)
            
            benchmark = os.path.basename(BENCHMARK)
            base = os.path.splitext(csv_file)[0]
            for model_path in LLMs:
                model_name = os.path.basename(model_path)
                out_dir = os.path.join(RESULTS_DIR, benchmark, 'PB')
                os.makedirs(out_dir, exist_ok=True)

                out_file = os.path.join(out_dir, f"{base}-{model_name}.csv")

                batches = [cases[i::len(NPU_IDS)] for i in range(len(NPU_IDS))]
                counter = mp.Value("i", 0)
                procs = []

                for i,b in enumerate(batches):
                    if not b:
                        continue
                    tmp = f"{out_file}.tmp{i}"
                    p = mp.Process(
                        target=worker,
                        args=(NPU_IDS[i], b, counter, tmp, model_path)
                    )
                    p.start()
                    procs.append(p)

                with tqdm(total=len(cases), desc=f"{model_name} PB") as bar:
                    prev = 0
                    while any(p.is_alive() for p in procs):
                        cur = counter.value
                        bar.update(cur - prev)
                        prev = cur
                        time.sleep(0.3)
                    bar.update(counter.value - prev)

                for p in procs:
                    p.join()

                dfs = []
                for i in range(len(NPU_IDS)):
                    tmp = f"{out_file}.tmp{i}"
                    if os.path.exists(tmp):
                        dfs.append(pd.read_csv(tmp))
                        os.remove(tmp)

                # Concatenate all temporary CSVs
                final_df = pd.concat(dfs, ignore_index=True)

                # Rename answer_set to answer for final CSV
                if "answer_set" in final_df.columns:
                    final_df = final_df.rename(columns={"answer_set": "answer"})

                # Reorder columns: answer first, then options, then positional + predicted
                ordered_cols = (
                    ["id", "question", "answer"] +  # renamed column
                    [f"option{l}" for l in ALLOWED_LETTERS if f"option{l}" in final_df.columns] +
                    ["target_answer", "target_position", "predicted", "raw_output"]
                )

                # Keep only existing columns to avoid KeyError
                ordered_cols = [c for c in ordered_cols if c in final_df.columns]

                # Save final CSV
                final_df = final_df[ordered_cols]
                final_df.to_csv(out_file, index=False)

                print(f"[OK] {out_file}")

                os.system("pkill -9 -f python runC6_PB.py")
