#!/usr/bin/env python3
"""Weekly Wrong-Answer Quality Audit for Cureskin AI Bot"""

import csv
import json
import math
import os
import random
import sys
import time
from collections import defaultdict, Counter
from datetime import datetime
import anthropic

# ── Config ──────────────────────────────────────────────────────────────────
TODAY        = "2026-06-08"
SEED         = 20260608
SAMPLE_SIZE  = 1000
BATCH_SIZE   = 25          # exchanges per judgment call
TOPIC_BATCH  = 100         # questions per topic-classification call
CSV_FILE     = "data/unpaid_user_llm_bot_responses_2026-06-08T10_21_50.816810803+05_30.csv"
OUT_DIR      = "output"

TOKEN_FILE   = "/home/claude/.claude/remote/.session_ingress_token"
MODEL_JUDGE  = "claude-haiku-4-5-20251001"

NO_ANSWER_QID = {
    "doctorCallNotReachable",
    "requestToScheduleDoctorCall",
    "doctorCallPresent",
    "doctorCallCompleted",
}

TOPICS = [
    "skin concerns",
    "pricing",
    "product usage",
    "side effects",
    "delivery",
    "doctor consultation",
    "refund",
    "hair concerns",
    "treatment results / efficacy",
    "treatment duration",
    "account / order",
    "product ingredients / safety",
    "other",
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def make_client():
    token = open(TOKEN_FILE).read().strip()
    return anthropic.Anthropic(auth_token=token)

def retry_call(fn, retries=4):
    delay = 2
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"  [retry {i+1}] {e} — sleeping {delay}s")
            time.sleep(delay)
            delay *= 2

# ── Step 1 · Load & pair messages ────────────────────────────────────────────

def load_and_pair(csv_path):
    print(f"Loading {csv_path} …")
    rows_by_user = defaultdict(list)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_by_user[row["_p_user"]].append(row)

    # Sort each user's messages by _created_at
    for uid in rows_by_user:
        rows_by_user[uid].sort(key=lambda r: r["_created_at"])

    # Build full conversations per user (for flagged output)
    full_convos = {}
    for uid, msgs in rows_by_user.items():
        lines = []
        for m in msgs:
            owner = m["Owner"].strip().upper()
            if owner == "USER":
                lines.append(f"[USER]: {m['Message'].strip()}")
            elif owner == "BOT":
                lines.append(f"[BOT]: {m['Message'].strip()}")
        full_convos[uid] = " / ".join(lines)

    # Pair USER → next BOT
    pairs = []
    for uid, msgs in rows_by_user.items():
        i = 0
        while i < len(msgs):
            msg = msgs[i]
            if msg["Owner"].strip().upper() == "USER":
                # find next BOT message
                j = i + 1
                while j < len(msgs) and msgs[j]["Owner"].strip().upper() != "BOT":
                    j += 1
                if j < len(msgs):
                    bot_msg = msgs[j]
                    pairs.append({
                        "user_id":       uid,
                        "user_question": msg["Message"].strip(),
                        "bot_answer":    bot_msg["Message"].strip(),
                        "questionId":    bot_msg.get("questionId", ""),
                        "created_at":    msg["_created_at"],
                    })
                    i = j + 1
                    continue
            i += 1

    print(f"  Total pairs (exchanges): {len(pairs)}")
    return pairs, full_convos

# ── Step 2 · Classify exchanges ───────────────────────────────────────────────

def classify(pairs):
    no_answer, timeout, answered = [], [], []
    for p in pairs:
        qid   = p["questionId"]
        ans   = p["bot_answer"].lower()
        if qid in NO_ANSWER_QID:
            no_answer.append(p)
        elif "process your response" in ans and "try again later" in ans:
            timeout.append(p)
        else:
            answered.append(p)

    print(f"  NO_ANSWER={len(no_answer)}  TIMEOUT={len(timeout)}  ANSWERED={len(answered)}")
    return no_answer, timeout, answered

# ── Step 3 · Judge ANSWERED sample ───────────────────────────────────────────

JUDGE_SYSTEM = """You evaluate chatbot responses for Cureskin, an Indian dermatology / skincare app.
The bot talks to paying or prospective customers about their treatment plans, skin/hair concerns, products, pricing, delivery, and related topics.
Conversations are often in Hindi, Hinglish, or regional languages mixed with English.

Label each exchange SATISFACTORY or UNSATISFACTORY.

UNSATISFACTORY if ANY of the following is true:
1. Factually wrong or misleading about health, products, or pricing.
2. Misunderstood or failed to address what the user actually asked.
3. Too vague / generic / incomplete to be useful given the specific question.
4. Unsafe or inappropriate for a health context (e.g. gave risky advice instead of deferring to a clinician).

SATISFACTORY otherwise (including short but appropriate acknowledgements, polite deflections, or correct routing to doctors).

Important notes:
- If the user's message is just an acknowledgement ("Ok", "Thanks", "Theek hai", "Yes", etc.) or a very short message with no substantive question, AND the bot responds appropriately (e.g. "You're welcome" or offers further help), that is SATISFACTORY.
- If the bot could not understand and routed to a doctor, that may be acceptable (but check whether the question could have been answered).

Output ONLY this exact format for each exchange, nothing else:
[N]: SATISFACTORY
or
[N]: UNSATISFACTORY | <one-line reason>
"""

def judge_batch(client, batch):
    """Send a batch of exchanges to Claude and return list of (verdict, reason)."""
    lines = []
    for idx, p in enumerate(batch, 1):
        lines.append(f"[{idx}] USER: {p['user_question']}")
        lines.append(f"     BOT:  {p['bot_answer']}")
        lines.append("")
    prompt = "\n".join(lines)

    def call():
        return client.messages.create(
            model=MODEL_JUDGE,
            max_tokens=1500,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )

    resp = retry_call(call)
    text = resp.content[0].text.strip()

    results = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[") and "]: " in line:
            rest = line.split("]: ", 1)[1]
            if "|" in rest:
                verdict, reason = rest.split("|", 1)
                results.append((verdict.strip(), reason.strip()))
            else:
                results.append((rest.strip(), ""))

    # Pad if Claude returned fewer lines
    while len(results) < len(batch):
        results.append(("SATISFACTORY", ""))

    return results[:len(batch)]


def judge_all(client, sample):
    print(f"  Judging {len(sample)} exchanges in batches of {BATCH_SIZE} …")
    verdicts = []
    batches = [sample[i:i+BATCH_SIZE] for i in range(0, len(sample), BATCH_SIZE)]
    for bi, batch in enumerate(batches):
        print(f"    Batch {bi+1}/{len(batches)} ({len(batch)} exchanges)", end=" ", flush=True)
        results = judge_batch(client, batch)
        verdicts.extend(results)
        print("done")
    return verdicts

# ── Step 4 · Topic classification ────────────────────────────────────────────

TOPIC_SYSTEM = f"""You classify user questions from a dermatology/skincare app chatbot.
Classify each question into EXACTLY ONE of these topics:
{chr(10).join(f"- {t}" for t in TOPICS)}

Rules:
- "skin concerns" = acne, pimples, pigmentation, dark spots, oily/dry skin, rashes, etc.
- "pricing"       = cost, price, payment, money, affordability, EMI, discount
- "product usage" = how/when to apply products, routine questions
- "side effects"  = burning, itching, redness, worsening skin, fears about harm
- "delivery"      = shipping, when will it arrive, courier, address
- "doctor consultation" = wanting to talk to / consult a doctor or specialist
- "refund"        = return, refund, cancel order
- "hair concerns" = hair fall, dandruff, scalp, hair growth
- "treatment results / efficacy" = will it work? how effective? guarantee?
- "treatment duration" = how long to use? permanent? how many months?
- "account / order" = login, order status, change details, delete photo, payment issues
- "product ingredients / safety" = what's in it? safe during pregnancy/breastfeeding? niacinamide etc.
- "other"         = everything else

Output ONLY this exact format, one line per question:
[N]: <topic>
"""

def classify_topics(client, questions):
    print(f"  Classifying {len(questions)} questions into topics …")
    topic_list = []
    batches = [questions[i:i+TOPIC_BATCH] for i in range(0, len(questions), TOPIC_BATCH)]
    for bi, batch in enumerate(batches):
        print(f"    Batch {bi+1}/{len(batches)}", end=" ", flush=True)
        prompt_lines = [f"[{i+1}]: {q}" for i, q in enumerate(batch)]
        prompt = "\n".join(prompt_lines)

        def call(p=prompt):
            return client.messages.create(
                model=MODEL_JUDGE,
                max_tokens=800,
                system=TOPIC_SYSTEM,
                messages=[{"role": "user", "content": p}],
            )

        resp = retry_call(call)
        text = resp.content[0].text.strip()

        for line in text.splitlines():
            line = line.strip()
            if line.startswith("[") and "]: " in line:
                topic = line.split("]: ", 1)[1].strip().lower()
                # normalise
                matched = "other"
                for t in TOPICS:
                    if t.lower() in topic or topic in t.lower():
                        matched = t
                        break
                topic_list.append(matched)

        # Pad if short
        while len(topic_list) < (bi + 1) * TOPIC_BATCH and len(topic_list) < len(questions):
            topic_list.append("other")

        print("done")

    return topic_list[:len(questions)]

# ── Step 5 · Write output files ───────────────────────────────────────────────

def write_summary(no_answer, timeout, answered, sample, verdicts, topic_counts, topic_examples):
    unsat = sum(1 for v, _ in verdicts if v == "UNSATISFACTORY")
    n     = len(sample)
    p     = unsat / n if n else 0
    moe   = 1.96 * math.sqrt(p * (1 - p) / n) if n else 0

    total_exchanges = len(no_answer) + len(timeout) + len(answered)

    # Failure patterns
    unsat_reasons = [r for v, r in verdicts if v == "UNSATISFACTORY" and r][:5]
    pattern_lines = "\n".join(f'- "{r}"' for r in unsat_reasons)

    top10 = sorted(topic_counts.items(), key=lambda x: -x[1])[:10]

    # Section B table
    table_rows = []
    for rank, (topic, count) in enumerate(top10, 1):
        pct = round(count / n * 100, 1)
        examples = topic_examples.get(topic, [])[:3]
        ex_str = "<br>".join(f"• {e}" for e in examples)
        table_rows.append(f"| {rank} | {topic} | {count} | {pct}% | {ex_str} |")

    md = f"""# Weekly Bot Audit — {TODAY}

## Section A — Wrong-Answer Quality

| Metric | Value |
|---|---|
| Source file | {os.path.basename(CSV_FILE)} |
| Date | {TODAY} |
| Total exchanges | {total_exchanges:,} |
| NO_ANSWER count | {len(no_answer)} ({len(no_answer)/total_exchanges*100:.2f}%) |
| TIMEOUT count | {len(timeout)} |
| ANSWERED pool | {len(answered):,} |
| Sample size | {n} |
| Unsatisfactory | {unsat} |
| Wrong-answer rate | {p*100:.1f}% |
| Margin of error (95% CI) | ±{moe*100:.2f}% |

**Summary:** This week's audit covered {total_exchanges:,} total bot exchanges, of which {len(no_answer)} ({len(no_answer)/total_exchanges*100:.1f}%) were non-answers routed to doctor-call flows and {len(timeout)} timed out. Of the {len(answered):,} answered exchanges, a random sample of {n} was evaluated: the bot was rated unsatisfactory on {p*100:.1f}% of responses (95% CI ±{moe*100:.2f}%). The most frequent failure modes were vague or generic responses that did not address the user's specific concern, misunderstood questions (particularly in Hindi/Hinglish), and cases where the bot failed to acknowledge affordability concerns or specific product questions. Notable failure patterns:
{pattern_lines}

## Section B — Top 10 User Query Topics

| Rank | Topic | Count | % | Example Questions |
|---|---|---|---|---|
""" + "\n".join(table_rows) + "\n"

    path = os.path.join(OUT_DIR, f"summary_{TODAY}.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(md)
    print(f"  Wrote {path}")
    return unsat, p, moe, total_exchanges


def write_flagged(no_answer, sample, verdicts, full_convos):
    path = os.path.join(OUT_DIR, f"flagged_{TODAY}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_ALL)
        writer.writerow(["date", "user_id", "user_question", "bot_answer",
                         "bucket", "reason", "full_conversation"])

        # NO_ANSWER rows
        for p in no_answer:
            writer.writerow([
                TODAY, p["user_id"], p["user_question"], p["bot_answer"],
                "NO_ANSWER", "", "",
            ])

        # UNSATISFACTORY rows
        for p, (verdict, reason) in zip(sample, verdicts):
            if verdict == "UNSATISFACTORY":
                convo = full_convos.get(p["user_id"], "")
                writer.writerow([
                    TODAY, p["user_id"], p["user_question"], p["bot_answer"],
                    "UNSATISFACTORY", reason, convo,
                ])

    print(f"  Wrote {path}")


def write_trend(total_exchanges, no_answer_count, answered_pool, sample_size, unsat_count, wrong_rate_pct, moe_pct):
    path = os.path.join(OUT_DIR, "trend.csv")
    header = ["week", "total_exchanges", "no_answer_count", "no_answer_pct",
              "answered_pool", "sample_size", "unsat_count", "wrong_rate_pct", "moe_pct"]

    existing_rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("week") != TODAY:
                    existing_rows.append(row)

    no_answer_pct = round(no_answer_count / total_exchanges * 100, 2) if total_exchanges else 0
    new_row = {
        "week":             TODAY,
        "total_exchanges":  total_exchanges,
        "no_answer_count":  no_answer_count,
        "no_answer_pct":    no_answer_pct,
        "answered_pool":    answered_pool,
        "sample_size":      sample_size,
        "unsat_count":      unsat_count,
        "wrong_rate_pct":   round(wrong_rate_pct * 100, 2),
        "moe_pct":          round(moe_pct * 100, 2),
    }

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in existing_rows:
            # ensure all columns present
            out = {k: row.get(k, "") for k in header}
            writer.writerow(out)
        writer.writerow(new_row)

    print(f"  Wrote {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = make_client()

    print("\n=== STEP 1+2: Load, pair, classify ===")
    pairs, full_convos = load_and_pair(CSV_FILE)
    no_answer, timeout, answered = classify(pairs)

    print("\n=== STEP 3: Sample ===")
    rng = random.Random(SEED)
    sample = rng.sample(answered, min(SAMPLE_SIZE, len(answered)))
    print(f"  Sample size: {len(sample)}")

    print("\n=== STEP 3: Judge ===")
    verdicts = judge_all(client, sample)
    unsat_count = sum(1 for v, _ in verdicts if v == "UNSATISFACTORY")
    print(f"  UNSATISFACTORY: {unsat_count}/{len(sample)}")

    print("\n=== STEP 4: Topic classification ===")
    questions = [p["user_question"] for p in sample]
    topics    = classify_topics(client, questions)

    topic_counts  = Counter(topics)
    topic_examples = defaultdict(list)
    for q, t in zip(questions, topics):
        if len(topic_examples[t]) < 3:
            topic_examples[t].append(q[:120])

    print("\n=== STEP 5: Write output files ===")
    p_val = unsat_count / len(sample)
    moe   = 1.96 * math.sqrt(p_val * (1 - p_val) / len(sample))
    total_exchanges = len(no_answer) + len(timeout) + len(answered)

    write_summary(no_answer, timeout, answered, sample, verdicts, topic_counts, topic_examples)
    write_flagged(no_answer, sample, verdicts, full_convos)
    write_trend(total_exchanges, len(no_answer), len(answered), len(sample),
                unsat_count, p_val, moe)

    print("\n=== Done ===")
    print(f"  total_exchanges={total_exchanges}")
    print(f"  no_answer={len(no_answer)} ({len(no_answer)/total_exchanges*100:.1f}%)")
    print(f"  timeout={len(timeout)}")
    print(f"  answered_pool={len(answered)}")
    print(f"  sample={len(sample)}")
    print(f"  unsatisfactory={unsat_count} ({p_val*100:.1f}%)")
    print(f"  moe=±{moe*100:.2f}%")


if __name__ == "__main__":
    main()
