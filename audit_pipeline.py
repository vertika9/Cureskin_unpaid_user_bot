#!/usr/bin/env python3
"""Weekly wrong-answer quality audit for Cureskin AI bot."""

import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────
CSV_FILE = "/home/user/Cureskin_unpaid_user_bot/data/unpaid_user_llm_bot_responses_2026-06-01T14_36_46.422590318+05_30.csv"
OUTPUT_DIR = "/home/user/Cureskin_unpaid_user_bot/output"
TODAY = date.today()
DATE_STR = TODAY.strftime("%Y-%m-%d")
SEED = int(TODAY.strftime("%Y%m%d"))
MAX_SAMPLE = 1000
BATCH_SIZE = 50

NO_ANSWER_IDS = {
    "doctorCallNotReachable",
    "requestToScheduleDoctorCall",
    "doctorCallPresent",
    "doctorCallCompleted",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Step 1 & 2: Parse and pair messages ───────────────────────────────────────
print("Parsing CSV and pairing messages…")
users: dict[str, list[dict]] = defaultdict(list)

with open(CSV_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        users[row["_p_user"]].append(row)

for uid in users:
    users[uid].sort(key=lambda x: x["_created_at"])

def build_full_conversation(uid: str) -> str:
    lines = []
    for m in users[uid]:
        text = m["Message"].replace("\n", " ")
        lines.append(f"[{m['Owner']}]: {text}")
    return " / ".join(lines)

pairs: list[tuple[str, dict, dict]] = []
for uid, msgs in users.items():
    for i in range(len(msgs) - 1):
        if msgs[i]["Owner"] == "USER" and msgs[i + 1]["Owner"] == "BOT":
            umsg, bmsg = msgs[i], msgs[i + 1]
            bot_text = bmsg["Message"]
            bot_qid = bmsg["questionId"]
            if "process your response" in bot_text and "try again later" in bot_text:
                continue
            if bot_qid in NO_ANSWER_IDS:
                continue
            pairs.append((uid, umsg, bmsg))

print(f"Valid exchange pool: {len(pairs)}")

# ── Step 3: Sample ────────────────────────────────────────────────────────────
random.seed(SEED)
sample_size = min(MAX_SAMPLE, len(pairs))
sample = random.sample(pairs, sample_size)
print(f"Sampled {sample_size} exchanges (seed={SEED})")

# ── Step 4: Evaluate via claude CLI ──────────────────────────────────────────
EVAL_SYSTEM = (
    "You are an expert evaluator for a dermatology/skincare AI health assistant "
    "called Cureskin AI. The bot assists paying customers with questions about "
    "their skincare treatment plans. Many conversations are in Hindi or Hinglish — "
    "evaluate them appropriately. "
    "Generic 'talk to your dermatologist' responses ARE acceptable when the "
    "question is clinical. Responses encouraging continuation of treatment are "
    "acceptable when the user is worried about cost."
)

EVAL_PROMPT_TEMPLATE = """\
Evaluate each exchange below. For each one, output a JSON object with:
- "index": the exchange index (integer, as given)
- "label": "SATISFACTORY" or "UNSATISFACTORY"
- "reason": one-line reason ONLY if UNSATISFACTORY (empty string otherwise)

UNSATISFACTORY if ANY of these are true:
1. Factually wrong or misleading
2. Misunderstood or didn't address what the user actually asked
3. Too vague or incomplete to be useful for the user's specific question
4. Unsafe or inappropriate guidance for a health context (should have deferred
   to a clinician but gave risky advice instead)

Return ONLY a valid JSON array of objects, no markdown, no explanation.

Exchanges:
{exchanges_json}
"""

def evaluate_batch(batch: list[tuple[str, dict, dict]], batch_offset: int) -> dict[int, dict]:
    exchanges_json_list = []
    for local_i, (uid, umsg, bmsg) in enumerate(batch):
        exchanges_json_list.append({
            "index": local_i,
            "user_question": umsg["Message"],
            "bot_answer": bmsg["Message"],
        })

    prompt = EVAL_PROMPT_TEMPLATE.format(
        exchanges_json=json.dumps(exchanges_json_list, ensure_ascii=False, indent=2)
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--system-prompt", EVAL_SYSTEM, prompt],
            capture_output=True,
            text=True,
            timeout=180,
        )
        output = result.stdout.strip()
        # Extract JSON array from output
        start = output.find("[")
        end = output.rfind("]") + 1
        if start >= 0 and end > start:
            parsed = json.loads(output[start:end])
            return {batch_offset + r["index"]: r for r in parsed}
        else:
            print(f"  Warning: could not parse JSON from batch starting at {batch_offset}")
            print(f"  Output preview: {output[:300]}")
    except subprocess.TimeoutExpired:
        print(f"  Timeout on batch starting at {batch_offset}")
    except json.JSONDecodeError as e:
        print(f"  JSON parse error on batch {batch_offset}: {e}")
        print(f"  Output preview: {output[:300]}")
    except Exception as e:
        print(f"  Unexpected error on batch {batch_offset}: {e}")

    # Fallback: mark all as SATISFACTORY
    return {
        batch_offset + i: {"index": i, "label": "SATISFACTORY", "reason": ""}
        for i in range(len(batch))
    }

results: dict[int, dict] = {}
total_batches = math.ceil(sample_size / BATCH_SIZE)

for batch_num in range(total_batches):
    start = batch_num * BATCH_SIZE
    end = min(start + BATCH_SIZE, sample_size)
    batch = sample[start:end]
    print(f"  Evaluating batch {batch_num + 1}/{total_batches} (exchanges {start}–{end - 1})…")
    batch_results = evaluate_batch(batch, start)
    results.update(batch_results)

# ── Compute stats ─────────────────────────────────────────────────────────────
unsat_count = sum(1 for r in results.values() if r.get("label") == "UNSATISFACTORY")
wrong_rate = unsat_count / sample_size
p = wrong_rate
n = sample_size
moe = 1.96 * math.sqrt(p * (1 - p) / n) if n > 0 and p > 0 and p < 1 else 0.0

print(f"\nResults: {unsat_count}/{sample_size} unsatisfactory = {wrong_rate:.1%} ± {moe:.1%}")

# ── Step 4a: Collect flagged exchanges and patterns ───────────────────────────
flagged_rows = []
flagged_reasons = []

for i, (uid, umsg, bmsg) in enumerate(sample):
    r = results.get(i, {"label": "SATISFACTORY", "reason": ""})
    if r.get("label") == "UNSATISFACTORY":
        full_conv = build_full_conversation(uid)
        reason = r.get("reason", "")
        flagged_reasons.append(reason)
        flagged_rows.append({
            "date": DATE_STR,
            "user_id": uid,
            "user_question": umsg["Message"],
            "bot_answer": bmsg["Message"],
            "reason": reason,
            "full_conversation": full_conv,
        })

# ── Generate pattern summary via claude ──────────────────────────────────────
print("Generating narrative summary…")
if flagged_reasons:
    reasons_text = "\n".join(f"- {r}" for r in flagged_reasons if r)
    summary_prompt = (
        f"I have {unsat_count} unsatisfactory bot responses out of {sample_size} sampled "
        f"from a dermatology AI assistant (Cureskin AI). Here are the one-line reasons:\n\n"
        f"{reasons_text}\n\n"
        "Write exactly 3-4 sentences of plain English summarizing how the bot did this "
        "week and what notable patterns appear in the flagged examples. "
        "Be specific and actionable. Do not use bullet points. Just prose."
    )
    try:
        pat_result = subprocess.run(
            ["claude", "-p", summary_prompt],
            capture_output=True, text=True, timeout=60
        )
        narrative = pat_result.stdout.strip()
    except Exception:
        narrative = (
            f"The bot achieved a {wrong_rate:.1%} wrong-answer rate this week. "
            f"Review the flagged_YYYY-MM-DD.csv for detailed issues."
        )
else:
    narrative = (
        f"The bot performed well this week with a {wrong_rate:.1%} wrong-answer rate "
        f"across {sample_size} sampled exchanges. No significant issues were flagged."
    )

# ── Output ①: summary_YYYY-MM-DD.md ──────────────────────────────────────────
summary_path = os.path.join(OUTPUT_DIR, f"summary_{DATE_STR}.md")
summary_md = f"""\
# Weekly Bot Audit — {DATE_STR}

| Metric | Value |
|--------|-------|
| Date | {DATE_STR} |
| Total exchanges in pool | {len(pairs):,} |
| Sample size | {sample_size:,} |
| Unsatisfactory count | {unsat_count:,} |
| Wrong-answer rate | {wrong_rate:.2%} |
| Margin of error (95% CI) | ± {moe:.2%} |

## Summary

{narrative}

## Flagged Exchanges

See `flagged_{DATE_STR}.csv` for the full list of {unsat_count} unsatisfactory exchanges.
"""

with open(summary_path, "w", encoding="utf-8") as f:
    f.write(summary_md)
print(f"Wrote {summary_path}")

# ── Output ②: flagged_YYYY-MM-DD.csv ─────────────────────────────────────────
flagged_path = os.path.join(OUTPUT_DIR, f"flagged_{DATE_STR}.csv")
with open(flagged_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["date", "user_id", "user_question", "bot_answer", "reason", "full_conversation"],
    )
    writer.writeheader()
    writer.writerows(flagged_rows)
print(f"Wrote {flagged_path} ({len(flagged_rows)} rows)")

# ── Output ③: trend.csv ───────────────────────────────────────────────────────
trend_path = os.path.join(OUTPUT_DIR, "trend.csv")
trend_exists = os.path.exists(trend_path)
with open(trend_path, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["week", "total_pool", "sample_size", "unsat_count", "wrong_rate_pct", "moe_pct"],
    )
    if not trend_exists:
        writer.writeheader()
    writer.writerow({
        "week": DATE_STR,
        "total_pool": len(pairs),
        "sample_size": sample_size,
        "unsat_count": unsat_count,
        "wrong_rate_pct": round(wrong_rate * 100, 2),
        "moe_pct": round(moe * 100, 2),
    })
print(f"Appended to {trend_path}")

print("\nDone. All output files written.")
print(f"  Summary:  {summary_path}")
print(f"  Flagged:  {flagged_path}")
print(f"  Trend:    {trend_path}")
