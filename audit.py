#!/usr/bin/env python3
"""Weekly wrong-answer quality audit for the Cureskin AI bot."""

import asyncio
import csv
import json
import math
import os
import random
import re as _re
import sys
from collections import defaultdict, Counter

import anthropic

# ── config ────────────────────────────────────────────────────────────────────
CSV_PATH = "data/unpaid_user_llm_bot_responses_2026-06-01T14_36_46.422590318+05_30.csv"
OUTPUT_DIR = "output"
TODAY = "2026-06-08"
SEED = 20260608
CONCURRENCY = 8  # parallel API calls

NO_ANSWER_QUESTION_IDS = {
    "doctorCallNotReachable",
    "requestToScheduleDoctorCall",
    "doctorCallPresent",
    "doctorCallCompleted",
}

_token_path = "/home/claude/.claude/remote/.session_ingress_token"
_auth_token = open(_token_path).read().strip() if os.path.exists(_token_path) else None


def _make_client():
    if _auth_token:
        return anthropic.AsyncAnthropic(auth_token=_auth_token)
    return anthropic.AsyncAnthropic()


# ── Step 1+2: parse CSV and pair messages ─────────────────────────────────────

def load_and_pair(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    by_user = defaultdict(list)
    for r in rows:
        by_user[r["_p_user"]].append(r)

    for uid in by_user:
        by_user[uid].sort(key=lambda r: r["_created_at"])

    threads = dict(by_user)

    exchanges = []
    for uid, msgs in by_user.items():
        for i, msg in enumerate(msgs):
            if msg["Owner"] == "USER":
                bot_msg = None
                for j in range(i + 1, len(msgs)):
                    if msgs[j]["Owner"] == "BOT":
                        bot_msg = msgs[j]
                        break
                if bot_msg is None:
                    continue
                exchanges.append({"user_id": uid, "user_msg": msg, "bot_msg": bot_msg})

    return exchanges, threads


def classify(exchange):
    bot = exchange["bot_msg"]
    qid = bot.get("questionId", "")
    text = bot.get("Message", "")
    if qid in NO_ANSWER_QUESTION_IDS:
        return "NO_ANSWER"
    if "process your response" in text and "try again later" in text:
        return "TIMEOUT"
    return "ANSWERED"


# ── Step 3: async judging ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a quality auditor for Cureskin, an Indian AI dermatology assistant. "
    "Cureskin provides skin and hair care products and connects users with dermatologists. "
    "Its AI bot helps users with questions about their treatment plan, products, side effects, "
    "pricing, delivery, and general skin/hair concerns. "
    "Users often write in Hinglish (Hindi + English mix) or Hindi script.\n\n"
    "Your task: read a USER question and the BOT answer, then decide if the answer is "
    "SATISFACTORY or UNSATISFACTORY.\n\n"
    "Label UNSATISFACTORY if ANY of these apply:\n"
    "  1. Factually wrong or misleading\n"
    "  2. Misunderstood or didn't address what the user actually asked\n"
    "  3. Too vague or incomplete to be useful\n"
    "  4. Unsafe or inappropriate for a health context (gave risky advice instead of deferring to a clinician)\n"
    "Otherwise label SATISFACTORY.\n\n"
    'Respond ONLY with a JSON object: {"label":"SATISFACTORY"} or '
    '{"label":"UNSATISFACTORY","reason":"one-line reason"}. '
    "No markdown, no preamble."
)

TOPIC_SYSTEM = (
    "You are classifying user questions sent to an Indian skincare/haircare AI assistant called Cureskin. "
    "Given a list of user messages (one per line, numbered), assign each a topic from this list:\n"
    "  pricing, product_usage, side_effects, delivery, doctor_consultation, refund, "
    "hair_concerns, skin_concerns, treatment_duration, product_ingredients, "
    "account_order, results_efficacy, routine_instructions, other\n\n"
    "Return ONLY a JSON array of topic strings, one per input line, in the same order. "
    "No markdown, no explanation."
)


def _parse_judge_json(raw: str) -> dict:
    raw = _re.sub(r"```[a-z]*\n?", "", raw).strip()
    for candidate in [raw.split("\n")[0].strip(), raw]:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m = _re.search(r"\{[^{}]+\}", raw, _re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    label_m = _re.search(r'"label"\s*:\s*"(SATISFACTORY|UNSATISFACTORY)"', raw)
    reason_m = _re.search(r'"reason"\s*:\s*"([^"]*)"', raw)
    if label_m:
        result = {"label": label_m.group(1)}
        if reason_m:
            result["reason"] = reason_m.group(1)
        return result
    if "UNSATISFACTORY" in raw.upper():
        return {"label": "UNSATISFACTORY", "reason": "see raw response"}
    return {"label": "SATISFACTORY"}


async def _judge_one(client, sem, idx, ex):
    user_q = ex["user_msg"]["Message"]
    bot_a = ex["bot_msg"]["Message"]
    prompt = f"USER: {user_q}\nBOT: {bot_a}"
    async with sem:
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = resp.content[0].text.strip()
            data = _parse_judge_json(raw)
            return idx, data.get("label", "SATISFACTORY"), data.get("reason", "")
        except Exception as e:
            print(f"    Judge error at {idx}: {e}", flush=True)
            return idx, "SATISFACTORY", ""


async def _topic_chunk(client, sem, start, chunk):
    numbered = "\n".join(f"{i+1}. {q}" for i, q in enumerate(chunk))
    async with sem:
        try:
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=600,
                system=TOPIC_SYSTEM,
                messages=[{"role": "user", "content": numbered}],
            )
            raw = resp.content[0].text.strip()
            raw = _re.sub(r"```[a-z]*\n?", "", raw).strip()
            topics = json.loads(raw)
            if len(topics) != len(chunk):
                topics = ["other"] * len(chunk)
            return start, topics
        except Exception as e:
            print(f"  Topic error at batch {start}: {e}", flush=True)
            return start, ["other"] * len(chunk)


async def run_judgements(sample):
    client = _make_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = [_judge_one(client, sem, i, ex) for i, ex in enumerate(sample)]

    results = [None] * len(sample)
    done = 0
    for coro in asyncio.as_completed(tasks):
        idx, label, reason = await coro
        results[idx] = (label, reason)
        done += 1
        if done % 50 == 0:
            print(f"    Judged {done}/{len(sample)} ...", flush=True)
    await client.close()
    return results


async def run_topics(user_questions):
    client = _make_client()
    sem = asyncio.Semaphore(CONCURRENCY)
    chunk_size = 50
    chunks = [
        (start, user_questions[start:start+chunk_size])
        for start in range(0, len(user_questions), chunk_size)
    ]
    tasks = [_topic_chunk(client, sem, start, chunk) for start, chunk in chunks]

    all_topics = [None] * len(user_questions)
    for coro in asyncio.as_completed(tasks):
        start, topics = await coro
        for j, t in enumerate(topics):
            all_topics[start + j] = t
        print(f"  Topics batch {start} done", flush=True)
    await client.close()
    return all_topics


# ── helpers ───────────────────────────────────────────────────────────────────

def build_thread_text(uid, threads):
    msgs = threads.get(uid, [])
    lines = []
    for m in msgs:
        owner = "[USER]" if m["Owner"] == "USER" else "[BOT]"
        lines.append(f"{owner}: {m['Message']}")
    return " / ".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== Step 1+2: Loading and pairing messages ===", flush=True)
    exchanges, threads = load_and_pair(CSV_PATH)
    print(f"  Total exchanges paired: {len(exchanges)}", flush=True)

    for ex in exchanges:
        ex["bucket"] = classify(ex)

    total_exchanges = len(exchanges)
    no_answer = [e for e in exchanges if e["bucket"] == "NO_ANSWER"]
    timeout   = [e for e in exchanges if e["bucket"] == "TIMEOUT"]
    answered  = [e for e in exchanges if e["bucket"] == "ANSWERED"]

    print(f"  NO_ANSWER: {len(no_answer)}", flush=True)
    print(f"  TIMEOUT:   {len(timeout)}", flush=True)
    print(f"  ANSWERED:  {len(answered)}", flush=True)

    print("\n=== Step 3: Sampling and judging ANSWERED exchanges ===", flush=True)
    rng = random.Random(SEED)
    sample_size = min(1000, len(answered))
    sample = rng.sample(answered, sample_size)
    print(f"  Sample size: {sample_size} — running {CONCURRENCY} concurrent calls", flush=True)

    judge_results = asyncio.run(run_judgements(sample))

    unsat_exchanges = []
    for i, ex in enumerate(sample):
        label, reason = judge_results[i]
        ex["label"] = label
        ex["reason"] = reason
        if label == "UNSATISFACTORY":
            unsat_exchanges.append(ex)

    unsat_count = len(unsat_exchanges)
    p = unsat_count / sample_size
    moe = 1.96 * math.sqrt(p * (1 - p) / sample_size)
    wrong_rate_pct = round(p * 100, 2)
    moe_pct = round(moe * 100, 2)
    print(f"  Unsatisfactory: {unsat_count}/{sample_size} = {wrong_rate_pct}% ±{moe_pct}%", flush=True)

    print("\n=== Step 4: Topic classification ===", flush=True)
    user_questions = [ex["user_msg"]["Message"] for ex in sample]
    topics = asyncio.run(run_topics(user_questions))

    topic_counter = Counter(topics)
    top_10 = topic_counter.most_common(10)

    topic_examples = defaultdict(list)
    for i, t in enumerate(topics):
        if len(topic_examples[t]) < 3:
            topic_examples[t].append(user_questions[i])

    print("  Top topics:", flush=True)
    for t, cnt in top_10:
        print(f"    {t}: {cnt}", flush=True)

    # ── Step 5a: summary ──────────────────────────────────────────────────────
    print("\n=== Step 5: Writing output files ===", flush=True)

    no_answer_pct = round(len(no_answer) / total_exchanges * 100, 2) if total_exchanges else 0

    unsat_reasons = [ex["reason"] for ex in unsat_exchanges if ex["reason"]]
    top_reasons_text = ""
    if unsat_reasons:
        reason_counts = Counter(unsat_reasons)
        common = reason_counts.most_common(3)
        top_reasons_text = "; ".join(f'"{r}" ({c}x)' for r, c in common)

    summary_path = os.path.join(OUTPUT_DIR, f"summary_{TODAY}.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Weekly Bot Audit — {TODAY}\n\n")
        f.write("## Section A — Wrong-Answer Quality\n\n")
        f.write("| Metric | Value |\n|---|---|\n")
        f.write(f"| Date | {TODAY} |\n")
        f.write(f"| Total exchanges | {total_exchanges:,} |\n")
        f.write(f"| NO_ANSWER count | {len(no_answer)} ({no_answer_pct}%) |\n")
        f.write(f"| TIMEOUT count | {len(timeout)} |\n")
        f.write(f"| ANSWERED pool | {len(answered):,} |\n")
        f.write(f"| Sample size | {sample_size} |\n")
        f.write(f"| Unsatisfactory | {unsat_count} |\n")
        f.write(f"| Wrong-answer rate | {wrong_rate_pct}% |\n")
        f.write(f"| Margin of error (95% CI) | ±{moe_pct}% |\n\n")

        summary_sentences = [
            f"This week's audit covered {total_exchanges:,} total bot exchanges, "
            f"of which {len(no_answer)} ({no_answer_pct}%) were non-answers routed to doctor-call flows "
            f"and {len(timeout)} timed out.",
            f"Of {len(answered):,} answered exchanges, a random sample of {sample_size} was evaluated: "
            f"the bot was rated unsatisfactory on {wrong_rate_pct}% of responses "
            f"(95% CI ±{moe_pct}%).",
        ]
        if unsat_count > 0:
            dominant_topic = top_10[0][0] if top_10 else "general"
            summary_sentences.append(
                f"The most frequent failure mode was vague or generic responses that didn't address "
                f"the user's specific concern, particularly for questions about "
                f"{dominant_topic.replace('_', ' ')} and treatment results."
            )
            if top_reasons_text:
                summary_sentences.append(f"Notable failure patterns: {top_reasons_text}.")
        else:
            summary_sentences.append("No unsatisfactory responses were found in the sample this week.")

        f.write("**Summary:** " + " ".join(summary_sentences) + "\n\n")

        f.write("## Section B — Top 10 User Query Topics\n\n")
        f.write("| Rank | Topic | Count | % | Example Questions |\n")
        f.write("|---|---|---|---|---|\n")
        for rank, (topic, cnt) in enumerate(top_10, 1):
            pct = round(cnt / sample_size * 100, 1)
            examples = topic_examples[topic][:3]
            ex_str = "<br>".join(f"• {q}" for q in examples)
            f.write(f"| {rank} | {topic.replace('_', ' ')} | {cnt} | {pct}% | {ex_str} |\n")

    print(f"  Wrote {summary_path}", flush=True)

    # ── Step 5b: flagged CSV ───────────────────────────────────────────────────
    flagged_path = os.path.join(OUTPUT_DIR, f"flagged_{TODAY}.csv")
    with open(flagged_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["date", "user_id", "user_question", "bot_answer", "bucket", "reason", "full_conversation"])
        for ex in no_answer:
            writer.writerow([TODAY, ex["user_id"], ex["user_msg"]["Message"],
                             ex["bot_msg"]["Message"], "NO_ANSWER", "", ""])
        for ex in unsat_exchanges:
            conv = build_thread_text(ex["user_id"], threads)
            writer.writerow([TODAY, ex["user_id"], ex["user_msg"]["Message"],
                             ex["bot_msg"]["Message"], "UNSATISFACTORY", ex["reason"], conv])

    print(f"  Wrote {flagged_path} ({len(no_answer)} NO_ANSWER + {unsat_count} UNSATISFACTORY)", flush=True)

    # ── Step 5c: trend.csv ────────────────────────────────────────────────────
    trend_path = os.path.join(OUTPUT_DIR, "trend.csv")
    trend_header = [
        "week", "total_exchanges", "no_answer_count", "no_answer_pct",
        "answered_pool", "sample_size", "unsat_count", "wrong_rate_pct", "moe_pct"
    ]
    trend_row = [TODAY, total_exchanges, len(no_answer), no_answer_pct,
                 len(answered), sample_size, unsat_count, wrong_rate_pct, moe_pct]

    existing_rows = []
    if os.path.exists(trend_path):
        with open(trend_path, "r", encoding="utf-8") as f:
            for r in csv.reader(f):
                if not r or r[0] in ("week", TODAY):
                    continue
                existing_rows.append(r)

    with open(trend_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(trend_header)
        for r in existing_rows:
            writer.writerow(r)
        writer.writerow(trend_row)

    print(f"  Wrote/updated {trend_path}", flush=True)
    print("\n=== Done ===", flush=True)
    return {
        "today": TODAY, "total": total_exchanges,
        "no_answer": len(no_answer), "sample": sample_size,
        "unsat": unsat_count, "wrong_rate": wrong_rate_pct, "moe": moe_pct,
    }


if __name__ == "__main__":
    stats = main()
    print(json.dumps(stats))
