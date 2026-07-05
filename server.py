#!/usr/bin/env python3
"""
Cureskin Bot Quality Dashboard — local web server
Usage: python3 server.py [--port 5000]
"""

import csv
import json
import os
import subprocess
import sys
import glob
from pathlib import Path
from flask import Flask, jsonify, send_file, request, abort

BASE = Path(__file__).parent
OUT  = BASE / "output"
DATA = BASE / "data"

app = Flask(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

def newest_csv():
    """Return the path of the most recently timestamped CSV in data/."""
    csvs = sorted(DATA.glob("unpaid_user_llm_bot_responses_*.csv"))
    return csvs[-1] if csvs else None


def load_flagged_csv(path: Path, has_bucket: bool = True, week: str = "") -> list:
    rows = []
    CONVO_LIMIT = 600
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            convo = row.get("full_conversation", "")
            if len(convo) > CONVO_LIMIT:
                convo = convo[:CONVO_LIMIT] + "…"
            bucket = row.get("bucket", "UNSATISFACTORY") if has_bucket else "UNSATISFACTORY"
            rows.append({
                "uid":    row.get("user_id", "")[-10:],
                "q":      row.get("user_question", "")[:200],
                "a":      row.get("bot_answer", "")[:300],
                "bucket": bucket,
                "reason": row.get("reason", "")[:200],
                "convo":  convo,
            })
    return rows


def load_audit_data() -> dict:
    """
    Load the structured audit snapshot (audit_data.json) and refresh
    the flagged exchanges live from the individual flagged_*.csv files.
    """
    snapshot_path = OUT / "audit_data.json"
    if not snapshot_path.exists():
        abort(503, "audit_data.json not found — run the audit first")

    with open(snapshot_path, encoding="utf-8") as f:
        data = json.load(f)

    # Refresh flagged data from the live CSV files so new audit runs
    # are picked up without restarting the server.
    flagged = {}
    for csv_path in sorted(OUT.glob("flagged_*.csv")):
        # extract week from filename: flagged_2026-06-29.csv → 2026-06-29
        week = csv_path.stem.replace("flagged_", "")
        has_bucket = week != "2026-06-02"
        flagged[week] = load_flagged_csv(csv_path, has_bucket, week)
    data["flagged"] = flagged

    # Refresh trend from trend.csv (so new rows are picked up)
    trend_path = OUT / "trend.csv"
    if trend_path.exists():
        week_labels = {
            "2026-06-02": "Jun 02", "2026-06-08": "Jun 08",
            "2026-06-22": "Jun 22", "2026-06-29": "Jun 29",
        }
        trend = []
        with open(trend_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                w = row["week"]
                def si(v): return int(float(v)) if v else None
                def sf(v): return float(v) if v else None
                total   = si(row["total_exchanges"])
                no_ans  = si(row["no_answer_count"])
                ans     = si(row["answered_pool"]) if row.get("answered_pool") else None
                timeout = (total - no_ans - ans) if (ans is not None and no_ans is not None) else None
                trend.append({
                    "week":         w,
                    "label":        week_labels.get(w, w[-5:]),
                    "total":        total,
                    "no_answer":    no_ans,
                    "no_answer_pct": sf(row["no_answer_pct"]),
                    "timeout":      timeout,
                    "answered":     ans,
                    "sample":       si(row["sample_size"]) if row.get("sample_size") else None,
                    "unsat":        si(row["unsat_count"])  if row.get("unsat_count")  else None,
                    "rate":         sf(row["wrong_rate_pct"]),
                    "moe":          sf(row["moe_pct"]),
                })
        # June-02 was written in old format — fix it
        for t in trend:
            if t["week"] == "2026-06-02":
                t.update({"total":2322,"no_answer":None,"no_answer_pct":None,
                           "timeout":None,"answered":None,"sample":1000,
                           "unsat":92,"rate":9.2,"moe":1.79})
        data["trend"] = trend

    return data


def find_audit_script(week: str) -> Path | None:
    """Find the most specific audit script for the given week."""
    date = week.replace("-", "")
    specific = BASE / f"run_audit_{date}.py"
    if specific.exists():
        return specific
    generic = BASE / "run_audit.py"
    return generic if generic.exists() else None


# ── routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_file(OUT / "dashboard_app.html")


@app.route("/api/data")
def api_data():
    data = load_audit_data()
    return jsonify(data)


@app.route("/api/weeks")
def api_weeks():
    """List all available weeks (from flagged CSVs + trend.csv)."""
    weeks = sorted(
        p.stem.replace("flagged_", "")
        for p in OUT.glob("flagged_*.csv")
    )
    newest = newest_csv()
    return jsonify({
        "weeks": weeks,
        "newest_source_csv": newest.name if newest else None,
    })


@app.route("/api/run-audit", methods=["POST"])
def run_audit():
    """
    Run the audit for the newest CSV. Streams subprocess output.
    Returns JSON with status, stdout, stderr.
    """
    payload  = request.get_json(silent=True) or {}
    csv_file = newest_csv()
    if not csv_file:
        return jsonify({"ok": False, "error": "No source CSV found in data/"}), 400

    # Guess the week date from filename
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", csv_file.name)
    week = m.group(1) if m else None

    script = find_audit_script(week) if week else None
    if not script:
        return jsonify({"ok": False, "error": "No audit script found. Create run_audit_YYYYMMDD.py"}), 400

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(BASE),
            capture_output=True,
            text=True,
            timeout=600,
        )
        return jsonify({
            "ok":     result.returncode == 0,
            "week":   week,
            "script": script.name,
            "stdout": result.stdout[-4000:],
            "stderr": result.stderr[-2000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Audit timed out (10 min)"}), 504
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/output/<path:filename>")
def serve_output(filename):
    """Serve files from output/ directly (e.g. CSVs for download)."""
    return send_file(OUT / filename)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n  Cureskin Bot Quality Dashboard")
    print(f"  → http://{args.host}:{args.port}/\n")
    app.run(host=args.host, port=args.port, debug=True)
