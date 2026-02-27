"""
Positional Bias Mitigation Experiment for Gemini 3 Flash

Tests 4 mitigation strategies against the baseline 68% second-position bias:
  1. BPC  (Balanced Position Calibration) - Dual swap evaluation
  2. MEC  (Multi-Evidence Calibration)    - Chain-of-Thought reasoning first
  3. CAI  (Constitutional AI)             - Anti-bias constitution in system prompt
  4. Combined (BPC + MEC + CAI)           - All three combined

Usage:
  python bias_mitigation_test.py --strategy bpc
  python bias_mitigation_test.py --strategy mec
  python bias_mitigation_test.py --strategy cai
  python bias_mitigation_test.py --strategy combined
"""

import os
import sys
import json
import time
import argparse
from typing import Literal
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=API_KEY)

# Model init
try:
    model = genai.GenerativeModel("gemini-3-flash-preview")
    MODEL_NAME = "gemini-3-flash-preview"
except Exception:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        MODEL_NAME = "gemini-2.0-flash-exp"
    except Exception:
        model = genai.GenerativeModel("gemini-1.5-flash")
        MODEL_NAME = "gemini-1.5-flash"

print(f"Using model: {MODEL_NAME}")

# ── Answer Definitions (same as baseline) ─────────────────────────────────────
ANSWER_A = """A process is a program in execution. It is the unit of work in a system. A process needs resources like CPU time, memory, files, and I/O devices to accomplish its task. Each process has its own address space, meaning if one process crashes, it doesn't affect others. It contains a code section, data section, and stack."""

ANSWER_B = """A process is an instance of a running program. It is managed by the OS using a Process Control Block (PCB), which stores the program counter, registers, and scheduling info. A process goes through states like New, Ready, Running, Waiting, and Terminated during its lifecycle."""


# ── Prompt Templates ──────────────────────────────────────────────────────────

# Baseline prompt (same structure as bias_test.py)
BASELINE_SYSTEM = "You are an expert technical evaluator specializing in Operating Systems."

BASELINE_USER_TEMPLATE = """Compare these two explanations of "What is a Process in an Operating System?":

Answer 1:
{answer1}

Answer 2:
{answer2}

Which explanation is better overall? Consider completeness, accuracy, and clarity.

Respond ONLY with valid JSON in this exact format:
{{
  "winner": "1" or "2",
  "reason": "brief explanation"
}}"""

# MEC prompt: forces reasoning BEFORE winner
MEC_USER_TEMPLATE = """Compare these two explanations of "What is a Process in an Operating System?":

Answer 1:
{answer1}

Answer 2:
{answer2}

IMPORTANT: Before deciding the winner, you MUST first write a detailed comparative analysis covering the strengths and weaknesses of EACH answer. Only AFTER completing your analysis should you determine which is better.

Consider: completeness, accuracy, clarity, and technical depth.

Respond ONLY with valid JSON in this exact format:
{{
  "analysis_answer_1": "strengths and weaknesses of Answer 1",
  "analysis_answer_2": "strengths and weaknesses of Answer 2",
  "reasoning": "comparative conclusion based on your analysis above",
  "winner": "1" or "2"
}}"""

# CAI system prompt: constitutional anti-bias rules
CAI_SYSTEM = """You are an expert technical evaluator specializing in Operating Systems.

=== EVALUATION CONSTITUTION (MANDATORY) ===
You MUST obey these constitutional principles when evaluating:

1. POSITION NEUTRALITY: The order in which answers are presented has ZERO relevance to their quality. You must NOT favor an answer simply because it appears first or second.
2. CONTENT-ONLY JUDGMENT: Judge ONLY based on factual accuracy, completeness, technical depth, and clarity of explanation. Nothing else.
3. LENGTH INDEPENDENCE: A longer answer is NOT automatically better. A shorter answer is NOT automatically worse. Judge by information density and correctness.
4. SELF-AUDIT: Before finalizing your decision, ask yourself: "Would my decision change if the two answers swapped positions?" If yes, re-evaluate.
===

Apply these constitutional principles strictly to every evaluation."""

# Combined prompt uses CAI system + MEC user template


# ── Core API Call ─────────────────────────────────────────────────────────────

def call_model(system_prompt: str, user_prompt: str) -> str:
    """Make a single API call and return cleaned text."""
    model_with_system = genai.GenerativeModel(
        MODEL_NAME,
        system_instruction=system_prompt
    )
    response = model_with_system.generate_content(user_prompt)
    return response.text.replace("```json", "").replace("```", "").strip()


def parse_winner(raw: str, first: str, second: str) -> tuple:
    """Parse JSON response and map positional winner to A/B."""
    try:
        data = json.loads(raw)
        winner_pos = str(data.get("winner", ""))
        if winner_pos == "1":
            return first, data.get("reason", data.get("reasoning", ""))
        elif winner_pos == "2":
            return second, data.get("reason", data.get("reasoning", ""))
        else:
            return "INVALID", str(data)
    except json.JSONDecodeError:
        return "PARSE_ERROR", raw[:200]


# ── Strategy Runners ──────────────────────────────────────────────────────────

def run_trial_bpc(order: Literal["A-B", "B-A"], trial_num: int) -> dict:
    """BPC: Run two calls with swapped order, cross-validate."""
    # Call 1: original order
    if order == "A-B":
        prompt1 = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        prompt2 = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        first1, second1 = "A", "B"
        first2, second2 = "B", "A"
    else:
        prompt1 = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        prompt2 = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        first1, second1 = "B", "A"
        first2, second2 = "A", "B"

    try:
        raw1 = call_model(BASELINE_SYSTEM, prompt1)
        time.sleep(1)
        raw2 = call_model(BASELINE_SYSTEM, prompt2)

        winner1, reason1 = parse_winner(raw1, first1, second1)
        winner2, reason2 = parse_winner(raw2, first2, second2)

        # Cross-validate: both must agree
        if winner1 == winner2 and winner1 in ("A", "B"):
            final_winner = winner1
        else:
            final_winner = "TIE"

        return {
            "trial": trial_num, "order": order, "strategy": "bpc",
            "call1_winner": winner1, "call2_winner": winner2,
            "winner": final_winner,
            "reason": f"Call1({winner1}): {reason1} | Call2({winner2}): {reason2}",
        }
    except Exception as e:
        return {
            "trial": trial_num, "order": order, "strategy": "bpc",
            "winner": "ERROR", "reason": str(e),
        }


def run_trial_mec(order: Literal["A-B", "B-A"], trial_num: int) -> dict:
    """MEC: Chain-of-Thought reasoning before winner."""
    if order == "A-B":
        prompt = MEC_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        first, second = "A", "B"
    else:
        prompt = MEC_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        first, second = "B", "A"

    try:
        raw = call_model(BASELINE_SYSTEM, prompt)
        winner, reason = parse_winner(raw, first, second)
        return {
            "trial": trial_num, "order": order, "strategy": "mec",
            "winner": winner, "reason": reason, "raw_response": raw,
        }
    except Exception as e:
        return {
            "trial": trial_num, "order": order, "strategy": "mec",
            "winner": "ERROR", "reason": str(e),
        }


def run_trial_cai(order: Literal["A-B", "B-A"], trial_num: int) -> dict:
    """CAI: Constitutional AI anti-bias system prompt."""
    if order == "A-B":
        prompt = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        first, second = "A", "B"
    else:
        prompt = BASELINE_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        first, second = "B", "A"

    try:
        raw = call_model(CAI_SYSTEM, prompt)
        winner, reason = parse_winner(raw, first, second)
        return {
            "trial": trial_num, "order": order, "strategy": "cai",
            "winner": winner, "reason": reason, "raw_response": raw,
        }
    except Exception as e:
        return {
            "trial": trial_num, "order": order, "strategy": "cai",
            "winner": "ERROR", "reason": str(e),
        }


def run_trial_combined(order: Literal["A-B", "B-A"], trial_num: int) -> dict:
    """Combined: CAI system + MEC prompt + BPC dual-call."""
    if order == "A-B":
        prompt1 = MEC_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        prompt2 = MEC_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        first1, second1 = "A", "B"
        first2, second2 = "B", "A"
    else:
        prompt1 = MEC_USER_TEMPLATE.format(answer1=ANSWER_B, answer2=ANSWER_A)
        prompt2 = MEC_USER_TEMPLATE.format(answer1=ANSWER_A, answer2=ANSWER_B)
        first1, second1 = "B", "A"
        first2, second2 = "A", "B"

    try:
        raw1 = call_model(CAI_SYSTEM, prompt1)
        time.sleep(1)
        raw2 = call_model(CAI_SYSTEM, prompt2)

        winner1, reason1 = parse_winner(raw1, first1, second1)
        winner2, reason2 = parse_winner(raw2, first2, second2)

        if winner1 == winner2 and winner1 in ("A", "B"):
            final_winner = winner1
        else:
            final_winner = "TIE"

        return {
            "trial": trial_num, "order": order, "strategy": "combined",
            "call1_winner": winner1, "call2_winner": winner2,
            "winner": final_winner,
            "reason": f"Call1({winner1}): {reason1} | Call2({winner2}): {reason2}",
        }
    except Exception as e:
        return {
            "trial": trial_num, "order": order, "strategy": "combined",
            "winner": "ERROR", "reason": str(e),
        }


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_results(results: list, strategy: str):
    """Print analysis comparing against baseline."""
    ab_trials = [r for r in results if r["order"] == "A-B"]
    ba_trials = [r for r in results if r["order"] == "B-A"]

    a_wins_first = sum(1 for r in ab_trials if r["winner"] == "A")
    a_wins_second = sum(1 for r in ba_trials if r["winner"] == "A")
    b_wins_first = sum(1 for r in ba_trials if r["winner"] == "B")
    b_wins_second = sum(1 for r in ab_trials if r["winner"] == "B")
    ties = sum(1 for r in results if r["winner"] == "TIE")
    errors = sum(1 for r in results if r["winner"] in ("ERROR", "INVALID", "PARSE_ERROR"))

    valid = len(results) - errors
    first_wins = sum(1 for r in results if r["winner"] == r.get("first", ("A" if r["order"] == "A-B" else "B")))
    # Compute positional wins more carefully
    first_pos_wins = 0
    second_pos_wins = 0
    for r in results:
        if r["winner"] in ("TIE", "ERROR", "INVALID", "PARSE_ERROR"):
            continue
        first_label = "A" if r["order"] == "A-B" else "B"
        if r["winner"] == first_label:
            first_pos_wins += 1
        else:
            second_pos_wins += 1

    valid_no_tie = first_pos_wins + second_pos_wins

    print(f"\n{'='*60}")
    print(f"📊 RESULTS: Strategy = {strategy.upper()}")
    print(f"{'='*60}")
    print(f"Model: {MODEL_NAME}")
    print(f"Total Trials: {len(results)}  |  Errors: {errors}  |  Ties: {ties}")
    print()
    print("Answer A Performance:")
    print(f"  Win when FIRST  (A-B order): {a_wins_first}/50 ({a_wins_first*2}%)")
    print(f"  Win when SECOND (B-A order): {a_wins_second}/50 ({a_wins_second*2}%)")
    print()
    print("Answer B Performance:")
    print(f"  Win when FIRST  (B-A order): {b_wins_first}/50 ({b_wins_first*2}%)")
    print(f"  Win when SECOND (A-B order): {b_wins_second}/50 ({b_wins_second*2}%)")
    print()

    if valid_no_tie > 0:
        fp = first_pos_wins / valid_no_tie * 100
        sp = second_pos_wins / valid_no_tie * 100
    else:
        fp = sp = 0

    print("Positional Bias Analysis (excluding Ties):")
    print(f"  First position wins:  {first_pos_wins}/{valid_no_tie} ({fp:.1f}%)")
    print(f"  Second position wins: {second_pos_wins}/{valid_no_tie} ({sp:.1f}%)")
    if ties > 0:
        print(f"  Ties (position-ambiguous): {ties}/{len(results)} ({ties/len(results)*100:.1f}%)")

    print()
    print("─── Comparison vs Baseline ───")
    print(f"  Baseline:  First 32% / Second 68%  (Δ = 36pp)")
    if valid_no_tie > 0:
        delta = abs(fp - sp)
        print(f"  {strategy.upper()}:  First {fp:.1f}% / Second {sp:.1f}%  (Δ = {delta:.1f}pp)")
        improvement = 36 - delta
        if improvement > 0:
            print(f"  ✅ Bias reduced by {improvement:.1f}pp")
        elif improvement == 0:
            print(f"  ⚠️  No change from baseline")
        else:
            print(f"  ❌ Bias increased by {-improvement:.1f}pp")
    else:
        print(f"  {strategy.upper()}: All Ties (no positional winner)")

    print(f"{'='*60}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

STRATEGY_MAP = {
    "bpc": run_trial_bpc,
    "mec": run_trial_mec,
    "cai": run_trial_cai,
    "combined": run_trial_combined,
}


def main():
    parser = argparse.ArgumentParser(description="Positional Bias Mitigation Test")
    parser.add_argument("--strategy", required=True, choices=STRATEGY_MAP.keys(),
                        help="Mitigation strategy to test")
    parser.add_argument("--trials", type=int, default=100,
                        help="Total trials (split 50/50 between A-B and B-A)")
    args = parser.parse_args()

    run_fn = STRATEGY_MAP[args.strategy]
    half = args.trials // 2
    results = []

    print(f"\nStarting {args.strategy.upper()} experiment: {args.trials} trials")
    print("=" * 60)

    # A-B order
    for i in range(half):
        idx = i + 1
        print(f"Trial {idx}/{args.trials} (A-B)...", end=" ", flush=True)
        result = run_fn("A-B", idx)
        results.append(result)
        print(f"Winner: {result['winner']}")
        time.sleep(1)

    # B-A order
    for i in range(half):
        idx = half + i + 1
        print(f"Trial {idx}/{args.trials} (B-A)...", end=" ", flush=True)
        result = run_fn("B-A", idx)
        results.append(result)
        print(f"Winner: {result['winner']}")
        time.sleep(1)

    # Save results
    outfile = f"results_{args.strategy}.json"
    with open(outfile, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {outfile}")

    # Analysis
    analyze_results(results, args.strategy)


if __name__ == "__main__":
    main()
