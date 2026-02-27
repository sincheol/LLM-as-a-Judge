"""
PAIRS (Pairwise-preference Search) Positional Bias Calibration (Track 2)

Based on the PAIRS paper (https://arxiv.org/abs/2403.16950), this script
extracts logprobs from the LLM's token generation process and uses
mathematical calibration (Softmax + Cross-position Averaging) to debias
pairwise comparison results, with Information Entropy as a robustness metric.

Pipeline:
  Step 1: Controlled Prompt Design      → Force single-token output ("1" or "2")
  Step 2: Logprobs Data Extraction      → Extract raw log probabilities from API
  Step 3: Softmax Normalization         → Convert logprobs to [0, 1] probabilities
  Step 4: Cross-Validation (Permutations) → Dual-eval with swapped positions
  Step 5: Mathematical Calibration      → Average cross-position probabilities

Usage:
  python bias_pairs_test.py --trials 100
"""

import os
import sys
import json
import math
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

# ── Use the new google.genai SDK with Vertex AI (logprobs enabled) ────────────
from google import genai
from google.genai import types

GCP_PROJECT = "gen-lang-client-0733328373"
GCP_LOCATION = "global"

client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)

# Model selection: gemini-3-flash-preview (same model as Track 1 baseline)
MODEL_ID = "gemini-3-flash-preview"

# Enum schema: forces model to output exactly "1" or "2" (replaces max_output_tokens trick)
RESPONSE_SCHEMA = {"type": "STRING", "enum": ["1", "2"]}

print(f"🧠 Using model: {MODEL_ID} (Vertex AI)")
print(f"📊 SDK: google.genai (logprobs enabled via Vertex AI, enum schema)")
print(f"🌍 Project: {GCP_PROJECT}, Location: {GCP_LOCATION}")

# ── Answer Definitions (same as baseline experiment) ──────────────────────────
ANSWER_A = """A process is a program in execution. It is the unit of work in a system. A process needs resources like CPU time, memory, files, and I/O devices to accomplish its task. Each process has its own address space, meaning if one process crashes, it doesn't affect others. It contains a code section, data section, and stack."""

ANSWER_B = """A process is an instance of a running program. It is managed by the OS using a Process Control Block (PCB), which stores the program counter, registers, and scheduling info. A process goes through states like New, Ready, Running, Waiting, and Terminated during its lifecycle."""

# ══════════════════════════════════════════════════════════════════════════════
# Step 1: Controlled Prompt Design
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert technical evaluator. You will be given two answers to compare.
CRITICAL INSTRUCTION: You must respond with ONLY a single character: either "1" or "2".
Do NOT write any explanation, reasoning, or additional text.
Your entire response must be exactly one character."""

def build_pairs_prompt(text_first: str, text_second: str) -> str:
    """Build a minimal prompt that forces single-token output for logprob extraction."""
    return f"""Which of the following two explanations of "What is a Process in an Operating System?" is better?

Answer 1:
{text_first}

Answer 2:
{text_second}

Respond with only "1" or "2". Nothing else."""


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: Logprobs Data Extraction
# ══════════════════════════════════════════════════════════════════════════════
def call_with_logprobs(prompt: str) -> dict:
    """
    Call the Gemini API with logprobs enabled.
    Returns a dict with:
      - raw_text: the actual generated text
      - logprobs_data: list of {token, log_probability} for top candidates
      - chosen_token: the token the model actually chose
    """
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        temperature=0.0,
        response_mime_type="text/x.enum",  # Force enum output (official Google approach)
        response_schema=RESPONSE_SCHEMA,   # Constrain to exactly "1" or "2"
        response_logprobs=True,
        logprobs=5,  # Top-5 alternative tokens
    )

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config,
    )

    raw_text = response.text.strip() if response.text else ""

    # Parse logprobs from the response
    logprobs_data = []
    chosen_token_info = None

    if (response.candidates
        and response.candidates[0].logprobs_result):

        logprobs_result = response.candidates[0].logprobs_result

        # Get the chosen token (first token generated)
        if logprobs_result.chosen_candidates:
            chosen = logprobs_result.chosen_candidates[0]
            chosen_token_info = {
                "token": chosen.token,
                "log_probability": chosen.log_probability
            }

        # Get top alternative candidates
        if logprobs_result.top_candidates:
            top = logprobs_result.top_candidates[0]
            for candidate in top.candidates:
                logprobs_data.append({
                    "token": candidate.token,
                    "log_probability": candidate.log_probability
                })

    return {
        "raw_text": raw_text,
        "chosen_token": chosen_token_info,
        "top_candidates": logprobs_data,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3: Softmax Normalization
# ══════════════════════════════════════════════════════════════════════════════
FALLBACK_LOGPROB = -100.0  # For tokens not found in top-5

def extract_choice_probabilities(api_result: dict) -> dict:
    """
    From the logprobs top candidates, find the log probabilities for
    tokens "1" and "2", then apply softmax normalization.

    Returns:
      {
        "raw_winner": "1" or "2" (text output),
        "logprob_1": float,
        "logprob_2": float,
        "prob_1": float (softmax normalized, 0~1),
        "prob_2": float (softmax normalized, 0~1),
        "token_1_found": bool,
        "token_2_found": bool,
      }
    """
    logprob_1 = FALLBACK_LOGPROB
    logprob_2 = FALLBACK_LOGPROB
    token_1_found = False
    token_2_found = False

    # Search through top candidates for "1" and "2" tokens
    for candidate in api_result["top_candidates"]:
        token_stripped = candidate["token"].strip()
        if token_stripped == "1" and not token_1_found:
            logprob_1 = candidate["log_probability"]
            token_1_found = True
        elif token_stripped == "2" and not token_2_found:
            logprob_2 = candidate["log_probability"]
            token_2_found = True

    # Softmax normalization: P(i) = exp(L_i) / (exp(L_1) + exp(L_2))
    exp_1 = math.exp(logprob_1)
    exp_2 = math.exp(logprob_2)
    total = exp_1 + exp_2

    prob_1 = exp_1 / total
    prob_2 = exp_2 / total

    # Raw winner from text output
    raw_text = api_result["raw_text"].strip()
    if "1" in raw_text:
        raw_winner = "1"
    elif "2" in raw_text:
        raw_winner = "2"
    else:
        raw_winner = "INVALID"

    return {
        "raw_winner": raw_winner,
        "logprob_1": logprob_1,
        "logprob_2": logprob_2,
        "prob_1": prob_1,
        "prob_2": prob_2,
        "token_1_found": token_1_found,
        "token_2_found": token_2_found,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Step 3.5: Information Entropy (Uncertainty / Robustness Metric)
# ══════════════════════════════════════════════════════════════════════════════
def compute_entropy(p_a: float, p_b: float) -> float:
    """
    Compute the information entropy (uncertainty) of a binary preference distribution.

    Formula:
      U(y_i, y_j) = -P(y_i > y_j) * log(P(y_i > y_j)) - P(y_j > y_i) * log(P(y_j > y_i))

    Interpretation:
      - U ≈ 0.0  → Low Uncertainty (Robust):  모델 판단이 확고하여 위치 노이즈에 강건함.
      - U ≈ 0.693 (= ln(2)) → Max Uncertainty: 동전 던지기 수준 (P=0.5/0.5), 판단 불가.

    NOTE: Entropy is applied to PAIRS-calibrated probabilities (post cross-position
    averaging), NOT raw per-prompt probabilities. This ensures the entropy measures
    robustness of the position-independent judgment, not the raw biased output.

    Args:
        p_a: Calibrated probability P(A > B) after PAIRS averaging
        p_b: Calibrated probability P(B > A) after PAIRS averaging

    Returns:
        Entropy value U ∈ [0, ln(2)], where 0 = maximally robust, ln(2) = maximally uncertain.
    """
    # Clamp to avoid log(0) = -inf
    eps = 1e-15
    p_a = max(eps, min(1 - eps, p_a))
    p_b = max(eps, min(1 - eps, p_b))

    entropy = -(p_a * math.log(p_a) + p_b * math.log(p_b))
    return entropy


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 & 5: Cross-Validation (Permutation Swap) + Calibration
# ══════════════════════════════════════════════════════════════════════════════
def run_pairs_trial(trial_num: int) -> dict:
    """
    Execute a single PAIRS trial:
      - Prompt 1: [Answer A as 1st, Answer B as 2nd]
      - Prompt 2: [Answer B as 1st, Answer A as 2nd]
      - Calibrate by averaging P(A) across both prompts

    Variable Tracking (CRITICAL to avoid mapping bugs):
      Prompt 1 [A-B]: "1" = A, "2" = B  →  P_prompt1(A) = prob_1
      Prompt 2 [B-A]: "1" = B, "2" = A  →  P_prompt2(A) = prob_2
    """
    results = {}

    # ── Prompt 1: A-B order ──────────────────────────────────────────────────
    prompt_ab = build_pairs_prompt(ANSWER_A, ANSWER_B)
    api_result_ab = call_with_logprobs(prompt_ab)
    probs_ab = extract_choice_probabilities(api_result_ab)

    # In A-B order: position "1" = Answer A, position "2" = Answer B
    p_A_in_prompt1 = probs_ab["prob_1"]  # P(A) when A is at position 1
    p_B_in_prompt1 = probs_ab["prob_2"]  # P(B) when B is at position 2

    # Map raw text winner to actual answer
    if probs_ab["raw_winner"] == "1":
        raw_winner_ab = "A"
    elif probs_ab["raw_winner"] == "2":
        raw_winner_ab = "B"
    else:
        raw_winner_ab = "INVALID"

    time.sleep(1)  # Rate limiting between the two API calls

    # ── Prompt 2: B-A order ──────────────────────────────────────────────────
    prompt_ba = build_pairs_prompt(ANSWER_B, ANSWER_A)
    api_result_ba = call_with_logprobs(prompt_ba)
    probs_ba = extract_choice_probabilities(api_result_ba)

    # In B-A order: position "1" = Answer B, position "2" = Answer A
    p_A_in_prompt2 = probs_ba["prob_2"]  # P(A) when A is at position 2
    p_B_in_prompt2 = probs_ba["prob_1"]  # P(B) when B is at position 1

    # Map raw text winner to actual answer
    if probs_ba["raw_winner"] == "1":
        raw_winner_ba = "B"
    elif probs_ba["raw_winner"] == "2":
        raw_winner_ba = "A"
    else:
        raw_winner_ba = "INVALID"

    # ── Step 5: Mathematical Calibration (Averaging) ─────────────────────────
    # P_calibrated(A) = (P_prompt1(A) + P_prompt2(A)) / 2
    p_calibrated_A = (p_A_in_prompt1 + p_A_in_prompt2) / 2
    p_calibrated_B = (p_B_in_prompt1 + p_B_in_prompt2) / 2

    # ── Step 5.5: Entropy (Uncertainty) on calibrated probabilities ──────────
    entropy = compute_entropy(p_calibrated_A, p_calibrated_B)

    # Determine PAIRS-calibrated winner
    if p_calibrated_A > p_calibrated_B:
        pairs_winner = "A"
    elif p_calibrated_B > p_calibrated_A:
        pairs_winner = "B"
    else:
        pairs_winner = "TIE"

    return {
        "trial": trial_num,
        # ── Raw results (text-based, bias-exposed) ──
        "raw_winner_ab": raw_winner_ab,
        "raw_winner_ba": raw_winner_ba,
        # ── Logprob details ──
        "prompt1_ab": {
            "logprob_1": probs_ab["logprob_1"],
            "logprob_2": probs_ab["logprob_2"],
            "prob_1_A": round(p_A_in_prompt1, 6),
            "prob_2_B": round(p_B_in_prompt1, 6),
            "token_1_found": probs_ab["token_1_found"],
            "token_2_found": probs_ab["token_2_found"],
            "raw_text": api_result_ab["raw_text"],
            "top_candidates": api_result_ab["top_candidates"],
        },
        "prompt2_ba": {
            "logprob_1": probs_ba["logprob_1"],
            "logprob_2": probs_ba["logprob_2"],
            "prob_1_B": round(p_B_in_prompt2, 6),
            "prob_2_A": round(p_A_in_prompt2, 6),
            "raw_text": api_result_ba["raw_text"],
            "token_1_found": probs_ba["token_1_found"],
            "token_2_found": probs_ba["token_2_found"],
            "top_candidates": api_result_ba["top_candidates"],
        },
        # ── PAIRS Calibrated results ──
        "p_calibrated_A": round(p_calibrated_A, 6),
        "p_calibrated_B": round(p_calibrated_B, 6),
        "pairs_winner": pairs_winner,
        # ── Entropy (Uncertainty / Robustness) ──
        "entropy": round(entropy, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Analysis & Reporting
# ══════════════════════════════════════════════════════════════════════════════
def analyze_results(results: list):
    """Comprehensive analysis comparing Raw vs PAIRS outcomes."""
    total = len(results)
    print(f"\n{'='*70}")
    print(f"📊 PAIRS EXPERIMENT ANALYSIS  (Model: {MODEL_ID})")
    print(f"{'='*70}")
    print(f"Total Trials: {total} (API Calls: {total * 2})")

    # ── 1. Raw Text Analysis (Bias-Exposed) ──────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"🔴 [A군] Raw Baseline Analysis (텍스트 출력 기준 - 편향 노출)")
    print(f"{'─'*70}")

    # A-B order raw results: who won?
    raw_ab_first_wins = 0   # position 1 wins in A-B prompts
    raw_ab_second_wins = 0  # position 2 wins in A-B prompts
    raw_ba_first_wins = 0   # position 1 wins in B-A prompts
    raw_ba_second_wins = 0  # position 2 wins in B-A prompts

    for r in results:
        # A-B order: position 1 = A, position 2 = B
        if r["raw_winner_ab"] == "A":
            raw_ab_first_wins += 1
        elif r["raw_winner_ab"] == "B":
            raw_ab_second_wins += 1

        # B-A order: position 1 = B, position 2 = A
        if r["raw_winner_ba"] == "B":
            raw_ba_first_wins += 1
        elif r["raw_winner_ba"] == "A":
            raw_ba_second_wins += 1

    total_raw_calls = total * 2
    total_first_wins = raw_ab_first_wins + raw_ba_first_wins
    total_second_wins = raw_ab_second_wins + raw_ba_second_wins

    print(f"  [A-B 순서] 1번 위치(A) 승: {raw_ab_first_wins}/{total}, 2번 위치(B) 승: {raw_ab_second_wins}/{total}")
    print(f"  [B-A 순서] 1번 위치(B) 승: {raw_ba_first_wins}/{total}, 2번 위치(A) 승: {raw_ba_second_wins}/{total}")
    print(f"  ──────────────────────────────────────────────────")
    valid_raw = total_first_wins + total_second_wins
    if valid_raw > 0:
        print(f"  📌 전체 위치 편향: 1번 위치 승 {total_first_wins}/{valid_raw} ({total_first_wins/valid_raw*100:.1f}%)")
        print(f"                    2번 위치 승 {total_second_wins}/{valid_raw} ({total_second_wins/valid_raw*100:.1f}%)")

    # Raw A vs B wins across all calls
    raw_a_total = raw_ab_first_wins + raw_ba_second_wins
    raw_b_total = raw_ab_second_wins + raw_ba_first_wins
    print(f"\n  📌 답변별 Raw 승률: A = {raw_a_total}/{valid_raw} ({raw_a_total/valid_raw*100:.1f}%), B = {raw_b_total}/{valid_raw} ({raw_b_total/valid_raw*100:.1f}%)")

    # ── 2. PAIRS Calibrated Analysis ─────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"🟢 [B군] PAIRS Calibrated Analysis (Logprob 보정 기준)")
    print(f"{'─'*70}")

    pairs_a_wins = sum(1 for r in results if r["pairs_winner"] == "A")
    pairs_b_wins = sum(1 for r in results if r["pairs_winner"] == "B")
    pairs_ties = sum(1 for r in results if r["pairs_winner"] == "TIE")

    print(f"  PAIRS 보정 결과: A 승 = {pairs_a_wins}, B 승 = {pairs_b_wins}, TIE = {pairs_ties}")

    valid_pairs = pairs_a_wins + pairs_b_wins
    if valid_pairs > 0:
        print(f"  📌 보정 후 승률: A = {pairs_a_wins}/{valid_pairs} ({pairs_a_wins/valid_pairs*100:.1f}%), B = {pairs_b_wins}/{valid_pairs} ({pairs_b_wins/valid_pairs*100:.1f}%)")

    # ── 3. Probability Statistics ────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"📈 Probability Statistics")
    print(f"{'─'*70}")

    avg_p_cal_A = sum(r["p_calibrated_A"] for r in results) / total
    avg_p_cal_B = sum(r["p_calibrated_B"] for r in results) / total

    print(f"  평균 P_calibrated(A) = {avg_p_cal_A:.6f}")
    print(f"  평균 P_calibrated(B) = {avg_p_cal_B:.6f}")

    # Token detection statistics
    ab_tok1_found = sum(1 for r in results if r["prompt1_ab"]["token_1_found"])
    ab_tok2_found = sum(1 for r in results if r["prompt1_ab"]["token_2_found"])
    ba_tok1_found = sum(1 for r in results if r["prompt2_ba"]["token_1_found"])
    ba_tok2_found = sum(1 for r in results if r["prompt2_ba"]["token_2_found"])

    print(f"\n  토큰 탐지율 (Top-5 내 발견 비율):")
    print(f"    [A-B] '1' 토큰: {ab_tok1_found}/{total}, '2' 토큰: {ab_tok2_found}/{total}")
    print(f"    [B-A] '1' 토큰: {ba_tok1_found}/{total}, '2' 토큰: {ba_tok2_found}/{total}")

    # ── 3.5. Entropy (Uncertainty / Robustness) Statistics ───────────────────
    print(f"\n{'─'*70}")
    print(f"🔒 Entropy (Uncertainty / Robustness) Analysis")
    print(f"{'─'*70}")

    max_entropy = math.log(2)  # ln(2) ≈ 0.693 = theoretical max for binary
    entropies = [r["entropy"] for r in results]
    avg_entropy = sum(entropies) / len(entropies)
    min_entropy = min(entropies)
    max_entropy_obs = max(entropies)

    # Classify trials by robustness
    robust_count = sum(1 for e in entropies if e < 0.1)    # Very robust
    moderate_count = sum(1 for e in entropies if 0.1 <= e < 0.5)
    fragile_count = sum(1 for e in entropies if e >= 0.5)  # Near coin-flip

    print(f"  U(y_i, y_j) = -P(A>B)·log(P(A>B)) - P(B>A)·log(P(B>A))")
    print(f"  이론적 최대 엔트로피 (동전 던지기): ln(2) ≈ {max_entropy:.4f}")
    print(f"")
    print(f"  📊 엔트로피 통계:")
    print(f"    평균 U = {avg_entropy:.6f}  (0에 가까울수록 판단이 확고함)")
    print(f"    최소 U = {min_entropy:.6f}")
    print(f"    최대 U = {max_entropy_obs:.6f}")
    print(f"")
    print(f"  📊 Robustness 분류:")
    print(f"    🟢 Low Uncertainty  (U < 0.1, Robust):   {robust_count}/{total} ({robust_count/total*100:.1f}%)")
    print(f"    🟡 Mid Uncertainty  (0.1 ≤ U < 0.5):     {moderate_count}/{total} ({moderate_count/total*100:.1f}%)")
    print(f"    🔴 High Uncertainty (U ≥ 0.5, Fragile):  {fragile_count}/{total} ({fragile_count/total*100:.1f}%)")
    print(f"")
    if avg_entropy < 0.1:
        print(f"  ✅ 결론: 평균 엔트로피 {avg_entropy:.4f} → 모델의 판단이 극도로 확고(Robust)합니다.")
        print(f"     프롬프트 위치(순서)를 바꾸는 등의 얕은 외부 노이즈에 흔들리지 않는 '단단한 결과'입니다.")
    elif avg_entropy < 0.5:
        print(f"  ⚠️  결론: 평균 엔트로피 {avg_entropy:.4f} → 중간 수준의 확신도. 일부 Trial에서 위치 노이즈에 취약할 수 있습니다.")
    else:
        print(f"  🔴 결론: 평균 엔트로피 {avg_entropy:.4f} → 동전 던지기 수준. 모델의 판단을 신뢰하기 어렵습니다.")

    # ── 4. Verdict Comparison ────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"⚖️  Raw vs PAIRS 판정 불일치 분석")
    print(f"{'─'*70}")

    # For each trial, compare the "Raw majority" vs "PAIRS calibrated"
    flips = 0
    for r in results:
        # Raw combined: if both agree on A or B
        raw_votes = [r["raw_winner_ab"], r["raw_winner_ba"]]
        raw_a_count = raw_votes.count("A")
        raw_b_count = raw_votes.count("B")

        if raw_a_count > raw_b_count:
            raw_combined = "A"
        elif raw_b_count > raw_a_count:
            raw_combined = "B"
        else:
            raw_combined = "TIE"

        if raw_combined != r["pairs_winner"] and r["pairs_winner"] != "TIE":
            flips += 1

    print(f"  판정 역전(Flip) 횟수: {flips}/{total} ({flips/total*100:.1f}%)")
    print(f"  → Raw 텍스트에서는 A 또는 B가 이겼지만, PAIRS 로짓 보정 후 승자가 바뀐 경우")

    print(f"\n{'='*70}")
    if valid_raw > 0 and abs(total_first_wins/valid_raw - 0.5) > abs(pairs_a_wins/(valid_pairs if valid_pairs > 0 else 1) - 0.5):
        print(f"✅ PAIRS가 위치 편향을 성공적으로 완화했습니다!")
    else:
        print(f"⚠️  PAIRS 보정 결과를 추가 분석이 필요합니다.")
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="PAIRS Positional Bias Calibration Experiment")
    parser.add_argument("--trials", type=int, default=100, help="Number of paired trials (each = 2 API calls)")
    args = parser.parse_args()

    num_trials = args.trials
    print(f"\n🚀 Starting PAIRS experiment: {num_trials} trials ({num_trials * 2} API calls)")
    print(f"{'='*70}")

    results = []
    for i in range(num_trials):
        print(f"  Trial {i+1}/{num_trials}...", end=" ", flush=True)
        try:
            result = run_pairs_trial(i + 1)
            results.append(result)
            print(f"Raw[A-B]={result['raw_winner_ab']}, Raw[B-A]={result['raw_winner_ba']} "
                  f"→ PAIRS={result['pairs_winner']} "
                  f"(P_A={result['p_calibrated_A']:.4f}, P_B={result['p_calibrated_B']:.4f}, "
                  f"U={result['entropy']:.4f})")
        except Exception as e:
            print(f"❌ Error: {e}")
            results.append({
                "trial": i + 1,
                "error": str(e),
                "raw_winner_ab": "ERROR",
                "raw_winner_ba": "ERROR",
                "pairs_winner": "ERROR",
                "p_calibrated_A": 0.0,
                "p_calibrated_B": 0.0,
                "prompt1_ab": {"token_1_found": False, "token_2_found": False},
                "prompt2_ba": {"token_1_found": False, "token_2_found": False},
            })

        time.sleep(1)  # Rate limiting between trials

    # Save raw results
    output_file = "results_pairs.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 Raw results saved to {output_file}")

    # Run analysis
    valid_results = [r for r in results if "error" not in r]
    if valid_results:
        analyze_results(valid_results)
    else:
        print("❌ No valid results to analyze.")


if __name__ == "__main__":
    main()
