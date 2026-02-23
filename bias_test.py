"""
Positional Bias Experiment for Gemini 3 Flash

This script tests whether the order of presentation (A-B vs B-A) affects
the model's preference when comparing two similar technical explanations.

Topic: Operating System Process
Trials: 100 (50 A-first, 50 B-first)
"""

import os
import json
import time
from typing import Literal
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=API_KEY)

# Try Gemini 3 Flash first, fallback to available models
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

# Answer Definitions
ANSWER_A = """A process is a program in execution. It is the unit of work in a system. A process needs resources like CPU time, memory, files, and I/O devices to accomplish its task. Each process has its own address space, meaning if one process crashes, it doesn't affect others. It contains a code section, data section, and stack."""

ANSWER_B = """A process is an instance of a running program. It is managed by the OS using a Process Control Block (PCB), which stores the program counter, registers, and scheduling info. A process goes through states like New, Ready, Running, Waiting, and Terminated during its lifecycle."""

# Prompt Template
SYSTEM_PROMPT = "You are an expert technical evaluator specializing in Operating Systems."

def build_comparison_prompt(answer1: str, answer2: str) -> str:
    return f"""Compare these two explanations of "What is a Process in an Operating System?":

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

def run_trial(order: Literal["A-B", "B-A"], trial_num: int) -> dict:
    """Run a single comparison trial."""
    if order == "A-B":
        prompt = build_comparison_prompt(ANSWER_A, ANSWER_B)
        first, second = "A", "B"
    else:
        prompt = build_comparison_prompt(ANSWER_B, ANSWER_A)
        first, second = "B", "A"
    
    try:
        response = model.generate_content(prompt)
        cleaned = response.text.replace("```json", "").replace("```", "").strip()
        result = json.loads(cleaned)
        
        # Map winner back to A/B
        winner_position = result["winner"]
        if winner_position == "1":
            winner = first
        elif winner_position == "2":
            winner = second
        else:
            winner = "INVALID"
        
        return {
            "trial": trial_num,
            "order": order,
            "first": first,
            "second": second,
            "winner": winner,
            "reason": result.get("reason", ""),
            "raw_response": cleaned
        }
    except Exception as e:
        print(f"Trial {trial_num} failed: {e}")
        return {
            "trial": trial_num,
            "order": order,
            "first": first,
            "second": second,
            "winner": "ERROR",
            "reason": str(e),
            "raw_response": ""
        }

def main():
    results = []
    
    print("Starting experiment: 100 trials")
    print("=" * 60)
    
    # Run 50 A-B trials
    for i in range(50):
        print(f"Trial {i+1}/100 (A-B order)...", end=" ")
        result = run_trial("A-B", i+1)
        results.append(result)
        print(f"Winner: {result['winner']}")
        time.sleep(1)  # Rate limiting
    
    # Run 50 B-A trials
    for i in range(50):
        print(f"Trial {i+51}/100 (B-A order)...", end=" ")
        result = run_trial("B-A", i+51)
        results.append(result)
        print(f"Winner: {result['winner']}")
        time.sleep(1)  # Rate limiting
    
    # Save raw results
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("Experiment complete! Results saved to results.json")
    
    # Quick analysis
    analyze_results(results)

def analyze_results(results: list):
    """Perform quick statistical analysis."""
    a_first_trials = [r for r in results if r["order"] == "A-B"]
    b_first_trials = [r for r in results if r["order"] == "B-A"]
    
    # Count wins
    a_wins_when_first = sum(1 for r in a_first_trials if r["winner"] == "A")
    a_wins_when_second = sum(1 for r in b_first_trials if r["winner"] == "A")
    
    b_wins_when_first = sum(1 for r in b_first_trials if r["winner"] == "B")
    b_wins_when_second = sum(1 for r in a_first_trials if r["winner"] == "B")
    
    errors = sum(1 for r in results if r["winner"] in ["ERROR", "INVALID"])
    
    print("\n📊 ANALYSIS")
    print("-" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Total Trials: {len(results)}")
    print(f"Errors: {errors}")
    print()
    print("Answer A Performance:")
    print(f"  - Win rate when presented FIRST:  {a_wins_when_first}/50 ({a_wins_when_first*2}%)")
    print(f"  - Win rate when presented SECOND: {a_wins_when_second}/50 ({a_wins_when_second*2}%)")
    print()
    print("Answer B Performance:")
    print(f"  - Win rate when presented FIRST:  {b_wins_when_first}/50 ({b_wins_when_first*2}%)")
    print(f"  - Win rate when presented SECOND: {b_wins_when_second}/50 ({b_wins_when_second*2}%)")
    print()
    
    # Positional bias detection
    first_position_wins = sum(1 for r in results if r["winner"] == r["first"])
    second_position_wins = sum(1 for r in results if r["winner"] == r["second"])
    
    print("Positional Bias Analysis:")
    print(f"  - First position wins:  {first_position_wins}/{len(results)-errors} ({first_position_wins/(len(results)-errors)*100:.1f}%)")
    print(f"  - Second position wins: {second_position_wins}/{len(results)-errors} ({second_position_wins/(len(results)-errors)*100:.1f}%)")
    
    if abs(first_position_wins - second_position_wins) > 10:
        print("\n⚠️  POSITIONAL BIAS DETECTED!")
    else:
        print("\n✅ No significant positional bias detected.")

if __name__ == "__main__":
    main()
