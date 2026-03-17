import anthropic
import json
import os
import time
import random
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

TEST_MODEL  = "claude-sonnet-4-6"   # model being studied
JUDGE_MODEL = "claude-haiku-4-5-20251001"  # cheaper model used to evaluate responses
SLEEP_SECS  = 13                    # >12s keeps us safely under 5 RPM on free tier
RUNS_PER_PROMPT = 3                 # how many times to call each prompt variant
PROMPTS_FILE = "data/prompts.json"
RESULTS_FILE = "data/results.json"

# ── Client ────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Helper: call the test model ───────────────────────────────────────────────

def get_response(prompt: str, model: str) -> str:
    """Call the test model and return the response text."""
    message = client.messages.create(
        model=model,
        max_tokens=512,
        temperature=1,   # fixed for reproducibility
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── Helper: call the judge model ─────────────────────────────────────────────

JUDGE_SYSTEM = """You are an objective evaluator of AI responses.
You will be given a response from an AI assistant and asked to rate it.
You have NOT seen the original prompt — evaluate only the quality of the response itself.
Respond with JSON only. No preamble, no markdown fences."""

JUDGE_PROMPT = """Rate the following AI response on four dimensions.
Each dimension should be scored from 1 (very poor) to 5 (excellent).

Dimensions:
- accuracy: Is the information correct and well-grounded?
- completeness: Does the response cover the topic adequately?
- clarity: Is the response clearly written and easy to understand?
- depth: Is the level of detail appropriate — neither too shallow nor unnecessarily verbose?

Respond with this exact JSON structure:
{{
  "accuracy": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "depth": <1-5>,
  "reasoning": "<one sentence explaining your overall assessment>"
}}

Response to evaluate:
{response}"""

def judge_response(response: str) -> dict:
    """Have the judge model score a response. Returns a dict of scores."""
    message = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        system=JUDGE_SYSTEM,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(response=response)
        }]
    )
    raw = message.content[0].text.strip()
    try:
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except json.JSONDecodeError:
        print(f"  ⚠️  Judge returned non-JSON: {raw[:80]}")
        return {"accuracy": None, "completeness": None,
                "clarity": None, "depth": None, "reasoning": raw}

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_study():
    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)

    # Load existing results so we can resume if interrupted
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        done_ids = {(r["prompt_id"], r["condition"], r["run"]) for r in results}
        print(f"Resuming — {len(results)} results already saved.")
    else:
        results = []
        done_ids = set()

    total_calls = len(prompts) * 2 * RUNS_PER_PROMPT * 2  # test + judge per variant per run
    print(f"Study plan: {len(prompts)} prompts × 2 conditions × {RUNS_PER_PROMPT} runs")
    print(f"Estimated API calls: {total_calls}")
    print(f"Estimated time: ~{total_calls * SLEEP_SECS // 60} minutes\n")

    for prompt in prompts:
        pid = prompt["id"]
        for condition in ["polite", "blunt"]:
            text = prompt[condition]
            for run in range(1, RUNS_PER_PROMPT + 1):
                key = (pid, condition, run)
                if key in done_ids:
                    print(f"  Skipping {pid} / {condition} / run {run} (already done)")
                    continue

                print(f"  [{pid}] {condition} — run {run}/{RUNS_PER_PROMPT}")

                # 1. Get test model response
                response_text = get_response(text, TEST_MODEL)
                time.sleep(SLEEP_SECS)

                # 2. Judge the response (blinded — judge never sees the prompt)
                scores = judge_response(response_text)
                time.sleep(SLEEP_SECS)

                # 3. Save result
                result = {
                    "prompt_id": pid,
                    "domain": prompt["domain"],
                    "condition": condition,
                    "run": run,
                    "response": response_text,
                    "scores": scores
                }
                results.append(result)
                done_ids.add(key)

                # Save after every result so progress isn't lost
                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)

    print(f"\n✅ Study complete. {len(results)} results saved to {RESULTS_FILE}")

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Quick sanity check before the full run
    print("Testing API connection...")
    test = get_response("Say 'API connection successful' and nothing else.", TEST_MODEL)
    print(f"  {test.strip()}\n")
    run_study()
