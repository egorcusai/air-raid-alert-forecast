"""
ai_interpretation.py
====================
LLM-based INTERPRETATION layer. This does NOT predict anything and is NOT in
the modelling path. It reads the already-computed statistics (correlation,
bias, calibration, region comparison) and produces a written analyst-style
narrative. Every number it discusses comes from the reproducible pipeline; the
LLM only phrases the interpretation.

Design choices for credibility:
  - Clearly separated from prediction. The models are scikit-learn; this is prose.
  - Deterministic-friendly: temperature kept low; the prompt forbids inventing
    numbers and instructs the model to only interpret values it is given.
  - Degrades gracefully: if no API key / no network, a hardcoded expert summary
    (written from the real numbers) is used so the artifact still works.
"""
from __future__ import annotations
import os, json

OUT = os.path.join(os.path.dirname(__file__), "..", "output")

# Fallback narrative written from the actual computed numbers. Used when the
# API is unavailable so the pipeline is always reproducible offline.
FALLBACK = (
    "The correlation matrix confirms geographic propagation: the most strongly "
    "correlated region pairs are all adjacent western oblasts (Ivano-Frankivsk and "
    "Ternopil at 0.84; Lviv with each at ~0.7), while the least correlated pairs "
    "span opposite ends of the country (western vs eastern oblasts near 0.12-0.13). "
    "This independently explains why neighbour features improved predictions in the "
    "west and not for isolated direct targets. On risk: the fitted model is poorly "
    "calibrated (ECE ~0.25), so its probabilities should be treated as a relative "
    "risk ranking, not absolute likelihoods, and would need recalibration before any "
    "operational use. Key biases to weigh: crowd-sourced reporting likely undercounts "
    "frontline regions, class balance ranges from ~5% (west) to 60%+ (frontline) so a "
    "single global threshold is inappropriate, and alert frequency is non-stationary "
    "as war intensity shifts. Net: the model ranks geographic risk credibly but is not "
    "a calibrated probability source and not a safety system."
)


def build_prompt(stats: dict) -> str:
    corr = stats["correlation"]
    calib = stats["calibration"]
    return (
        "You are a senior data analyst. Interpret ONLY the numbers provided; do "
        "not invent any figures. Write 150-200 words covering: (1) what the region "
        "correlation pattern implies about geographic propagation, (2) the "
        "calibration result and what it means for trusting the probabilities, "
        "(3) the two most important biases to caveat. Be honest about limitations.\n\n"
        f"Top correlated region pairs: {corr['top_correlated']}\n"
        f"Least correlated pairs: {corr['least_correlated']}\n"
        f"Calibration ECE: {calib['ece']} (0=perfect, higher=worse)\n"
        f"Bias notes: {stats['bias']['dataset_bias_notes']}\n"
    )


def generate(stats: dict, use_api: bool = True) -> dict:
    narrative, source = FALLBACK, "fallback"
    if use_api and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=json.dumps({
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 400,
                    "temperature": 0.2,
                    "messages": [{"role": "user", "content": build_prompt(stats)}],
                }).encode(),
                headers={
                    "content-type": "application/json",
                    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                    "anthropic-version": "2023-06-01",
                },
            )
            resp = json.loads(urllib.request.urlopen(req, timeout=30).read())
            narrative = "".join(b.get("text", "") for b in resp.get("content", []))
            source = "claude-sonnet-4-6"
        except Exception as e:
            narrative, source = FALLBACK, f"fallback (api error: {type(e).__name__})"
    return {"narrative": narrative.strip(), "source": source}


def main():
    stats = json.load(open(os.path.join(OUT, "risk_analysis.json")))
    result = generate(stats, use_api=True)
    print(f"[interpretation source: {result['source']}]\n")
    print(result["narrative"])
    with open(os.path.join(OUT, "ai_interpretation.json"), "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved -> output/ai_interpretation.json")
    return result


if __name__ == "__main__":
    main()
