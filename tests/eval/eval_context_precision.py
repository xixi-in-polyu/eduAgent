"""Compute LLMContextPrecisionWithReference for naive / mix / agentic answers.

Runs independently from the main eval pipeline — reads existing answers files
and judge_state to determine which questions were answered, then evaluates only
context_precision (no faithfulness / answer_relevancy re-run).

Saves results incrementally to:
  tests/eval/results/ragas_{prefix}_cp_scores.csv

Usage:
  python -m tests.eval.eval_context_precision
  python -m tests.eval.eval_context_precision --prefix naive
  python -m tests.eval.eval_context_precision --prefix naive mix agentic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.eval._common import (
    DATA_DIR,
    RESULTS_DIR,
    _bootstrap,
    load_json,
    non_thinking_extra_body,
)

_bootstrap()

PREFIXES = ["naive", "mix", "agentic"]
ANSWERS_FILES = {
    "naive": RESULTS_DIR / "answers_naive.json",
    "mix": RESULTS_DIR / "answers_mix.json",
    "agentic": RESULTS_DIR / "answers_agentic.json",
}
QUESTIONS_PATH = DATA_DIR / "ragas_custom_questions.json"
METRIC_COL = "llm_context_precision_with_reference"


# ---------------------------------------------------------------------------
# Helpers (copied from eval_ragas_custom.py)
# ---------------------------------------------------------------------------

def _build_ragas_llm():
    from rag_mvp.config import settings
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    extra_body = non_thinking_extra_body(
        settings.effective_chat_base_url,
        settings.effective_chat_model,
    )
    llm = ChatOpenAI(
        model=settings.effective_chat_model,
        api_key=settings.effective_chat_api_key or "placeholder",
        base_url=settings.effective_chat_base_url,
        temperature=0.0,
        extra_body=extra_body,
        max_retries=8,
    )
    return LangchainLLMWrapper(llm)


def _parse_context_field(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(c) for c in raw if str(c).strip()]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed if str(c).strip()]
            except Exception:
                pass
        if stripped:
            return [stripped]
    return []


def _load_answered_ids(prefix: str) -> set[str]:
    """Return question IDs that were judged as answered (judge_state True) or are in scores.csv."""
    import csv

    answered: set[str] = set()

    judge_path = RESULTS_DIR / f"ragas_{prefix}_judge_state.json"
    if judge_path.exists():
        state = json.loads(judge_path.read_text(encoding="utf-8"))
        answered.update(qid for qid, v in state.items() if v)

    scores_path = RESULTS_DIR / f"ragas_{prefix}_scores.csv"
    if scores_path.exists():
        with open(scores_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("is_answered", "").lower() in ("true", "1"):
                    answered.add(str(row["question_id"]))

    return answered


def _build_dataset(questions: list[dict], answers: list[dict], keep_ids: set[str]):
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    ans_map = {str(a.get("id", "")): a for a in answers}
    samples = []
    ordered_ids = []

    for q in questions:
        qid = str(q.get("id", ""))
        if qid not in keep_ids:
            continue
        question_text = str(q.get("question", ""))
        gold_answer = str(q.get("gold_answer", ""))
        a = ans_map.get(qid, {})
        generated_answer = str(a.get("generated_answer", ""))

        ans_contexts = a.get("retrieved_contexts")
        if isinstance(ans_contexts, list) and any(str(c).strip() for c in ans_contexts):
            contexts = [str(c) for c in ans_contexts if str(c).strip()]
        else:
            contexts = _parse_context_field(q.get("context", []))
        if not contexts:
            contexts = [gold_answer] if gold_answer else ["(no context)"]

        samples.append(SingleTurnSample(
            user_input=question_text,
            response=generated_answer,
            retrieved_contexts=contexts,
            reference=gold_answer,
        ))
        ordered_ids.append(qid)

    return EvaluationDataset(samples=samples), ordered_ids


def _run_prefix(prefix: str, questions: list[dict]) -> None:
    import pandas as pd
    from ragas import evaluate
    from ragas.metrics import LLMContextPrecisionWithReference  # noqa: deprecated but still functional
    from ragas.run_config import RunConfig

    answers_path = ANSWERS_FILES.get(prefix)
    if not answers_path or not answers_path.exists():
        print(f"[skip] No answers file for {prefix}: {answers_path}")
        return

    answers = load_json(str(answers_path))
    out_csv = RESULTS_DIR / f"ragas_{prefix}_cp_scores.csv"

    # Load already-computed rows
    done_ids: set[str] = set()
    existing_df = None
    if out_csv.exists():
        df = pd.read_csv(out_csv)
        ok = df[METRIC_COL].notna() if METRIC_COL in df.columns else pd.Series([], dtype=bool)
        if ok.any():
            existing_df = df[ok].reset_index(drop=True)
            done_ids = set(df.loc[ok, "question_id"].astype(str))
        print(f"  [incremental] {len(done_ids)} already done for {prefix}")

    answered_ids = _load_answered_ids(prefix)
    pending_ids = answered_ids - done_ids
    pending_questions = [q for q in questions if str(q.get("id", "")) in pending_ids]

    print(f"  answered={len(answered_ids)}  done={len(done_ids)}  pending={len(pending_questions)}")

    new_df = None
    if pending_questions:
        dataset, ordered_ids = _build_dataset(questions, answers, pending_ids)
        if not dataset.samples:
            print(f"  [warn] No samples built for {prefix}")
            return

        llm = _build_ragas_llm()
        print(f"  [ragas] Evaluating context_precision for {len(dataset.samples)} questions...")
        result = evaluate(
            dataset=dataset,
            metrics=[LLMContextPrecisionWithReference(llm=llm)],
            run_config=RunConfig(timeout=300, max_retries=15, max_workers=2, max_wait=60),
        )
        new_df = result.to_pandas()
        new_df.insert(0, "question_id", ordered_ids)
        # Rename RAGAS output column to our standard name
        cp_col = [c for c in new_df.columns if "context_precision" in c.lower() and c != "question_id"]
        if cp_col and cp_col[0] != METRIC_COL:
            new_df = new_df.rename(columns={cp_col[0]: METRIC_COL})
        elif not cp_col:
            print(f"  [warn] context_precision column not found in result: {list(new_df.columns)}")

    frames = [df for df in (existing_df, new_df) if df is not None]
    if not frames:
        print(f"  [warn] No data for {prefix}")
        return

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if new_df is not None:
        combined.to_csv(out_csv, index=False)
        print(f"  Saved: {out_csv}")

    if METRIC_COL in combined.columns:
        mean_cp = combined[METRIC_COL].dropna().mean()
        print(f"  {prefix} context_precision (mean, n={len(combined[METRIC_COL].dropna())}): {mean_cp:.4f}")


def _print_comparison() -> None:
    import pandas as pd

    print("\n=== Context Precision Comparison ===")
    results = {}
    for prefix in PREFIXES:
        csv_path = RESULTS_DIR / f"ragas_{prefix}_cp_scores.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if METRIC_COL not in df.columns:
            continue
        vals = df[METRIC_COL].dropna()
        results[prefix] = {"n": len(vals), "mean": round(vals.mean(), 4)}

    header = f"  {'mode':<12} {'n':>5}  {'context_precision':>18}"
    print(header)
    print("  " + "-" * 40)
    for mode, r in results.items():
        print(f"  {mode:<12} {r['n']:>5}  {r['mean']:>18.4f}")

    # Also show breakdown by question type if questions data available
    if QUESTIONS_PATH.exists():
        questions = load_json(str(QUESTIONS_PATH))
        _LABELS = {
            "single_hop_specific_query_synthesizer": "single_hop",
            "multi_hop_specific_query_synthesizer": "multi_hop_specific",
            "multi_hop_abstract_query_synthesizer": "multi_hop_abstract",
        }
        qtype = {str(q["id"]): _LABELS.get(q.get("evolution_type", ""), "unknown") for q in questions}

        print("\n  [by question type]")
        for qt in ["single_hop", "multi_hop_specific", "multi_hop_abstract"]:
            print(f"\n  {qt}")
            for prefix in PREFIXES:
                csv_path = RESULTS_DIR / f"ragas_{prefix}_cp_scores.csv"
                if not csv_path.exists():
                    continue
                df = pd.read_csv(csv_path)
                if METRIC_COL not in df.columns:
                    continue
                subset = df[df["question_id"].astype(str).map(qtype) == qt][METRIC_COL].dropna()
                if len(subset):
                    print(f"    {prefix:<12} n={len(subset):3d}  {subset.mean():.4f}")


def main(prefixes: list[str]) -> None:
    questions = load_json(str(QUESTIONS_PATH))
    print(f"Questions: {len(questions)}\n")

    for prefix in prefixes:
        print(f"\n=== {prefix} ===")
        _run_prefix(prefix, questions)

    _print_comparison()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        nargs="+",
        default=PREFIXES,
        choices=PREFIXES,
        help="Which modes to evaluate (default: all three)",
    )
    args = parser.parse_args()
    main(args.prefix)
