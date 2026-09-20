"""Evaluate Ragas-custom (Chinese course material) answers with Ragas metrics.

Loads the synthetic QA pairs + TS agent answers, computes:
  answer_relevancy, faithfulness, context_precision, context_recall.

The context for each question is taken from the ``context`` field generated
by the Ragas TestsetGenerator during prepare_ragas_custom.py.

Usage:
  python -m tests.eval.eval_ragas_custom \
    --questions tests/eval/data/ragas_custom_questions.json \
    --answers   tests/eval/results/ragas_custom_answers.json
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tests.eval._common import (
    RESULTS_DIR,
    _bootstrap,
    load_json,
    non_thinking_extra_body,
    save_json,
)

_bootstrap()


def _build_ragas_llm():
    from rag_mvp.config import settings  # type: ignore[import-untyped]
    from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]
    from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]

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


def _build_ragas_embeddings():
    from rag_mvp.config import settings  # type: ignore[import-untyped]
    from langchain_openai import OpenAIEmbeddings  # type: ignore[import-untyped]
    from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]

    emb = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.embedding_api_key or settings.llm_api_key or "placeholder",
        base_url=settings.embedding_base_url or settings.llm_base_url,
    )
    return LangchainEmbeddingsWrapper(emb)


def _parse_context_field(raw) -> list[str]:
    """The context field from Ragas generator can be a string, list, or JSON."""
    import json as _json
    if isinstance(raw, list):
        return [str(c) for c in raw if str(c).strip()]
    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("["):
            try:
                parsed = _json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(c) for c in parsed if str(c).strip()]
            except Exception:
                pass
        if stripped:
            return [stripped]
    return []


def _load_judge_state(state_path: Path) -> dict[str, bool]:
    """Load persisted judge results {qid: True/False} from a JSON file."""
    if not state_path.exists():
        return {}
    import json as _json
    return _json.loads(state_path.read_text(encoding="utf-8"))


def _save_judge_state(state_path: Path, state: dict[str, bool]) -> None:
    import json as _json
    state_path.write_text(_json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


# Phrases that indicate the model declined to answer (rule-based answered detection).
# All patterns are matched case-insensitively against the generated answer.
_IDK_PATTERNS = [
    "i don't know",
    "i do not know",
    "i dont know",
    "未能找到",
]


def _rule_answered_batch(
    questions: list[dict],
    answers: list[dict],
) -> dict[str, bool]:
    """Rule-based check: answer is considered unanswered iff it is empty or contains an
    'I don't know' phrase (case-insensitive). No LLM call needed.

    Returns a dict mapping question id -> bool (True = answered, False = unanswered).
    """
    ans_map = {str(a.get("id", "")): a for a in answers}
    results: dict[str, bool] = {}

    for q in questions:
        qid = str(q.get("id", ""))
        answer_text = str(ans_map.get(qid, {}).get("generated_answer", "")).strip()
        if not answer_text:
            results[qid] = False
            continue
        lower = answer_text.lower()
        results[qid] = not any(p in lower for p in _IDK_PATTERNS)

    return results


def _build_ragas_dataset(questions: list[dict], answers: list[dict]):
    from ragas import EvaluationDataset  # type: ignore[import-untyped]
    from ragas.dataset_schema import SingleTurnSample  # type: ignore[import-untyped]

    ans_map = {str(a.get("id", "")): a for a in answers}

    samples = []
    for q in questions:
        qid = str(q.get("id", ""))
        question_text = str(q.get("question", ""))
        gold_answer = str(q.get("gold_answer", ""))

        a = ans_map.get(qid, {})
        generated_answer = str(a.get("generated_answer", ""))

        # Prefer actual retrieved contexts from the answers file (per-mode retrieval);
        # fall back to the Ragas-generated reference context from the questions file.
        ans_contexts = a.get("retrieved_contexts")
        if isinstance(ans_contexts, list) and any(str(c).strip() for c in ans_contexts):
            contexts = [str(c) for c in ans_contexts if str(c).strip()]
        else:
            contexts = _parse_context_field(q.get("context", []))

        if not contexts:
            contexts = [gold_answer] if gold_answer else ["(no context)"]

        samples.append(
            SingleTurnSample(
                user_input=question_text,
                response=generated_answer,
                retrieved_contexts=contexts,
                reference=gold_answer,
            )
        )

    return EvaluationDataset(samples=samples)


def _score_answer_correctness_batch(
    questions: list[dict],
    answers: list[dict],
) -> dict[str, float]:
    """LLM-as-a-judge: score each answer's correctness against the reference (0.0-1.0).

    Uses the same judge LLM as ``_judge_answered_batch`` (settings.llm_model).
    Prompt elicits one of five discrete levels: 1.0 / 0.75 / 0.5 / 0.25 / 0.0.
    """
    from rag_mvp.config import settings  # type: ignore[import-untyped]
    from langchain_openai import ChatOpenAI  # type: ignore[import-untyped]
    from langchain_core.messages import HumanMessage

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

    ans_map = {str(a.get("id", "")): a for a in answers}
    results: dict[str, float] = {}

    for q in questions:
        qid = str(q.get("id", ""))
        question_text = str(q.get("question", ""))
        reference = str(q.get("gold_answer", ""))
        answer_text = str(ans_map.get(qid, {}).get("generated_answer", "")).strip()

        prompt = (
            "你是一名客观的评分员。请根据参考答案，对以下生成答案的正确性打分。\n"
            "评分等级（只输出其中一个数字，不要包含其他任何内容）：\n"
            "  1.0 - 完全正确，与参考答案一致\n"
            "  0.75 - 基本正确，有少量遗漏或轻微表述差异\n"
            "  0.5 - 部分正确，核心内容有误或重要信息缺失\n"
            "  0.25 - 大部分不正确，仅包含少量正确内容\n"
            "  0.0 - 完全错误或与参考答案无关\n\n"
            f"问题：{question_text}\n\n"
            f"参考答案：{reference[:800]}\n\n"
            f"生成答案：{answer_text[:800]}\n\n"
            "评分："
        )
        try:
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = str(resp.content).strip().split()[0].rstrip("。，.,")
            score = float(raw)
            score = max(0.0, min(1.0, score))
        except Exception:
            score = float("nan")
        results[qid] = score

    return results


def _load_blacklist(blacklist_path: str | None) -> set[str]:
    """Load bad question IDs from a JSON file. Returns empty set if file missing or None."""
    import json as _json
    if not blacklist_path:
        return set()
    p = Path(blacklist_path)
    if not p.exists():
        return set()
    ids = _json.loads(p.read_text(encoding="utf-8"))
    return {str(i) for i in ids}


def main(questions_path: str, answers_path: str, output_prefix: str, limit: int | None = None, agentic: bool = False, blacklist_path: str | None = None) -> None:
    print(f"\n=== Ragas Custom Evaluation  [{output_prefix}] ===\n")

    questions = load_json(questions_path)
    if not Path(answers_path).exists():
        print(f"[error] Answers file not found: {answers_path}")
        return
    answers = load_json(answers_path)

    print(f"Questions : {len(questions)}")
    print(f"Answers   : {len(answers)}")
    print(f"Prefix    : {output_prefix}")

    # --- Apply blacklist (low-quality questions) ---
    blacklist = _load_blacklist(blacklist_path)
    if blacklist:
        before = len(questions)
        questions = [q for q in questions if str(q.get("id", "")) not in blacklist]
        print(f"Blacklist : {len(blacklist)} IDs loaded, {before - len(questions)} question(s) skipped  →  {len(questions)} eligible")

    eligible_count = len(questions)
    if limit:
        questions = questions[:limit]
        print(f"Limit     : first {limit} of {eligible_count} eligible")

    import pandas as pd  # type: ignore[import-untyped]

    out_csv = RESULTS_DIR / f"ragas_{output_prefix}_scores.csv"
    judge_state_path = RESULTS_DIR / f"ragas_{output_prefix}_judge_state.json"

    # --- Incremental: load what has already been done ---
    _SKIP_COLS = {"question_id", "is_answered", "user_input", "retrieved_contexts", "response", "reference"}
    existing_df: "pd.DataFrame | None" = None
    done_ids: set[str] = set()
    if out_csv.exists():
        raw_df = pd.read_csv(out_csv)
        # --- Migration: drop removed metric columns ---
        _DROP_COLS = ["llm_context_precision_with_reference", "factual_correctness(mode=f1)"]
        _dropped = [c for c in _DROP_COLS if c in raw_df.columns]
        if _dropped:
            raw_df = raw_df.drop(columns=_dropped)
            print(f"  [migration] Dropped legacy columns: {_dropped}")
        # --- Migration: compute llm_judge_correctness for existing rows ---
        if "llm_judge_correctness" not in raw_df.columns and len(raw_df) > 0:
            print(f"  [migration] Computing llm_judge_correctness for {len(raw_df)} existing rows...")
            _all_qs = load_json(questions_path)
            _existing_qids = set(raw_df["question_id"].astype(str))
            _qs_to_score = [q for q in _all_qs if str(q.get("id", "")) in _existing_qids]
            _scores = _score_answer_correctness_batch(_qs_to_score, answers)
            raw_df["llm_judge_correctness"] = raw_df["question_id"].astype(str).map(_scores)
            raw_df.to_csv(out_csv, index=False)
            print(f"  [migration] Saved with llm_judge_correctness")
        metric_cols_in_csv = [c for c in raw_df.columns if c not in _SKIP_COLS]
        if metric_cols_in_csv:
            # A row is "done" only when every metric column has a real value.
            # Rows with any NaN (from a previous run that hit quota errors) are
            # excluded so the affected questions get re-evaluated automatically.
            all_ok = raw_df[metric_cols_in_csv].notna().all(axis=1)
            existing_df = raw_df[all_ok].reset_index(drop=True) if all_ok.any() else None
            done_ids = set(raw_df.loc[all_ok, "question_id"].astype(str))
            n_incomplete = int((~all_ok).sum())
            if n_incomplete:
                print(f"  → {n_incomplete} row(s) with incomplete metrics will be re-evaluated")
        else:
            existing_df = raw_df
            done_ids = set(raw_df["question_id"].astype(str))
        print(f"Already evaluated : {len(done_ids)} (from {out_csv.name})")

    judge_state = _load_judge_state(judge_state_path)

    # Sync scores.csv → judge_state: questions that were fully evaluated (in done_ids)
    # but are missing from judge_state (e.g. from an earlier incremental run that skipped
    # the judge phase). Without this, they are excluded from judged_total in the summary.
    _synced = [qid for qid in done_ids if qid not in judge_state]
    if _synced:
        for qid in _synced:
            judge_state[qid] = True
        _save_judge_state(judge_state_path, judge_state)
        print(f"  [sync] Added {len(_synced)} missing judge_state entries from scores.csv")

    # Questions judged False are permanently skipped (no useful answer to evaluate).
    # Questions judged True but with incomplete Ragas metrics are NOT skipped so
    # they get a fresh Ragas pass (the judge result itself is discarded and re-run
    # to avoid the cost of tracking it separately).
    skip_ids = done_ids | {qid for qid, v in judge_state.items() if not v}
    pending_questions = [q for q in questions if str(q.get("id", "")) not in skip_ids]
    print(f"Pending (new)     : {len(pending_questions)}")

    new_df: "pd.DataFrame | None" = None

    if pending_questions:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            AnswerRelevancy,
            Faithfulness,
            LLMContextRecall,
        )

        llm = _build_ragas_llm()
        emb = _build_ragas_embeddings()

        metrics = [
            # strictness=1: ask LLM for 1 generation instead of 3; suppresses
            # "LLM returned 1 generations instead of requested 3" warning on
            # providers that don't support n>1.
            AnswerRelevancy(llm=llm, embeddings=emb, strictness=1),
            Faithfulness(llm=llm),
            LLMContextRecall(llm=llm),
        ]

        print("\n[filter] Rule-based answered detection (pattern: 'I don't know')...")
        answered_map = _rule_answered_batch(pending_questions, answers)
        n_yes_new = sum(answered_map.values())
        n_idk_new = len(answered_map) - n_yes_new
        print(f"  Answered: {n_yes_new}/{len(pending_questions)}  |  'I don't know': {n_idk_new}")

        judge_state.update(answered_map)
        _save_judge_state(judge_state_path, judge_state)

        answered_pending = [q for q in pending_questions if answered_map.get(str(q.get("id", "")), False)]
        if answered_pending:
            dataset = _build_ragas_dataset(answered_pending, answers)
            print(f"\n[ragas] Evaluating {len(dataset.samples)} new questions...")
            from ragas.run_config import RunConfig  # type: ignore[import-untyped]
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                run_config=RunConfig(timeout=300, max_retries=15, max_workers=2, max_wait=60),
            )

            new_df = result.to_pandas()
            id_col = [str(q.get("id", "")) for q in answered_pending]
            new_df.insert(0, "question_id", id_col)
            new_df.insert(1, "is_answered", True)

            print("\n[judge-correctness] Scoring answer correctness (LLM-as-a-judge)...")
            correctness_scores = _score_answer_correctness_batch(answered_pending, answers)
            new_df["llm_judge_correctness"] = [
                correctness_scores.get(str(q.get("id", "")), float("nan"))
                for q in answered_pending
            ]
        else:
            print("[warn] No new questions judged as answered — nothing new to evaluate.")
    else:
        print("[incremental] All questions already processed — skipping judge + Ragas.")

    # Ensure llm_judge_correctness column exists in existing_df to avoid concat column mismatches
    if existing_df is not None and "llm_judge_correctness" not in existing_df.columns:
        existing_df["llm_judge_correctness"] = float("nan")

    # Combine existing + new rows and save
    frames = [df for df in (existing_df, new_df) if df is not None]
    if not frames:
        print("[warn] No evaluated data — cannot compute summary.")
        return
    combined_df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    if new_df is not None:
        combined_df.to_csv(out_csv, index=False)
        print(f"\n[ragas] Per-question scores saved: {out_csv}")

    # Summary over all evaluated questions in this slice.
    # Two variants for each metric:
    #   {metric}          — Overall:     sum / judged_total  (unanswered questions count as 0)
    #   {metric}_answered — Conditional: mean of yes-only    (only questions that were answered)
    metric_cols = [
        c for c in combined_df.columns
        if c not in ("question_id", "is_answered", "user_input", "retrieved_contexts", "response", "reference")
    ]
    slice_ids = {str(q.get("id", "")) for q in questions}
    judged_in_slice = {qid: v for qid, v in judge_state.items() if qid in slice_ids}
    judged_total = len(judged_in_slice)
    judged_answered_count = sum(judged_in_slice.values())
    slice_df = combined_df[combined_df["question_id"].astype(str).isin(slice_ids)]
    summary: dict = {}
    for k in metric_cols:
        if slice_df[k].dtype.kind not in "fi":
            continue
        yes_sum = float(slice_df[k].sum(skipna=True))
        yes_mean = float(slice_df[k].mean())
        summary[k] = round(yes_sum / judged_total, 4) if judged_total > 0 else float("nan")
        summary[f"{k}_answered"] = round(yes_mean, 4)
    summary["judged_answered"] = judged_answered_count
    summary["judged_total"] = judged_total
    print("\n--- Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_summary = RESULTS_DIR / f"ragas_{output_prefix}_summary.json"
    save_json(out_summary, summary)
    print(f"\n✓ Summary saved: {out_summary}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Ragas-custom answers")
    parser.add_argument("--questions", default="tests/eval/data/ragas_custom_questions.json")
    parser.add_argument("--answers", default="tests/eval/results/ragas_custom_answers.json")
    parser.add_argument(
        "--output-prefix",
        default="custom",
        help="Prefix for output files: ragas_{prefix}_scores.csv / ragas_{prefix}_summary.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N questions (for quick tests)",
    )
    parser.add_argument(
        "--agentic",
        action="store_true",
        default=False,
        help="Mark this run as agentic mode (label only; no longer changes computed metrics).",
    )
    parser.add_argument(
        "--blacklist",
        default="tests/eval/data/bad_question_ids.json",
        help="Path to JSON file containing bad question IDs to skip (default: tests/eval/data/bad_question_ids.json)",
    )
    args = parser.parse_args()
    main(args.questions, args.answers, args.output_prefix, args.limit, args.agentic, args.blacklist)
