"""使用配置的评测模型筛选低质量题目（图注查找类、非问句类等）。

用法：
  python -m tests.eval.filter_bad_questions
  python -m tests.eval.filter_bad_questions --questions tests/eval/data/ragas_custom_questions.json
  python -m tests.eval.filter_bad_questions --output tests/eval/data/bad_question_ids.json

输出：
  tests/eval/data/bad_question_ids.json  —— 低质量题目的 id 列表
  tests/eval/data/bad_questions_detail.json  —— 包含 LLM 判断理由的详细信息
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap path so src/ packages are importable
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "edu-platform"))

from tests.eval._common import DATA_DIR, RESULTS_DIR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QUESTIONS_FILE = DATA_DIR / "ragas_custom_questions.json"
BAD_IDS_FILE = DATA_DIR / "bad_question_ids.json"
DETAIL_FILE = DATA_DIR / "bad_questions_detail.json"

# 每批发给 LLM 的题目数
BATCH_SIZE = 20
# 并发批次数
MAX_CONCURRENT = 3

JUDGE_SYSTEM = """\
你是一名专业的题目质量评估专家，负责筛查 RAG 评测数据集中的低质量题目。

低质量题目的判定标准（满足任意一条即为低质量）：
1. 问题本身不是真正的疑问句——例如只写了图片编号（"Figure 3.11"）或术语名称，没有动词和疑问词；
2. 答案仅仅是对图片标注的直接改写，例如 gold_answer 形如 "Figure X.XX shows ..." 且完全可以从图注中复制粘贴得到；
3. 问题不需要任何知识推理，答案可以直接从上下文的图片说明/表格标题/章节标题一字不差地找到；
4. 问题语法严重残缺、语义完全不清晰，无法判断用户真正想问什么。

高质量题目的特征（不应被过滤）：
- 需要理解、推理或综合多个知识点才能回答；
- 答案虽然来自文本，但需要整合信息或进行解释；
- 即使有语法小错误，只要语义明确、有实质内容也算高质量。

请对每道题目做出判断，严格按照以下格式逐题输出（不要输出其他内容）：
ID: <题目id>
判断: valid 或 invalid
原因: <一句话说明原因>
---
"""

JUDGE_USER_TEMPLATE = """\
请评判以下 {n} 道题目：

{items}
"""

ITEM_TEMPLATE = """\
=== 题目 {idx} ===
ID: {id}
问题: {question}
参考答案: {gold_answer}
"""


def _build_items_text(batch: list[dict]) -> str:
    return "\n".join(
        ITEM_TEMPLATE.format(
            idx=i + 1,
            id=q["id"],
            question=q["question"],
            gold_answer=q["gold_answer"],
        )
        for i, q in enumerate(batch)
    )


def _parse_response(text: str, batch: list[dict]) -> list[dict]:
    """解析 LLM 返回的判断结果，容忍格式不规整。"""
    results: list[dict] = []
    # 按 --- 或 ID: 切割块
    blocks = [b.strip() for b in text.split("---") if b.strip()]
    if not blocks:
        blocks = [text]

    parsed_ids: set[str] = set()
    for block in blocks:
        lines = [l.strip() for l in block.splitlines() if l.strip()]
        entry: dict = {}
        for line in lines:
            if line.lower().startswith("id:"):
                entry["id"] = line.split(":", 1)[1].strip()
            elif line.lower().startswith("判断:") or line.lower().startswith("判断："):
                verdict = line.split(":", 1)[-1].split("：", 1)[-1].strip().lower()
                entry["verdict"] = "invalid" if "invalid" in verdict else "valid"
            elif line.lower().startswith("原因:") or line.lower().startswith("原因："):
                entry["reason"] = line.split(":", 1)[-1].split("：", 1)[-1].strip()
        if "id" in entry and "verdict" in entry:
            parsed_ids.add(entry["id"])
            results.append(entry)

    # 对解析失败的题目，记录为 parse_error（保守处理：不过滤）
    batch_ids = {str(q["id"]) for q in batch}
    for missing_id in batch_ids - parsed_ids:
        results.append({
            "id": missing_id,
            "verdict": "parse_error",
            "reason": "LLM 输出中未找到该题目的判断结果",
        })
    return results


async def _judge_batch_async(
    client,
    model: str,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    items_text = _build_items_text(batch)
    user_msg = JUDGE_USER_TEMPLATE.format(n=len(batch), items=items_text)

    async with semaphore:
        request_kwargs: dict = {}
        if model.lower().startswith("deepseek"):
            request_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=2048,
            **request_kwargs,
        )
    text = resp.choices[0].message.content or ""
    return _parse_response(text, batch)


async def _run_all(questions: list[dict], api_key: str, base_url: str, model: str) -> list[dict]:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        print("ERROR: openai 包未安装，请运行 pip install openai", file=sys.stderr)
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    batches = [questions[i : i + BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]
    print(f"共 {len(questions)} 道题，分 {len(batches)} 批（每批 {BATCH_SIZE} 道），并发 {MAX_CONCURRENT}")

    tasks = [_judge_batch_async(client, model, batch, semaphore) for batch in batches]

    all_results: list[dict] = []
    done = 0
    for coro in asyncio.as_completed(tasks):
        batch_results = await coro
        all_results.extend(batch_results)
        done += 1
        invalid_so_far = sum(1 for r in all_results if r["verdict"] == "invalid")
        print(f"  [{done}/{len(batches)}] 已处理 {len(all_results)} 题，其中 invalid: {invalid_so_far}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="筛选低质量 RAG 评测题目")
    parser.add_argument("--questions", default=str(QUESTIONS_FILE))
    parser.add_argument("--output", default=str(BAD_IDS_FILE))
    parser.add_argument("--detail", default=str(DETAIL_FILE))
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    # 读取配置
    try:
        from rag_mvp.config import settings  # type: ignore[import-untyped]
        api_key = settings.effective_chat_api_key
        base_url = settings.effective_chat_base_url
        configured_model = settings.effective_chat_model
    except Exception as e:
        print(f"无法加载 settings: {e}，尝试直接读取环境变量", file=sys.stderr)
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_CHAT_BASE_URL") or os.environ.get(
            "LLM_BASE_URL", "https://api.deepseek.com"
        )
        configured_model = os.environ.get("LLM_CHAT_MODEL") or os.environ.get(
            "LLM_MODEL", "deepseek-flash"
        )

    model = args.model or configured_model

    if not api_key:
        print("ERROR: 未找到 LLM_API_KEY / llm_api_key，请检查 .env 或环境变量", file=sys.stderr)
        sys.exit(1)

    print(f"使用模型 : {model}")
    print(f"Base URL : {base_url}")
    print(f"题目文件 : {args.questions}")

    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    print(f"题目总数 : {len(questions)}")

    results = asyncio.run(_run_all(questions, api_key, base_url, model))

    # 排序结果
    results.sort(key=lambda r: int(r["id"]) if str(r["id"]).isdigit() else 0)

    # 统计
    invalid_results = [r for r in results if r["verdict"] == "invalid"]
    valid_results = [r for r in results if r["verdict"] == "valid"]
    error_results = [r for r in results if r["verdict"] == "parse_error"]

    print(f"\n=== 筛选结果 ===")
    print(f"  valid        : {len(valid_results)}")
    print(f"  invalid      : {len(invalid_results)}")
    print(f"  parse_error  : {len(error_results)}")

    # 输出低质量 id 列表
    bad_ids = [r["id"] for r in invalid_results]
    Path(args.output).write_text(
        json.dumps(bad_ids, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n低质量题目 id 已写入: {args.output}")

    # 输出详细信息
    Path(args.detail).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"详细判断结果已写入: {args.detail}")

    # 打印低质量题目预览
    if invalid_results:
        print(f"\n低质量题目 ID 列表: {bad_ids}")


if __name__ == "__main__":
    main()
