"""One-shot concurrent SSE TTFT benchmark for the course Agent endpoint.

Run from the repository root after starting the web and RAG services:
    uv run python tests/perf/ttft_concurrent.py --concurrency 100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]


async def _login(client: httpx.AsyncClient, base_url: str, index: int) -> str:
    username = f"mock_student_{index:02d}"
    password = os.environ.get("PERF_USER_PASSWORD", "MockStudent@2026")
    response = await client.post(
        f"{base_url}/api/v1/login",
        json={"username": username, "password": password},
    )
    response.raise_for_status()
    return str(response.json()["token"])


async def _chat(
    client: httpx.AsyncClient,
    *,
    base_url: str,
    course_id: str,
    token: str,
    question: str,
    eval_mode: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    first_event_ms: float | None = None
    knowledge_tool_call_ms: float | None = None
    first_text_ms: float | None = None
    post_tool_text_ms: float | None = None
    tool_seen = False
    tool_done = False
    tool_duration_ms: float | None = None
    sub_query_count = 0
    rewritten = False
    context_chars: float | None = None
    status_code = 0
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{base_url}/api/v1/courses/{course_id}/chat",
            headers={"authorization": f"Bearer {token}", "accept": "text/event-stream"},
            json={"message": question, "eval_mode": eval_mode},
        ) as response:
            status_code = response.status_code
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue
                elapsed_ms = (time.perf_counter() - started) * 1000
                if first_event_ms is None:
                    first_event_ms = elapsed_ms
                kind = event.get("type")
                if kind == "tool_call" and event.get("name") == "knowledge_query":
                    tool_seen = True
                    knowledge_tool_call_ms = knowledge_tool_call_ms or elapsed_ms
                    tool_input = event.get("input")
                    if isinstance(tool_input, dict) and isinstance(
                        tool_input.get("sub_queries"), list
                    ):
                        sub_query_count = len(tool_input["sub_queries"])
                elif kind == "tool_result" and event.get("name") == "knowledge_query":
                    tool_done = True
                    duration = event.get("duration_ms")
                    if isinstance(duration, (int, float)):
                        tool_duration_ms = float(duration)
                    meta = event.get("meta")
                    if isinstance(meta, dict):
                        rewritten = meta.get("rewritten") is True
                        chars = meta.get("context_chars")
                        if isinstance(chars, (int, float)):
                            context_chars = float(chars)
                elif kind == "text" and event.get("content"):
                    first_text_ms = first_text_ms or elapsed_ms
                    if tool_done and post_tool_text_ms is None:
                        post_tool_text_ms = elapsed_ms
                elif kind == "done" and event.get("error"):
                    error = str(event["error"])
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    meaningful_ttft_ms = post_tool_text_ms if tool_seen else first_text_ms
    return {
        "status": status_code,
        "first_event_ms": first_event_ms,
        "knowledge_tool_call_ms": knowledge_tool_call_ms,
        "meaningful_ttft_ms": meaningful_ttft_ms,
        "tool_duration_ms": tool_duration_ms,
        "sub_query_count": sub_query_count,
        "rewritten": rewritten,
        "context_chars": context_chars,
        "wall_ms": (time.perf_counter() - started) * 1000,
        "error": error,
    }


async def _main(args: argparse.Namespace) -> None:
    limits = httpx.Limits(
        max_connections=args.concurrency + args.accounts,
        max_keepalive_connections=args.concurrency + args.accounts,
    )
    timeout = httpx.Timeout(args.timeout, connect=20)
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        tokens = await asyncio.gather(
            *(_login(client, args.base_url, index) for index in range(1, args.accounts + 1))
        )
        warm = await _chat(
            client,
            base_url=args.base_url,
            course_id=args.course_id,
            token=tokens[0],
            question="预热请求：简要解释 TCP 三次握手的目的",
            eval_mode=args.eval_mode,
        )
        print(json.dumps({"warmup": warm}, ensure_ascii=False), flush=True)

        stamp = time.time_ns()
        started = time.perf_counter()
        results = await asyncio.gather(
            *(
                _chat(
                    client,
                    base_url=args.base_url,
                    course_id=args.course_id,
                    token=tokens[index % len(tokens)],
                    question=(
                        f"并发压测 {stamp}-{index + 1}：什么是 TCP 三次握手？"
                    ),
                    eval_mode=args.eval_mode,
                )
                for index in range(args.concurrency)
            )
        )
        batch_wall_ms = (time.perf_counter() - started) * 1000

    ttft = [
        float(item["meaningful_ttft_ms"])
        for item in results
        if item["meaningful_ttft_ms"] is not None
    ]
    tools = [
        float(item["tool_duration_ms"])
        for item in results
        if item["tool_duration_ms"] is not None
    ]
    first_events = [
        float(item["first_event_ms"])
        for item in results
        if item["first_event_ms"] is not None
    ]
    tool_calls = [
        float(item["knowledge_tool_call_ms"])
        for item in results
        if item["knowledge_tool_call_ms"] is not None
    ]
    walls = [float(item["wall_ms"]) for item in results]
    context_sizes = [
        float(item["context_chars"])
        for item in results
        if item["context_chars"] is not None
    ]
    summary = {
        "concurrency": args.concurrency,
        "http_200": sum(item["status"] == 200 for item in results),
        "meaningful_ttft_samples": len(ttft),
        "errors": sum(bool(item["error"]) for item in results),
        "knowledge_tool_samples": len(tools),
        "first_event_p50_ms": round(_percentile(first_events, 0.50) or 0),
        "first_event_p95_ms": round(_percentile(first_events, 0.95) or 0),
        "knowledge_tool_call_p50_ms": round(_percentile(tool_calls, 0.50) or 0),
        "knowledge_tool_call_p95_ms": round(_percentile(tool_calls, 0.95) or 0),
        "decomposed_requests": sum(item["sub_query_count"] >= 2 for item in results),
        "rewritten_requests": sum(item["rewritten"] for item in results),
        "context_chars_p50": round(_percentile(context_sizes, 0.50) or 0),
        "context_chars_p95": round(_percentile(context_sizes, 0.95) or 0),
        "batch_wall_ms": round(batch_wall_ms),
        "ttft_p50_ms": round(_percentile(ttft, 0.50) or 0),
        "ttft_p95_ms": round(_percentile(ttft, 0.95) or 0),
        "ttft_p99_ms": round(_percentile(ttft, 0.99) or 0),
        "tool_p50_ms": round(_percentile(tools, 0.50) or 0),
        "tool_p95_ms": round(_percentile(tools, 0.95) or 0),
        "response_wall_p95_ms": round(_percentile(walls, 0.95) or 0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    failures = [item for item in results if item["status"] != 200 or item["error"]]
    if failures:
        print(json.dumps({"failure_examples": failures[:5]}, ensure_ascii=False, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--accounts", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument(
        "--course-id",
        default=os.environ.get("PERF_COURSE_ID", "c8b8787f-9c7e-4f37-bab5-fb94a438d9cf"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_main(_parse_args()))
