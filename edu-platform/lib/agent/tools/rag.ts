/**
 * Vector RAG tool: knowledge_query
 * All call the Python RAG microservice via HTTP.
 */

import type { Tool, TurnContext, ToolResult } from "../types";
import { runSubAgent } from "../subagent";
import { getLLMClient, getRoleConfig } from "../llm-registry";
import { logger } from "@/lib/logger";

const log = logger.child({ component: "tool:rag" });

// ---- Shared HTTP helper ----------------------------------------------------

async function ragPost<T>(
  url: string,
  key: string,
  body: Record<string, unknown>,
): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(key ? { "x-internal-key": key } : {}),
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`RAG service error ${res.status}: ${text.slice(0, 400)}`);
  }
  return res.json() as Promise<T>;
}

// ---- Hit formatting --------------------------------------------------------

type HitItem = {
  chunk_id: string;
  text: string;
  origin: string;
  course_id?: string | null;
  material_id?: string | null;
  material_title?: string | null;
  relevance_score?: number;
  image_urls?: Array<{ page_idx: number; url: string }>;
};

function _boundedEnvInt(name: string, fallback: number, min: number, max: number): number {
  const parsed = Number.parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(parsed) ? Math.max(min, Math.min(max, parsed)) : fallback;
}

function _formatHitsForLlm(hits: HitItem[]): {
  content: string;
  contextChars: number;
  truncated: boolean;
} {
  if (hits.length === 0) {
    return { content: "（未找到相关内容）", contextChars: 0, truncated: false };
  }
  const totalLimit = _boundedEnvInt("RAG_CONTEXT_MAX_CHARS", 6000, 500, 50000);
  const perHitLimit = _boundedEnvInt("RAG_CONTEXT_MAX_CHARS_PER_HIT", 2000, 200, 20000);
  let remaining = totalLimit;
  let contextChars = 0;
  let truncated = false;
  const sections: string[] = [];
  for (const [index, hit] of hits.entries()) {
    if (remaining <= 0) {
      truncated = true;
      break;
    }
    const limit = Math.min(perHitLimit, remaining);
    const text = hit.text.slice(0, limit);
    const wasTruncated = text.length < hit.text.length;
    const src = hit.material_title ?? hit.course_id ?? hit.origin;
    sections.push(
      `[${index + 1}] 来源：${src}\n${text}${wasTruncated ? "\n（片段已截断）" : ""}`,
    );
    contextChars += text.length;
    remaining -= text.length;
    truncated ||= wasTruncated;
  }
  return { content: sections.join("\n\n---\n\n"), contextChars, truncated };
}

function _hitsToB3Citations(hits: HitItem[]): ToolResult["citations"] {
  return hits.map((h) => ({
    chunk_id: h.chunk_id,
    material_id: h.material_id ?? undefined,
    source_label: h.material_title ?? h.course_id ?? h.origin,
    chunk_text: h.text.slice(0, 300),
    eval_text: h.text,
    image_urls: h.image_urls?.length ? h.image_urls : undefined,
  }));
}

// ---- Agentic-RAG helpers ---------------------------------------------------

const RELEVANCE_THRESHOLD = 0.2;

/** Merge hits from multiple sub-queries, deduplicating by chunk_id (keep max score). */
function _mergeHits(hitArrays: HitItem[][]): HitItem[] {
  const map = new Map<string, HitItem>();
  for (const hits of hitArrays) {
    for (const hit of hits) {
      const existing = map.get(hit.chunk_id);
      if (!existing || (hit.relevance_score ?? 0) > (existing.relevance_score ?? 0)) {
        map.set(hit.chunk_id, hit);
      }
    }
  }
  return Array.from(map.values()).sort(
    (a, b) => (b.relevance_score ?? 0) - (a.relevance_score ?? 0),
  );
}

/** Ask sub-agent (title model, no tools) to rewrite a low-confidence query. */
async function _rewriteQuery(question: string, ctx: TurnContext): Promise<string | null> {
  const client = getLLMClient("title");
  const { model } = getRoleConfig("title");
  const task =
    `将以下查询改写为更精确、专业、适合知识库语义检索的形式。仅输出 JSON：{"rewritten": "改写后的查询"}\n\n原始查询：${question}`;
  try {
    const result = await runSubAgent(client, model, { task, allowedTools: [], ctx, temperature: ctx?.evalMode ? 0 : undefined }, 0);
    if (!result.success) return null;
    const jsonMatch = result.summary.match(/\{[\s\S]*\}/);
    if (!jsonMatch) return null;
    const parsed = JSON.parse(jsonMatch[0]) as { rewritten?: string };
    const rewritten = parsed.rewritten?.trim();
    return rewritten && rewritten !== question ? rewritten : null;
  } catch {
    return null;
  }
}

// ---- knowledge_query -------------------------------------------------------

function _normalizeSource(raw: unknown): string | null {
  if (typeof raw === "string") {
    const s = raw.trim().toLowerCase();
    if (["personal", "course", "all", "enrolled_courses"].includes(s)) return s;
  }
  if (Array.isArray(raw)) {
    const items = (raw as unknown[])
      .filter((x) => typeof x === "string")
      .map((x) => (x as string).trim().toLowerCase());
    if (items.every((x) => ["course", "personal"].includes(x))) {
      const set = new Set(items);
      if (set.has("course") && set.has("personal")) return "all";
      if (set.has("course")) return "course";
      if (set.has("personal")) return "personal";
    }
  }
  return null;
}

export const knowledgeQueryTool: Tool = {
  name: "knowledge_query",
  description:
    "从知识库中检索信息，回答关于已导入文档/课程资料（如 PPT、PDF、讲义）的任何问题。" +
    "在回答概念、原理、定义、事实类问题时应优先调用此工具。" +
    "你必须在本次工具调用中自行判断是否需要拆分：简单问题不要传 sub_queries；" +
    "只有包含多个独立检索意图时，才传入 2–3 个子查询。",
  parameters: {
    type: "object",
    properties: {
      question: { type: "string", minLength: 1, maxLength: 2000, description: "要查询的自然语言问题" },
      sources: {
        description:
          "必填。字符串：personal | course | all | enrolled_courses；或数组：[course, personal]。",
        oneOf: [
          { type: "string", enum: ["personal", "course", "all", "enrolled_courses"] },
          {
            type: "array",
            items: { type: "string", enum: ["course", "personal"] },
            minItems: 1,
            maxItems: 2,
            uniqueItems: true,
          },
        ],
      },
      top_k: { type: "integer", minimum: 1, maximum: 20, description: "返回最大片段数（默认 3，范围 1–20）" },
      sub_queries: {
        type: "array",
        items: { type: "string", minLength: 1, maxLength: 500 },
        minItems: 2,
        maxItems: 3,
        uniqueItems: true,
        description:
          "可选。仅当问题包含多个独立检索意图时，由你在本次工具调用中直接给出 2–3 个完整、可独立检索的子查询；简单问题必须省略。",
      },
    },
    required: ["question", "sources"],
  },
  async execute(args: Record<string, unknown>, ctx: TurnContext): Promise<ToolResult> {
    const t0 = Date.now();
    const ragUrl = process.env.RAG_SERVICE_URL ?? "http://localhost:8001";
    const ragKey = process.env.RAG_SERVICE_API_KEY ?? "";

    const question = typeof args.question === "string" ? args.question.trim() : "";
    if (!question) {
      return { content: JSON.stringify({ error: "缺少必要参数：question" }) };
    }
    log.debug({ question: question.slice(0, 80), sources: args.sources, top_k: args.top_k, courseId: ctx.courseId }, "knowledge_query start");

    const source = _normalizeSource(args.sources);
    if (!source && !ctx.evalMode) {
      return {
        content: JSON.stringify({
          error: "非法 sources（仅允许 personal、course、all、enrolled_courses 或 [course, personal]）",
        }),
      };
    }

    if (!ctx.evalMode) {
      if (ctx.courseId && source === "enrolled_courses") {
        return {
          content: JSON.stringify({
            error: "当前会话已绑定课程，禁止使用 sources=enrolled_courses",
          }),
        };
      }
      if (!ctx.courseId && (source === "course" || source === "all")) {
        return {
          content: JSON.stringify({
            error: "当前会话未绑定课程，仅允许使用 personal 或 enrolled_courses",
          }),
        };
      }
    }

    // In eval mode, always restrict retrieval to course KB regardless of what the LLM requested.
    const effectiveSource = ctx.evalMode ? "course" : source!;

    const top_k = typeof args.top_k === "number" ? Math.max(1, Math.min(20, args.top_k)) : 3;
    type QueryResp = { hits: HitItem[]; warnings: string[] };
    const baseBody = {
      source: effectiveSource,
      user_id: ctx.userId,
      accessible_course_ids: ctx.accessibleCourseIds,
      course_id: ctx.courseId ?? null,
      top_k,
    };

    // The main agent decides whether decomposition is needed while producing
    // this tool call. Avoid a second, unconditional LLM round-trip here.
    const requestedSubQueries = Array.isArray(args.sub_queries)
      ? args.sub_queries
          .filter((q): q is string => typeof q === "string")
          .map((q) => q.trim())
          .filter((q) => q.length > 0 && q.length <= 500)
      : [];
    const subQueries = [...new Set(requestedSubQueries)].slice(0, 3);
    const decomposed = subQueries.length >= 2;
    let hits: HitItem[];
    if (decomposed) {
      await ctx.onProgress?.(`正在分解为 ${subQueries.length} 个子查询并检索…`);
      const results = await Promise.all(
        subQueries.map((q) =>
          ragPost<QueryResp>(`${ragUrl}/rag/query`, ragKey, { ...baseBody, question: q }),
        ),
      );
      hits = _mergeHits(results.map((r) => r.hits)).slice(0, top_k);
    } else {
      await ctx.onProgress?.("正在检索知识库…");
      const resp = await ragPost<QueryResp>(`${ragUrl}/rag/query`, ragKey, {
        ...baseBody,
        question,
      });
      hits = resp.hits;
    }

    // ---- Phase 2: Relevance verification ---------------------------------
    const maxScore =
      hits.length > 0 ? Math.max(...hits.map((h) => h.relevance_score ?? 0)) : 0;
    let lowConfidence = hits.length === 0 || maxScore < RELEVANCE_THRESHOLD;

    // ---- Phase 3: Adaptive query rewriting --------------------------------
    let rewritten = false;
    let rewrittenQuery: string | undefined;
    if (lowConfidence) {
      await ctx.onProgress?.("置信度不足，正在改写查询…");
      const newQuery = await _rewriteQuery(question, ctx);
      if (newQuery) {
        await ctx.onProgress?.("正在以改写后的查询重新检索…");
        const retryResp = await ragPost<QueryResp>(`${ragUrl}/rag/query`, ragKey, {
          ...baseBody,
          question: newQuery,
        });
        const retryMax =
          retryResp.hits.length > 0
            ? Math.max(...retryResp.hits.map((h) => h.relevance_score ?? 0))
            : 0;
        if (retryMax > maxScore) {
          hits = retryResp.hits;
          lowConfidence = retryMax < RELEVANCE_THRESHOLD;
          rewritten = true;
          rewrittenQuery = newQuery;
        }
      }
    }

    // ---- Format result ---------------------------------------------------
    const formatted = _formatHitsForLlm(hits);
    let content = formatted.content;
    if (rewritten) content += "\n\n（已自动改写查询）";
    if (lowConfidence) content += "\n\n[置信度: 低]";

    const citations = _hitsToB3Citations(hits);
    log.debug({ hitCount: hits.length, lowConfidence, rewritten, durationMs: Date.now() - t0 }, "knowledge_query done");
    const meta: Record<string, unknown> = {
      decomposed,
      ...(decomposed ? { sub_queries: subQueries } : {}),
      rewritten,
      ...(rewritten && rewrittenQuery ? { rewritten_query: rewrittenQuery } : {}),
      hit_count: hits.length,
      context_chars: formatted.contextChars,
      context_truncated: formatted.truncated,
      max_score: hits.length > 0 ? Math.max(...hits.map((h) => h.relevance_score ?? 0)) : 0,
    };
    return { content, citations, meta };
  },
};
