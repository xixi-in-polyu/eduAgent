import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import type { TurnContext, ToolResult } from "@/lib/agent/types";

// Mock the LLM layer so the low-confidence rewrite fallback never calls a real API.
vi.mock("@/lib/agent/llm-registry", () => ({
  getLLMClient: vi.fn().mockReturnValue({}),
  getRoleConfig: vi.fn().mockReturnValue({ apiKey: "sk-test", baseURL: "http://test", model: "test-model" }),
  getExtraBody: vi.fn().mockReturnValue(undefined),
}));
vi.mock("@/lib/agent/subagent", () => ({
  runSubAgent: vi.fn().mockResolvedValue({ success: false, summary: "" }),
}));

import { knowledgeQueryTool } from "@/lib/agent/tools/rag";

const ctx: TurnContext = {
  userId: "u-1",
  sessionId: "s-1",
  accessibleCourseIds: ["c-1", "c-2"],
  courseId: "c-1",
};

/** 无课程上下文（个人知识库场景） */
const ctxNoCourse: TurnContext = {
  userId: "u-2",
  sessionId: "s-2",
  accessibleCourseIds: [],
};

function asToolResult(value: string | ToolResult): ToolResult {
  if (typeof value === "string") {
    throw new Error("Expected ToolResult but received string");
  }
  return value;
}

function asString(value: string | ToolResult): string {
  if (typeof value !== "string") {
    throw new Error("Expected string but received ToolResult");
  }
  return value;
}

describe("RAG tools", () => {
  beforeEach(() => {
    vi.stubEnv("RAG_SERVICE_URL", "http://rag.test");
    vi.stubEnv("RAG_SERVICE_API_KEY", "internal-key");
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.unstubAllGlobals();
  });

  it("性能规则：默认只请求 3 个片段并限制注入模型的上下文", async () => {
    // given
    vi.stubEnv("RAG_CONTEXT_MAX_CHARS", "500");
    vi.stubEnv("RAG_CONTEXT_MAX_CHARS_PER_HIT", "300");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        hits: [
          { chunk_id: "a", text: "A".repeat(600), origin: "course", relevance_score: 0.9 },
          { chunk_id: "b", text: "B".repeat(600), origin: "course", relevance_score: 0.8 },
        ],
        warnings: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    // when
    const result = asToolResult(await knowledgeQueryTool.execute(
      { question: "TCP 是什么？", sources: "course" },
      ctx,
    ));

    // then
    const [, req] = fetchMock.mock.calls[0] as [string, RequestInit];
    const payload = JSON.parse(String(req.body)) as { top_k: number };
    expect(payload.top_k).toBe(3);
    expect(result.meta).toMatchObject({ context_chars: 500, context_truncated: true });
    expect(result.content).toContain("A".repeat(300));
    expect(result.content).toContain("B".repeat(200));
    expect(result.content).not.toContain("B".repeat(201));
  });

  it("业务规则：knowledge_query 缺少 question 时应返回可读错误", async () => {
    // given

    // when
    const result = asToolResult(await knowledgeQueryTool.execute({ sources: "course" }, ctx));

    // then
    expect(result.content).toContain("缺少必要参数：question");
  });

  it("业务规则：knowledge_query 对非法 sources 应拒绝并返回契约错误", async () => {
    // given

    // when
    const result = asToolResult(await knowledgeQueryTool.execute(
      { question: "什么是 TCP", sources: "invalid-source" },
      ctx,
    ));

    // then
    expect(result.content).toContain("非法 sources");
  });

  it("业务规则：knowledge_query 应把课程+个人范围聚合为 all 并生成引用", async () => {
    // given
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        hits: [
          {
            chunk_id: "chunk-1",
            text: "TCP 通过三次握手建立可靠连接。",
            origin: "course",
            course_id: "c-1",
            material_id: "m-1",
            material_title: "网络基础",
          },
        ],
        warnings: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    // when
    const result = asToolResult(await knowledgeQueryTool.execute(
      {
        question: "TCP 如何建立连接？",
        sources: ["course", "personal"],
        top_k: 50,
      },
      ctx,
    ));

    // then
    expect(result.content).toContain("来源：网络基础");
    expect(result.content).toContain("TCP 通过三次握手建立可靠连接");
    expect(result.citations?.[0]).toMatchObject({
      chunk_id: "chunk-1",
      material_id: "m-1",
      source_label: "网络基础",
    });
    const [, req] = fetchMock.mock.calls[0] as [string, RequestInit];
    const payload = JSON.parse(String(req.body)) as Record<string, unknown>;
    expect(payload.source).toBe("all");
    expect(payload.top_k).toBe(20);
  });

  it("业务规则：主 Agent 给出子查询时应并行检索、去重并限制总返回数", async () => {
    // given
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          hits: [
            { chunk_id: "shared", text: "shared-a", origin: "course", relevance_score: 0.8 },
            { chunk_id: "a", text: "a", origin: "course", relevance_score: 0.7 },
          ],
          warnings: [],
        }),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          hits: [
            { chunk_id: "shared", text: "shared-b", origin: "course", relevance_score: 0.9 },
            { chunk_id: "b", text: "b", origin: "course", relevance_score: 0.6 },
          ],
          warnings: [],
        }),
      });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    // when
    const result = asToolResult(await knowledgeQueryTool.execute(
      {
        question: "IP 和 MAC 地址有什么区别？",
        sources: "course",
        sub_queries: ["IP 地址的作用", "MAC 地址的作用"],
        top_k: 2,
      },
      ctx,
    ));

    // then
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const questions = fetchMock.mock.calls.map(([, req]) =>
      (JSON.parse(String((req as RequestInit).body)) as { question: string }).question,
    );
    expect(questions).toEqual(["IP 地址的作用", "MAC 地址的作用"]);
    expect(result.meta).toMatchObject({
      decomposed: true,
      sub_queries: ["IP 地址的作用", "MAC 地址的作用"],
      hit_count: 2,
    });
    expect(result.citations).toHaveLength(2);
    expect(result.content).toContain("shared-b");
    expect(result.content).not.toContain("shared-a");
  });

  it("业务规则：未给出有效子查询时应直接检索原问题", async () => {
    // given
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ hits: [], warnings: [] }),
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    // when
    const result = asToolResult(await knowledgeQueryTool.execute(
      { question: "TCP 是什么？", sources: "course", sub_queries: ["TCP 定义"] },
      ctx,
    ));

    // then
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [, req] = fetchMock.mock.calls[0] as [string, RequestInit];
    const payload = JSON.parse(String(req.body)) as { question: string };
    expect(payload.question).toBe("TCP 是什么？");
    expect(result.meta).toMatchObject({ decomposed: false });
  });

  // ---- 个人知识库（personal KB）业务规则 -------------------------------------

  it("业务规则：sources=personal 在无课程上下文时应正常发起查询", async () => {
    // given
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        hits: [
          {
            chunk_id: "pchunk-1",
            text: "个人笔记：分布式系统核心概念。",
            origin: "personal",
            material_title: "我的笔记",
          },
        ],
        warnings: [],
      }),
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);

    // when
    const result = asToolResult(
      await knowledgeQueryTool.execute({ question: "什么是分布式系统", sources: "personal" }, ctxNoCourse),
    );

    // then
    expect(result.content).toContain("来源：我的笔记");
    expect(result.content).toContain("分布式系统核心概念");
    const [, req] = fetchMock.mock.calls[0] as [string, RequestInit];
    const payload = JSON.parse(String(req.body)) as Record<string, unknown>;
    expect(payload.source).toBe("personal");
    expect(payload.course_id).toBeNull();
    expect(payload.user_id).toBe("u-2");
  });

  it("业务规则：sources=course 在无课程上下文时应返回业务错误", async () => {
    // given

    // when
    const result = asToolResult(
      await knowledgeQueryTool.execute({ question: "课程内容", sources: "course" }, ctxNoCourse),
    );

    // then
    expect(result.content).toContain("当前会话未绑定课程");
  });

  it("业务规则：sources=enrolled_courses 在有课程上下文时应返回业务错误", async () => {
    // given

    // when
    const result = asToolResult(
      await knowledgeQueryTool.execute(
        { question: "相关课程有哪些", sources: "enrolled_courses" },
        ctx,
      ),
    );

    // then
    expect(result.content).toContain("禁止使用 sources=enrolled_courses");
  });
});
