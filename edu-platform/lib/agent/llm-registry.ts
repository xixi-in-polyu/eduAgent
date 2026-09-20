/**
 * LLM Registry — role-based LLM client factory.
 *
 * Roles:
 *   chat    — main conversation & assignment generation (e.g. deepseek-flash)
 *   vision  — image understanding (e.g. qwen3.6-plus)
 *   title   — chat title generation, cheap/fast (e.g. deepseek-flash)
 *   memory  — memory extraction, auxiliary (e.g. qwen-plus / gpt-4o-mini)
 *
 * Fallback chains (first non-empty value wins):
 *
 *   chat   → key: LLM_CHAT_API_KEY  → LLM_API_KEY → OPENAI_API_KEY
 *            url: LLM_CHAT_BASE_URL → LLM_BASE_URL
 *          model: LLM_CHAT_MODEL    → LLM_MODEL → "gpt-4o"
 *
 *   title  → key: LLM_TITLE_API_KEY → LLM_CHAT_API_KEY → LLM_API_KEY → OPENAI_API_KEY
 *            url: LLM_TITLE_BASE_URL → LLM_CHAT_BASE_URL → LLM_BASE_URL
 *          model: LLM_TITLE_MODEL   → LLM_AUXILIARY_MODEL → LLM_MODEL → "gpt-4o-mini"
 *
 *   vision → key: LLM_VISION_API_KEY → LLM_API_KEY → OPENAI_API_KEY
 *            url: LLM_VISION_BASE_URL → LLM_BASE_URL
 *          model: LLM_VISION_MODEL  → LLM_MODEL → "gpt-4o"
 *
 *   memory → key: LLM_API_KEY → OPENAI_API_KEY
 *            url: LLM_BASE_URL
 *          model: LLM_AUXILIARY_MODEL → LLM_MODEL → "gpt-4o-mini"
 *
 *   grading → key: LLM_API_KEY → OPENAI_API_KEY
 *             url: LLM_BASE_URL
 *           model: LLM_AUXILIARY_MODEL → LLM_MODEL → "gpt-4o-mini"
 */

import OpenAI from "openai";
import { logger } from "@/lib/logger";

const log = logger.child({ component: "llm-registry" });

export type LLMRole = "chat" | "vision" | "title" | "memory" | "grading" | "completion";

export type RoleConfig = {
  apiKey: string;
  baseURL: string | undefined;
  model: string;
};

/** Return non-empty env var value or undefined. */
function e(name: string): string | undefined {
  const v = process.env[name];
  return v && v.trim() ? v.trim() : undefined;
}

export function getRoleConfig(role: LLMRole): RoleConfig {
  const defaultKey = e("LLM_API_KEY") ?? e("OPENAI_API_KEY") ?? "";
  const defaultBase = e("LLM_BASE_URL");
  const defaultModel = e("LLM_MODEL") ?? "gpt-4o";

  switch (role) {
    case "chat":
      return {
        apiKey: e("LLM_CHAT_API_KEY") ?? defaultKey,
        baseURL: e("LLM_CHAT_BASE_URL") ?? defaultBase,
        model: e("LLM_CHAT_MODEL") ?? defaultModel,
      };

    case "title":
      return {
        apiKey: e("LLM_TITLE_API_KEY") ?? e("LLM_CHAT_API_KEY") ?? defaultKey,
        baseURL: e("LLM_TITLE_BASE_URL") ?? e("LLM_CHAT_BASE_URL") ?? defaultBase,
        model: e("LLM_TITLE_MODEL") ?? e("LLM_AUXILIARY_MODEL") ?? e("LLM_MODEL") ?? "gpt-4o-mini",
      };

    case "vision":
      return {
        apiKey: e("LLM_VISION_API_KEY") ?? defaultKey,
        baseURL: e("LLM_VISION_BASE_URL") ?? defaultBase,
        model: e("LLM_VISION_MODEL") ?? defaultModel,
      };

    case "memory":
      return {
        apiKey: e("LLM_MEMORY_API_KEY") ?? e("LLM_CHAT_API_KEY") ?? defaultKey,
        baseURL: e("LLM_MEMORY_BASE_URL") ?? e("LLM_CHAT_BASE_URL") ?? defaultBase,
        model: e("LLM_AUXILIARY_MODEL") ?? e("LLM_MODEL") ?? "gpt-4o-mini",
      };

    case "grading":
      return {
        apiKey: e("LLM_MEMORY_API_KEY") ?? e("LLM_CHAT_API_KEY") ?? defaultKey,
        baseURL: e("LLM_MEMORY_BASE_URL") ?? e("LLM_CHAT_BASE_URL") ?? defaultBase,
        model: e("LLM_AUXILIARY_MODEL") ?? e("LLM_MODEL") ?? "gpt-4o-mini",
      };

    // completion role — FIM endpoint (deepseek-flash via beta URL by default)
    // Env: LLM_COMPLETION_API_KEY, LLM_COMPLETION_BASE_URL, LLM_COMPLETION_MODEL
    case "completion":
      return {
        apiKey: e("LLM_COMPLETION_API_KEY") ?? defaultKey,
        baseURL: e("LLM_COMPLETION_BASE_URL") ?? "https://api.deepseek.com/beta",
        model: e("LLM_COMPLETION_MODEL") ?? "deepseek-flash",
      };
  }
}

// ---------------------------------------------------------------------------
// Debug logging — enabled by LLM_DEBUG=true in .env
// ---------------------------------------------------------------------------

const _LLM_DEBUG = process.env.LLM_DEBUG === "true" || process.env.LLM_DEBUG === "1";
const _LLM_DEBUG_MAX_CHARS = Math.max(0, parseInt(process.env.LLM_DEBUG_MAX_CHARS ?? "400", 10) || 400);

function _clip(s: string): string {
  const str = (s ?? "").replace(/\n/g, "↵");
  return _LLM_DEBUG_MAX_CHARS > 0 && str.length > _LLM_DEBUG_MAX_CHARS
    ? str.slice(0, _LLM_DEBUG_MAX_CHARS) + "…"
    : str;
}

/**
 * Wraps an OpenAI client's `chat.completions.create` to log inputs, outputs
 * and errors when LLM_DEBUG=true. API keys are never logged.
 * Streaming calls log the input only (STREAM tag); the stream object is returned as-is.
 */
function _wrapClientForDebug(client: OpenAI, role: LLMRole): OpenAI {
  if (!_LLM_DEBUG) return client;

  const completions = client.chat.completions;
  const origCreate = completions.create.bind(completions) as typeof completions.create;

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  (completions as any).create = async function (params: any, options?: any) {
    const model: string = params.model ?? "(unknown)";
    const msgsSummary: string = Array.isArray(params.messages)
      ? (params.messages as Array<{ role: string; content: unknown }>)
          .map((m) => {
            const content =
              typeof m.content === "string" ? m.content : "[multimodal]";
            return `${m.role}:${_clip(content)}`;
          })
          .join(" │ ")
      : "(no messages)";
    if (params.stream) {
      log.debug({ role, model, msgs: msgsSummary, tools: params.tools?.length }, "LLM stream");
      try {
        return await origCreate(params, options);
      } catch (err) {
        log.error({ err, role, model, msgs: msgsSummary }, "LLM stream error");
        throw err;
      }
    }

    log.debug({ role, model, msgs: msgsSummary, tools: params.tools?.length }, "LLM call");
    try {
      const result = await origCreate(params, options);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const out: string = _clip((result as any)?.choices?.[0]?.message?.content ?? JSON.stringify(result));
      log.debug({ role, model, result: out }, "LLM ok");
      return result;
    } catch (err) {
      log.error({ err, role, model, msgs: msgsSummary }, "LLM call error");
      throw err;
    }
  };

  return client;
}

/** Create a new OpenAI-compatible client for the given role. */
export function getLLMClient(role: LLMRole): OpenAI {
  const { apiKey, baseURL } = getRoleConfig(role);
  const client = new OpenAI({ apiKey, baseURL });
  return _wrapClientForDebug(client, role);
}

/**
 * Returns extra_body to disable DeepSeek thinking mode, preventing 400 errors
 * when tool calls are present (reasoning_content not echoed back).
 * Returns undefined for non-DeepSeek providers (Qwen, OpenAI, etc.).
 */
export function getRoleExtraBody(
  role: LLMRole,
): Record<string, unknown> | undefined {
  const { baseURL, model } = getRoleConfig(role);
  const isDeepSeek =
    (baseURL ?? "").includes("deepseek.com") ||
    model.toLowerCase().startsWith("deepseek");
  return isDeepSeek ? { thinking: { type: "disabled" } } : undefined;
}

export function getChatModel(): string {
  return getRoleConfig("chat").model;
}

export function getVisionModel(): string {
  return getRoleConfig("vision").model;
}

export function getTitleModel(): string {
  return getRoleConfig("title").model;
}

export function getMemoryModel(): string {
  return getRoleConfig("memory").model;
}
