/**
 * Vision tool: analyzeImage
 * Lets the main chat model (e.g. deepseek-flash) delegate image understanding
 * to the dedicated vision model (e.g. qwen3.6-plus) by passing image URLs and a question.
 */

import OpenAI from "openai";
import { getLLMClient, getVisionModel } from "../llm-registry";
import type { Tool, TurnContext } from "../types";
import { logger } from "@/lib/logger";

const log = logger.child({ component: "tool:vision" });

function isPrivateOrLocalHostname(hostname: string): boolean {
  const h = hostname.trim().toLowerCase();
  if (!h) return true;
  if (h === "localhost" || h === "::1" || h.endsWith(".local")) return true;
  if (!h.includes(".")) return true;
  if (/^127\./.test(h)) return true;
  if (/^10\./.test(h)) return true;
  if (/^192\.168\./.test(h)) return true;
  if (/^169\.254\./.test(h)) return true;
  if (/^172\.(1[6-9]|2\d|3[0-1])\./.test(h)) return true;
  return false;
}

function normalizeVisionUrl(raw: string): string {
  try {
    // URL constructor correctly handles already-percent-encoded sequences
    // (preserves them) and raw Unicode characters (encodes them).
    // Avoid encodeURI which would double-encode existing %XX sequences.
    return new URL(raw.trim()).href;
  } catch {
    return encodeURI(raw.trim());
  }
}

async function toInlineDataUrl(url: string): Promise<string | null> {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) return null;
  const contentType = (res.headers.get("content-type") ?? "image/png")
    .split(";")[0]
    .trim()
    .toLowerCase();
  if (!contentType.startsWith("image/")) return null;
  const bytes = Buffer.from(await res.arrayBuffer());
  return `data:${contentType};base64,${bytes.toString("base64")}`;
}

export async function buildVisionToolImageUrl(raw: string): Promise<string | null> {
  const value = raw.trim();
  if (!value) return null;
  if (value.startsWith("data:image/")) return value;

  try {
    const normalized = normalizeVisionUrl(value);
    const parsed = new URL(normalized);
    const isHttp = parsed.protocol === "http:" || parsed.protocol === "https:";
    if (!isHttp) return null;
    if (isPrivateOrLocalHostname(parsed.hostname)) {
      const inline = await toInlineDataUrl(normalized);
      if (inline) return inline;
      return null; // Cannot pass private/local URL to external vision model
    }
    return normalized;
  } catch {
    return null;
  }
}

export const analyzeImageTool: Tool = {
  name: "analyzeImage",
  description:
    "使用视觉模型（如 qwen3.6-plus）对图片内容进行详细分析或针对性提问。" +
    "当需要深入理解图片细节、提取文字、分析图表、识别公式时使用。" +
    "image_urls 中填写消息里提供的图片 URL。",
  category: "read",
  parameters: {
    type: "object",
    properties: {
      image_urls: {
        type: "array",
        items: { type: "string" },
        minItems: 1,
        maxItems: 10,
        description: "要分析的图片 URL 列表（presigned URL 或 data URI）",
      },
      question: {
        type: "string",
        minLength: 1,
        maxLength: 2000,
        description: "对图片提出的具体问题，例如：「图中有哪些网络节点？」「图中的公式是什么？」",
      },
    },
    required: ["image_urls", "question"],
  },
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  async execute(args: Record<string, unknown>, _ctx: TurnContext): Promise<string> {
    const t0 = Date.now();
    const imageUrls = Array.isArray(args.image_urls)
      ? (args.image_urls as unknown[]).filter((u): u is string => typeof u === "string")
      : [];
    const question = typeof args.question === "string" ? args.question.trim() : "";

    if (imageUrls.length === 0) {
      return JSON.stringify({ error: "缺少 image_urls 参数" });
    }
    if (!question) {
      return JSON.stringify({ error: "缺少 question 参数" });
    }
    log.debug({ imageCount: imageUrls.length, question: question.slice(0, 80) }, "analyzeImage start");

    try {
      const client = getLLMClient("vision");
      const model = getVisionModel();

      const normalizedUrls = (
        await Promise.all(imageUrls.map((url) => buildVisionToolImageUrl(url)))
      ).filter((u): u is string => !!u);
      if (normalizedUrls.length === 0) {
        return JSON.stringify({ error: "未找到可用的图片 URL" });
      }

      const contentParts: OpenAI.Chat.ChatCompletionContentPart[] = [
        { type: "text", text: question },
        ...normalizedUrls.map((url) => ({
          type: "image_url" as const,
          image_url: { url },
        })),
      ];

      const resp = await client.chat.completions.create({
        model,
        messages: [{ role: "user", content: contentParts }],
        max_tokens: 1500,
      });

      const answer = resp.choices[0]?.message?.content?.trim() ?? "";
      log.debug({ durationMs: Date.now() - t0, answerLen: answer.length }, "analyzeImage done");
      return answer || JSON.stringify({ error: "视觉模型未返回内容" });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return JSON.stringify({ error: `视觉模型调用失败：${msg}` });
    }
  },
};
