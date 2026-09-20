/**
 * Agentic-RAG Inference Script
 *
 * Reads a questions JSON file, sends each question to the TS ReAct agent
 * via the /api/v1/courses/{courseId}/chat SSE endpoint, and saves the answers.
 *
 * Run from the edu-platform/ directory:
 *
 *   npx tsx scripts/eval/run_inference.ts \
 *     --input  ../tests/eval/data/ragas_custom_questions.json \
 *     --output ../tests/eval/results/vector_baseline_answers.json \
 *     --course-id c0000001-0000-4000-8000-000000000000
 *
 *   npx tsx scripts/eval/run_inference.ts \
 *     --input  ../tests/eval/data/ragas_custom_questions.json \
 *     --output ../tests/eval/results/vector_agent_answers.json \
 *     --course-id c0000002-0000-4000-8000-000000000000
 *
 *   npx tsx scripts/eval/run_inference.ts \
 *     --input  ../tests/eval/data/frames_questions.json \
 *     --output ../tests/eval/results/frames_answers.json \
 *     --course-id c0000003-0000-4000-8000-000000000000
 *
 *   # Ragas custom: course_id stored per-question
 *   npx tsx scripts/eval/run_inference.ts \
 *     --input  ../tests/eval/data/ragas_custom_questions.json \
 *     --output ../tests/eval/results/ragas_custom_answers.json \
 *     --course-id-from-input
 *
 * Auth:
 *   JWT_SECRET in .env → auto-generates eval student JWT (7 days)
 *   Or set EVAL_JWT_TOKEN=<pre-signed-token> to override
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "fs";
import { dirname, resolve } from "path";
import { SignJWT } from "jose";
import { parseB3SseEventJson } from "../../lib/agent/b3-protocol";

// ---------------------------------------------------------------------------
// Load .env (edu-platform/.env or repo-root .env)
// ---------------------------------------------------------------------------
function loadDotEnv(): void {
  const candidates = [
    resolve(process.cwd(), ".env"),
    resolve(process.cwd(), "../.env"),
  ];
  for (const envPath of candidates) {
    if (!existsSync(envPath)) continue;
    const lines = readFileSync(envPath, "utf-8").split("\n");
    for (const raw of lines) {
      const line = raw.trim();
      if (!line || line.startsWith("#")) continue;
      const eqIdx = line.indexOf("=");
      if (eqIdx < 0) continue;
      const key = line.slice(0, eqIdx).trim();
      const val = line.slice(eqIdx + 1).trim().replace(/^["']|["']$/g, "");
      if (key && !(key in process.env)) {
        process.env[key] = val;
      }
    }
    break;
  }
}
loadDotEnv();

// ---------------------------------------------------------------------------
// Model config reporting (mirrors llm-registry.ts fallback logic)
// ---------------------------------------------------------------------------
function getEnv(name: string): string | undefined {
  const v = process.env[name];
  return v && v.trim() ? v.trim() : undefined;
}

function resolveModelConfig(): {
  chat: { model: string; baseURL: string };
  vision: { model: string; baseURL: string };
  title: { model: string; baseURL: string };
  memory: { model: string; baseURL: string };
} {
  const defaultBase = getEnv("LLM_BASE_URL") ?? "(OpenAI default)";
  const defaultModel = getEnv("LLM_MODEL") ?? "deepseek-flash";
  const auxModel = getEnv("LLM_AUXILIARY_MODEL") ?? defaultModel;

  const chatBase = getEnv("LLM_CHAT_BASE_URL") ?? defaultBase;
  const chatModel = getEnv("LLM_CHAT_MODEL") ?? defaultModel;

  const visionBase = getEnv("LLM_VISION_BASE_URL") ?? defaultBase;
  const visionModel = getEnv("LLM_VISION_MODEL") ?? defaultModel;

  const titleBase = getEnv("LLM_TITLE_BASE_URL") ?? chatBase;
  const titleModel = getEnv("LLM_TITLE_MODEL") ?? getEnv("LLM_AUXILIARY_MODEL") ?? defaultModel;

  const memBase = defaultBase;
  const memModel = getEnv("LLM_AUXILIARY_MODEL") ?? defaultModel;

  return {
    chat:   { model: chatModel,   baseURL: chatBase },
    vision: { model: visionModel, baseURL: visionBase },
    title:  { model: titleModel,  baseURL: titleBase },
    memory: { model: memModel,    baseURL: memBase },
  };
}

function printModelConfig(): void {
  const cfg = resolveModelConfig();
  console.log("[models] Active LLM configuration:");
  console.log(`  chat   → model=${cfg.chat.model}   baseURL=${cfg.chat.baseURL}`);
  console.log(`  vision → model=${cfg.vision.model}  baseURL=${cfg.vision.baseURL}`);
  console.log(`  title  → model=${cfg.title.model}   baseURL=${cfg.title.baseURL}`);
  console.log(`  memory → model=${cfg.memory.model}  baseURL=${cfg.memory.baseURL}`);
}

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------
interface CliArgs {
  input: string;
  output: string;
  courseId: string | null;
  courseIdFromInput: boolean;
  baseUrl: string;
  delayMs: number;
  dryRun: boolean;
  verbose: boolean;
}

function parseArgs(): CliArgs {
  const args = process.argv.slice(2);
  const get = (flag: string): string | undefined => {
    const i = args.indexOf(flag);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const has = (flag: string): boolean => args.includes(flag);

  const input = get("--input");
  const output = get("--output");
  if (!input || !output) {
    console.error(
      "Usage: run_inference.ts --input <q.json> --output <a.json> [--course-id <uuid> | --course-id-from-input] [--verbose]"
    );
    process.exit(1);
  }

  return {
    input: resolve(input),
    output: resolve(output),
    courseId: get("--course-id") ?? null,
    courseIdFromInput: has("--course-id-from-input"),
    baseUrl: get("--base-url") ?? process.env.EVAL_BASE_URL ?? "http://localhost:3000",
    delayMs: parseInt(get("--delay-ms") ?? "500", 10),
    dryRun: has("--dry-run"),
    verbose: has("--verbose"),
  };
}

// ---------------------------------------------------------------------------
// JWT generation
// ---------------------------------------------------------------------------
async function getOrGenerateToken(): Promise<string> {
  const explicit = process.env.EVAL_JWT_TOKEN?.trim();
  if (explicit) return explicit;

  const secret = process.env.JWT_SECRET?.trim();
  if (!secret) {
    throw new Error("Set EVAL_JWT_TOKEN or JWT_SECRET in .env");
  }

  const userId   = process.env.EVAL_USER_ID?.trim()       ?? "e0000002-0000-4000-8000-000000000000";
  const username = process.env.EVAL_USER_USERNAME?.trim() ?? "eval_student";
  const role     = process.env.EVAL_USER_ROLE?.trim()     ?? "STUDENT";
  const issuer   = process.env.JWT_ISSUER?.trim()         ?? "edu-platform";

  const key = new TextEncoder().encode(secret);
  const now = Math.floor(Date.now() / 1000);
  const token = await new SignJWT({ username, role })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(userId)
    .setIssuedAt(now)
    .setExpirationTime(now + 86400 * 7) // 7 days
    .setIssuer(issuer)
    .sign(key);

  console.log("[auth] Auto-generated eval JWT for user:", userId);
  return token;
}

// ---------------------------------------------------------------------------
// SSE chat endpoint
// ---------------------------------------------------------------------------
async function callChatEndpoint(
  baseUrl: string,
  courseId: string,
  question: string,
  token: string,
  verbose: boolean,
): Promise<{
  answer: string;
  toolCalls: Array<{ name: string; input?: unknown; output?: unknown; success?: boolean; duration_ms?: number }>;
  tokens: number;
  execTimeMs: number;
  streamError?: string;
}> {
  const url = `${baseUrl}/api/v1/courses/${courseId}/chat`;

  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      message: question,
      trim_history_to: 0,
      eval_mode: true,
    }),
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`HTTP ${response.status}: ${body.slice(0, 400)}`);
  }

  const text = await response.text();
  let answer = "";
  const toolCalls: Array<{ name: string; input?: unknown; output?: unknown; success?: boolean; duration_ms?: number }> = [];
  let tokens = 0;
  let execTimeMs = 0;
  let streamError: string | undefined;

  for (const line of text.split("\n")) {
    if (!line.startsWith("data: ")) continue;
    const payload = line.slice(6).trim();
    if (!payload || payload === "[DONE]") continue;
    const evt = parseB3SseEventJson(payload);
    if (!evt) {
      if (verbose) console.error(`  [sse] invalid B3 event: ${payload.slice(0, 120)}`);
      continue;
    }
    if (evt.type === "text") {
      answer += evt.content ?? "";
    } else if (evt.type === "tool_call") {
      if (verbose) console.log(`  [tool→] ${evt.name}`);
      toolCalls.push({ name: evt.name ?? "unknown", input: evt.input });
    } else if (evt.type === "tool_result") {
      const last = toolCalls.length > 0 ? toolCalls[toolCalls.length - 1] : null;
      if (last) {
        last.output = evt.output;
        last.success = evt.success;
        last.duration_ms = evt.duration_ms;
      }
      if (verbose || evt.success === false) {
        const statusIcon = evt.success === false ? "✗" : "✓";
        const dur = evt.duration_ms !== undefined ? ` ${evt.duration_ms}ms` : "";
        const outSnippet = evt.output !== undefined
          ? JSON.stringify(evt.output).slice(0, 200)
          : "(no output)";
        console.log(`  [tool←] ${statusIcon} ${evt.name ?? last?.name ?? "?"}${dur}: ${outSnippet}`);
      }
      // Surface rag-service errors prominently
      if (evt.success === false) {
        const errDetail = typeof evt.output === "string" ? evt.output
          : evt.output && typeof evt.output === "object" ? JSON.stringify(evt.output).slice(0, 300)
          : "(no detail)";
        console.error(`  [rag-error] tool=${evt.name ?? last?.name ?? "?"} → ${errDetail}`);
      }
    } else if (evt.type === "done") {
      tokens = evt.tokens ?? 0;
      execTimeMs = evt.exec_time_ms ?? 0;
      if (evt.error) {
        streamError = evt.error;
        console.error(`  [stream-error] ${evt.error}`);
      }
    }
  }

  return { answer: answer.trim(), toolCalls, tokens, execTimeMs, streamError };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
interface Question {
  id?: string;
  question?: string;
  Prompt?: string;
  gold_answer?: string;
  course_id?: string;
  [key: string]: unknown;
}

interface AnswerRecord extends Question {
  generated_answer: string;
  tool_calls: Array<{ name: string; input?: unknown; output?: unknown }>;
  tokens: number;
  exec_time_ms: number;
  error?: string;
}

async function main(): Promise<void> {
  const cli = parseArgs();
  printModelConfig();

  const rawQuestions: Question[] = JSON.parse(readFileSync(cli.input, "utf-8"));
  console.log(`[run_inference] ${rawQuestions.length} questions from ${cli.input}`);

  if (cli.dryRun) {
    console.log("[dry-run] Done.");
    process.exit(0);
  }

  const token = await getOrGenerateToken();
  mkdirSync(dirname(cli.output), { recursive: true });

  // Resume support
  let existingAnswers: AnswerRecord[] = [];
  try {
    existingAnswers = JSON.parse(readFileSync(cli.output, "utf-8"));
    console.log(`[resume] ${existingAnswers.length} existing answers found.`);
  } catch {
    /* fresh run */
  }
  const answeredIds = new Set(existingAnswers.map((a) => String(a.id ?? a.question)));
  const answers: AnswerRecord[] = [...existingAnswers];
  let processed = 0;
  let errors = 0;

  const pending = rawQuestions.filter(
    (q) => !answeredIds.has(String(q.id ?? q.question ?? q.Prompt))
  );
  console.log(`[run_inference] ${pending.length} questions to process.`);

  for (const q of pending) {
    const questionText = String(q.question ?? q.Prompt ?? "");
    const courseId = cli.courseIdFromInput
      ? String(q.course_id ?? "")
      : cli.courseId ?? "";

    if (!courseId) {
      console.error(`[skip] No course_id for: ${questionText.slice(0, 60)}`);
      answers.push({
        ...q,
        generated_answer: "",
        tool_calls: [],
        tokens: 0,
        exec_time_ms: 0,
        error: "no course_id",
      });
      errors++;
      continue;
    }

    process.stdout.write(`[${processed + 1}/${pending.length}] ${questionText.slice(0, 55)}... `);

    try {
      const r = await callChatEndpoint(cli.baseUrl, courseId, questionText, token, cli.verbose);
      const failedTools = r.toolCalls.filter((t) => t.success === false).length;
      const streamErrSuffix = r.streamError ? ` ⚠ stream:${r.streamError}` : "";
      const toolErrSuffix = failedTools > 0 ? ` ⚠ ${failedTools} tool-err` : "";
      answers.push({
        ...q,
        generated_answer: r.answer,
        tool_calls: r.toolCalls,
        tokens: r.tokens,
        exec_time_ms: r.execTimeMs,
        ...(r.streamError ? { error: r.streamError } : {}),
      });
      process.stdout.write(`✓ (${r.tokens}tok, ${r.toolCalls.length} tools${toolErrSuffix}${streamErrSuffix})\n`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      process.stdout.write(`✗ ${msg}\n`);
      answers.push({
        ...q,
        generated_answer: "",
        tool_calls: [],
        tokens: 0,
        exec_time_ms: 0,
        error: msg,
      });
      errors++;
    }

    processed++;
    writeFileSync(cli.output, JSON.stringify(answers, null, 2), "utf-8");
    if (cli.delayMs > 0) {
      await new Promise((r) => setTimeout(r, cli.delayMs));
    }
  }

  writeFileSync(cli.output, JSON.stringify(answers, null, 2), "utf-8");
  console.log(`\n[done] ${processed} processed, ${errors} errors → ${cli.output}`);
}

main().catch((err) => {
  console.error("[fatal]", err);
  process.exit(1);
});
